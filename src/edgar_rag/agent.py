"""The agent loop: the LLM decides which tools to call, we run them, until it answers.

    question -> LLM -> tool calls? --yes--> run tools, add results -> LLM -> ...
                            \\--no---> final answer (with [n] citations)
"""

import json

from pydantic import BaseModel, ValidationError

from edgar_rag import config
from edgar_rag.answer import parse_citations
from edgar_rag.chunking import Chunk
from edgar_rag.llm import complete
from edgar_rag.prompts import AGENT_SYSTEM
from edgar_rag.tools import TOOLS, Source, Sources

MAX_STEPS = 6  # LLM calls per question; stops runaway loops


class ToolCall(BaseModel):
    step: int
    name: str
    arguments: dict
    result: str
    error: bool


class AgentAnswer(BaseModel):
    question: str
    text: str
    sources: list[Source]
    cited: list[int]
    invalid_citations: list[int]
    tool_calls: list[ToolCall]
    steps: int
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False  # final answer hit max_tokens


def available_companies() -> str:
    """Describe what's indexed, e.g. 'AAPL (Apple Inc., latest 10-K fiscal year 2025)'."""
    entries = []
    for path in sorted((config.data_dir() / "processed").glob("*/*.chunks.jsonl")):
        first = Chunk.model_validate_json(path.open().readline())
        entries.append(
            f"{first.ticker} ({first.company}, latest {first.form} fiscal year {first.report_date[:4]})"
        )
    return "; ".join(entries) or "none indexed"


def run_tool(name: str, raw_args: str, sources: Sources) -> tuple[dict, str, bool]:
    """Validate the LLM's arguments with the tool's Pydantic model, then run it."""
    try:
        args_dict = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {}, "ERROR: arguments were not valid JSON", True
    tool = TOOLS.get(name)
    if not tool:
        return args_dict, f"ERROR: unknown tool '{name}'", True
    try:
        args = tool.args_model.model_validate(args_dict)
    except ValidationError as e:
        # Sent back to the LLM so it can fix its call.
        return args_dict, f"ERROR: invalid arguments: {e.errors(include_url=False)}", True
    result = tool.fn(args, sources)
    return args_dict, result, result.startswith("ERROR")


def ask(question: str) -> AgentAnswer:
    sources = Sources()
    tool_calls: list[ToolCall] = []
    schemas = [t.schema_for_llm() for t in TOOLS.values()]
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM.format(companies=available_companies())},
        {"role": "user", "content": question},
    ]
    prompt_tokens = completion_tokens = 0
    text = ""
    truncated = False

    for step in range(1, MAX_STEPS + 1):
        # On the last step, offer no tools so the model must answer with what it has.
        response = complete(messages, tools=schemas if step < MAX_STEPS else None)
        if response.usage:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
        message = response.choices[0].message

        if not message.tool_calls:
            text = (message.content or "").strip()
            truncated = response.choices[0].finish_reason == "length"
            break

        # Echo the assistant's tool request back into the history, then add each result.
        assistant = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump(include={"id", "type", "function"}) for tc in message.tool_calls],
        }
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)

        for tc in message.tool_calls:
            args, result, error = run_tool(tc.function.name, tc.function.arguments, sources)
            tool_calls.append(
                ToolCall(step=step, name=tc.function.name, arguments=args, result=result, error=error)
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    cited, invalid = parse_citations(text, len(sources.items))
    return AgentAnswer(
        question=question,
        text=text,
        sources=sources.items,
        cited=cited,
        invalid_citations=invalid,
        tool_calls=tool_calls,
        steps=step,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        truncated=truncated,
    )
