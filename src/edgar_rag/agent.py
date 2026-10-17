"""The agent loop: the LLM decides which tools to call, we run them, until it answers.

    question -> LLM -> tool calls? --yes--> run tools, add results -> LLM -> ...
                            \\--no---> final answer (with [n] citations)
"""

import json
import re

from pydantic import BaseModel, ValidationError

from edgar_rag import companies
from edgar_rag.answer import parse_citations
from edgar_rag.llm import complete
from edgar_rag.prompts import AGENT_SYSTEM
from edgar_rag.tools import TOOLS, Source, Sources

MAX_STEPS = 6  # LLM calls per question; stops runaway loops
# Up to this many companies, list them all in the prompt (~15 tokens each). Beyond it the list
# would cost thousands of tokens on every call, so the agent looks companies up instead.
LIST_COMPANIES_UP_TO = 50

FINAL_STEP_NUDGE = (
    "You have reached the tool-call limit. Answer the question now using only the tool results "
    "above, with citations. Say what you could not find."
)
# Tool-call syntax the model sometimes writes into its text instead of a structured call.
TOOL_MARKUP_RE = re.compile(r"DSML|<｜|invoke name=|</?tool_call", re.IGNORECASE)


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
    malformed: bool = False  # the model wrote tool-call markup instead of an answer


def available_companies() -> str:
    """Describe what's indexed, e.g. 'AAPL (Apple Inc., FY2025)', or just a count when large."""
    indexed = companies.indexed_companies()
    if not indexed:
        return "none indexed"
    if len(indexed) > LIST_COMPANIES_UP_TO:
        return (
            f"{len(indexed)} companies are indexed. Use find_company to get a ticker and its "
            "latest fiscal year."
        )
    return "; ".join(f"{c.ticker} ({c.name}, FY{c.fiscal_year})" for c in indexed)


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
        last = step == MAX_STEPS
        if last:
            # Force an answer. Removing the tools entirely made the model write tool calls out
            # as text (seen in evals), so keep them listed but disallow calling them.
            messages.append({"role": "user", "content": FINAL_STEP_NUDGE})
        response = complete(messages, tools=schemas, tool_choice="none" if last else None)
        if response.usage:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
        message = response.choices[0].message

        if not message.tool_calls or last:
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

    # Never show the user raw tool-call markup; report it instead.
    malformed = bool(TOOL_MARKUP_RE.search(text))
    if malformed:
        text = (
            "I couldn't complete this answer: the model returned an incomplete tool request "
            "instead of an answer. Try rephrasing or narrowing the question."
        )

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
        malformed=malformed,
    )
