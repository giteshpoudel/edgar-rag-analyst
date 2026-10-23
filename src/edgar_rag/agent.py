"""The agent loop: the LLM decides which tools to call, we run them, until it answers.

    question -> LLM -> tool calls? --yes--> run tools, add results -> LLM -> ...
                            \\--no---> final answer (with [n] citations)
"""

import asyncio
import json
import re

from opentelemetry import trace
from pydantic import BaseModel, ValidationError

from edgar_rag import companies, config, tracing
from edgar_rag.answer import parse_citations
from edgar_rag.grounding import check as check_grounding
from edgar_rag.llm import acomplete
from edgar_rag.prompts import AGENT_SYSTEM
from edgar_rag.tools import TOOLS, Source, Sources, StagedSources

MAX_STEPS = 6  # LLM calls per question; stops runaway loops
# Up to this many companies, list them all in the prompt (~15 tokens each). Beyond it the list
# would cost thousands of tokens on every call, so the agent looks companies up instead.
LIST_COMPANIES_UP_TO = 50

REPAIR_STEPS = 3  # LLM calls allowed for the one repair pass
REPAIR_PROMPT = (
    "These numbers in your answer don't appear in any tool result and can't be recomputed from "
    "them: {numbers}. Revise the answer: get each number from a tool (use calculate for any "
    "derived figure) or remove it. Keep the citations and answer the original question."
)

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
    # Numbers the grounding guard flagged in the first answer (empty if none, or guard off).
    unsupported_before_repair: list[str] = []


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


async def _run_tools(tool_calls_requested, sources: Sources, step: int) -> list[ToolCall]:
    """Run every tool call from one model turn at the same time (W5), each in a worker thread
    (the tools are ordinary blocking code). Citations are numbered afterwards, in the order the
    model requested the calls, so the numbering doesn't depend on which finished first."""
    batch = len(tool_calls_requested)
    staged = [StagedSources() for _ in tool_calls_requested]
    spans = [
        tracing.start_span(f"tool.{tc.function.name}", "TOOL", **{
            "tool.name": tc.function.name, "input.value": tc.function.arguments,
            "edgar.step": step, "edgar.parallel_calls": batch,
        })
        for tc in tool_calls_requested
    ]

    def work(tc, span, st):
        with trace.use_span(span, end_on_exit=False):  # spans the tool creates nest under it
            return run_tool(tc.function.name, tc.function.arguments, st)

    try:
        outputs = await asyncio.gather(*(
            asyncio.to_thread(work, tc, span, st) for tc, span, st in zip(tool_calls_requested, spans, staged)
        ))
        calls = []
        for tc, span, st, (args, result, error) in zip(tool_calls_requested, spans, staged, outputs):
            result = st.commit(result, sources)
            tracing.set_attributes(span, {"tool.parameters": args, "output.value": result})
            if error:
                tracing.mark_error(span, result[:200])
            calls.append(ToolCall(step=step, name=tc.function.name, arguments=args, result=result, error=error))
        return calls
    finally:
        for span in spans:
            span.end()


async def _loop(messages: list[dict], schemas: list[dict], sources: Sources, tool_calls: list[ToolCall],
                max_steps: int, first_step: int = 1) -> tuple[str, bool, int, int, int]:
    """Call the LLM and run the tools it asks for, until it answers or hits max_steps.
    Returns (text, truncated, last step number, prompt tokens, completion tokens)."""
    prompt_tokens = completion_tokens = 0
    text, truncated, step = "", False, first_step
    for i in range(max_steps):
        step = first_step + i
        last = i == max_steps - 1
        if last:
            # Force an answer. Removing the tools entirely made the model write tool calls out
            # as text (seen in evals), so keep them listed but disallow calling them.
            messages.append({"role": "user", "content": FINAL_STEP_NUDGE})
        response = await acomplete(messages, tools=schemas, tool_choice="none" if last else None)
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

        calls = await _run_tools(message.tool_calls, sources, step)
        tool_calls.extend(calls)
        for tc, call in zip(message.tool_calls, calls):
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": call.result})
    return text, truncated, step, prompt_tokens, completion_tokens


def ask(question: str) -> AgentAnswer:
    """Answer one question (blocking). Inside async code, await aask() instead."""
    return asyncio.run(aask(question))


async def aask(question: str) -> AgentAnswer:
    with tracing.span("agent", "AGENT", **{"input.value": question}) as span:
        result = await _ask(question)
        tracing.set_attributes(span, {
            "output.value": result.text, "edgar.steps": result.steps,
            "edgar.tool_calls": len(result.tool_calls), "edgar.malformed": result.malformed,
            "edgar.repaired": bool(result.unsupported_before_repair),
        })
        return result


async def _ask(question: str) -> AgentAnswer:
    sources = Sources()
    tool_calls: list[ToolCall] = []
    schemas = [t.schema_for_llm() for t in TOOLS.values()]
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM.format(companies=available_companies())},
        {"role": "user", "content": question},
    ]
    text, truncated, step, prompt_tokens, completion_tokens = await _loop(messages, schemas, sources, tool_calls, MAX_STEPS)

    # Number grounding guard (W3): if the answer states numbers no tool result supports, give the
    # model one repair pass, with tools, to source or remove them.
    unsupported_before: list[str] = []
    if config.grounding_guard() == "on" and text and not TOOL_MARKUP_RE.search(text):
        with tracing.span("grounding.check", "CHAIN", **{"input.value": text}) as span:
            report = check_grounding(text, "\n".join(c.result for c in tool_calls), question)
            unsupported_before = [c.raw for c in report.unsupported]
            tracing.set_attributes(span, {"edgar.numbers_stated": len(report.claims),
                                          "edgar.unsupported": unsupported_before})
        if unsupported_before:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": REPAIR_PROMPT.format(numbers=", ".join(unsupported_before))})
            repaired, truncated, step, p_tok, c_tok = await _loop(
                messages, schemas, sources, tool_calls, REPAIR_STEPS, first_step=step + 1
            )
            prompt_tokens += p_tok
            completion_tokens += c_tok
            if repaired:
                text = repaired

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
        unsupported_before_repair=unsupported_before,
    )
