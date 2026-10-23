"""Thin wrapper over the LLM API. DeepSeek is OpenAI-compatible, so we use the openai SDK."""

import asyncio
import weakref
from functools import lru_cache

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from edgar_rag import config, retry, tracing

TIMEOUT_SECONDS = 120.0  # a reasoning model's long answer can take a while; a hung call can't


class LlmResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False  # hit max_tokens before finishing


# max_retries=0: the SDK would otherwise retry twice on its own, invisibly. retry.py does it
# instead, so every retry is an event on the trace with the error and the wait.
@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=config.llm_api_key(), base_url=config.llm_base_url(),
                  max_retries=0, timeout=TIMEOUT_SECONDS)


# An async client's connections belong to the event loop that opened them, so each loop gets
# its own (reusing one across asyncio.run() calls fails with "Event loop is closed").
_async_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI]" = weakref.WeakKeyDictionary()


def _aclient() -> AsyncOpenAI:
    loop = asyncio.get_running_loop()
    if loop not in _async_clients:
        _async_clients[loop] = AsyncOpenAI(api_key=config.llm_api_key(), base_url=config.llm_base_url(),
                                           max_retries=0, timeout=TIMEOUT_SECONDS)
    return _async_clients[loop]


def _request(messages: list[dict], tools, max_tokens: int, model: str, tool_choice) -> tuple[dict, dict]:
    """API arguments, and the parameters recorded on the span."""
    kwargs: dict = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    params = {"temperature": 0, "max_tokens": max_tokens, "tool_choice": tool_choice,
              "tools": [t["function"]["name"] for t in tools or []]}
    return kwargs, params


# deepseek-flash reasons before answering, and those hidden tokens count against max_tokens.
# Limits that are too low end the call with finish_reason "length" and an empty answer.
def complete(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4000,
    model: str | None = None,
    tool_choice: str | None = None,
    span_name: str = "llm",
):
    """One API call (retried on transient errors), recorded as an LLM span with tokens, cost
    and latency. Returns the SDK response object."""
    model = model or config.llm_model()
    kwargs, params = _request(messages, tools, max_tokens, model, tool_choice)
    with tracing.span(span_name, "LLM") as span:
        response = retry.call(lambda: _client().chat.completions.create(**kwargs))
        tracing.record_llm(span, model, messages, response, params)
    return response


async def acomplete(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4000,
    model: str | None = None,
    tool_choice: str | None = None,
    span_name: str = "llm",
):
    """Async version of complete: while one question waits on the API, others keep running."""
    model = model or config.llm_model()
    kwargs, params = _request(messages, tools, max_tokens, model, tool_choice)
    with tracing.span(span_name, "LLM") as span:
        response = await retry.acall(lambda: _aclient().chat.completions.create(**kwargs))
        tracing.record_llm(span, model, messages, response, params)
    return response


def chat(messages: list[dict], max_tokens: int = 4000) -> LlmResponse:
    # temperature=0 (set in complete) makes answers as repeatable as possible, which matters for evals.
    response = complete(messages, max_tokens=max_tokens)
    usage = response.usage
    choice = response.choices[0]
    return LlmResponse(
        text=(choice.message.content or "").strip(),
        model=response.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        truncated=choice.finish_reason == "length",
    )
