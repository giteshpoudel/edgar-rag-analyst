"""Thin wrapper over the LLM API. DeepSeek is OpenAI-compatible, so we use the openai SDK."""

from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from edgar_rag import config, tracing


class LlmResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False  # hit max_tokens before finishing


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=config.llm_api_key(), base_url=config.llm_base_url())


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
    """One raw API call, optionally offering tools, recorded as an LLM span (tokens, cost,
    latency). Returns the SDK response object."""
    model = model or config.llm_model()
    kwargs: dict = {"tools": tools} if tools else {}
    if tools and tool_choice:
        kwargs["tool_choice"] = tool_choice
    params = {"temperature": 0, "max_tokens": max_tokens, "tool_choice": tool_choice,
              "tools": [t["function"]["name"] for t in tools or []]}
    with tracing.span(span_name, "LLM") as span:
        response = _client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            **kwargs,
        )
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
