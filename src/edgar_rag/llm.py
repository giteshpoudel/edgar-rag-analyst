"""Thin wrapper over the LLM API. DeepSeek is OpenAI-compatible, so we use the openai SDK."""

from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from edgar_rag import config


class LlmResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=config.llm_api_key(), base_url=config.llm_base_url())


def complete(messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1500):
    """One raw API call, optionally offering tools. Returns the SDK response object."""
    kwargs = {"tools": tools} if tools else {}
    return _client().chat.completions.create(
        model=config.llm_model(),
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        **kwargs,
    )


def chat(messages: list[dict], max_tokens: int = 800) -> LlmResponse:
    # temperature=0 makes answers as repeatable as possible, which matters for evals.
    response = _client().chat.completions.create(
        model=config.llm_model(),
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    usage = response.usage
    return LlmResponse(
        text=(response.choices[0].message.content or "").strip(),
        model=response.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
