"""Retries with exponential backoff, for the LLM API and SEC requests (W5).

Only transient failures are retried: timeouts, dropped connections, 429 (rate limited) and 5xx
(server errors). A 400 or 401 means the request itself is wrong, so retrying would just fail
again more slowly.

Wait before attempt n+1 = random between 0 and min(cap, base * 2^n), "full jitter": when many
concurrent questions hit the same error, random waits spread their retries out instead of
sending them all back at the same moment. A server's Retry-After header, when present, wins.

Each retry is recorded as an event on the current span, so it shows up in the trace.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
import openai
from opentelemetry import trace

T = TypeVar("T")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
ATTEMPTS = 4  # 1 try + 3 retries
BASE_SECONDS = 1.0
CAP_SECONDS = 30.0


def _status(exc: BaseException) -> int | None:
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError)):
        return True  # includes timeouts
    status = _status(exc)
    return status in RETRY_STATUS


def retry_after(exc: BaseException) -> float | None:
    """Seconds from a Retry-After header (the numeric form), if the server sent one."""
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None  # the HTTP-date form; fall back to our own backoff


def backoff(attempt: int, exc: BaseException | None = None, base: float = BASE_SECONDS,
            cap: float = CAP_SECONDS) -> float:
    """Seconds to wait after failed attempt number `attempt` (0-based)."""
    hinted = retry_after(exc) if exc is not None else None
    if hinted is not None:
        return min(cap, hinted)
    return random.uniform(0, min(cap, base * 2**attempt))


def _record(attempt: int, exc: BaseException, wait: float) -> None:
    trace.get_current_span().add_event("retry", {
        "attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}"[:300], "wait_seconds": round(wait, 2),
    })


def call(fn: Callable[[], T], attempts: int = ATTEMPTS, sleep: Callable[[float], None] = time.sleep) -> T:
    """Run fn, retrying transient failures. The last error is raised if every attempt fails."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1 or not is_retryable(exc):
                raise
            wait = backoff(attempt, exc)
            _record(attempt, exc, wait)
            sleep(wait)
    raise AssertionError("unreachable")


async def acall(fn: Callable[[], Awaitable[T]], attempts: int = ATTEMPTS) -> T:
    """Async version of call: waits with asyncio.sleep, so other questions keep running."""
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if attempt == attempts - 1 or not is_retryable(exc):
                raise
            wait = backoff(attempt, exc)
            _record(attempt, exc, wait)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")
