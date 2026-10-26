"""HTTP API for the public website demo (giteshpoudel.com/demo/edgar).

    browser -> website /api/edgar (Vercel: origin check, Turnstile, rate limits)
            -> POST /ask here, with a shared Bearer secret -> agent -> DeepSeek

The browser never calls this service directly, and the DeepSeek key exists only here. Run it
with DEMO_MODE=1 so the agent only touches the 15 indexed companies and never calls the SEC.

    uvicorn edgar_rag.server:app --port 8080

The response models mirror the website's zod schema (portfolio-website/src/lib/edgar-contract.ts);
anything that doesn't fit is rejected there with a 502, so the limits below must match it.
"""

import asyncio
import hmac
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from edgar_rag import config, tracing

MAX_QUESTION_CHARS = 400
MIN_QUESTION_CHARS = 5
REQUEST_TIMEOUT_SECONDS = 40.0  # the website gives up at 45 s
MAX_IN_FLIGHT = 4  # matches Cloud Run --concurrency 4; more waiting requests get a 429
SEC_URL = re.compile(r"^https://(www\.)?sec\.gov/")
CONTROL_CHARS = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F]")  # same as the website's cleaning

log = logging.getLogger("edgar_rag.server")


# --- Contract (mirror of the website's zod schema) -----------------------------------------


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def clean(cls, v: str) -> str:
        v = CONTROL_CHARS.sub("", v).strip()
        if not MIN_QUESTION_CHARS <= len(v) <= MAX_QUESTION_CHARS:
            raise ValueError(f"question must be {MIN_QUESTION_CHARS}-{MAX_QUESTION_CHARS} characters")
        return v


class SourceOut(BaseModel):
    n: int
    label: str = Field(max_length=500)
    url: str | None = Field(None, max_length=500)  # omitted (not null) when unknown: zod rejects null


class StepOut(BaseModel):
    tool: str = Field(max_length=60)
    args: str = Field(max_length=500)
    result: str = Field(max_length=800)


class UsageOut(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


class AskResponse(BaseModel):
    answer: str = Field(max_length=12_000)
    sources: list[SourceOut] = Field(max_length=30)
    steps: list[StepOut] = Field(max_length=30)
    usage: UsageOut
    refused: bool


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def to_response(result, llm_calls: int) -> AskResponse:
    """Map the agent's answer to the contract, cutting every field to its limit."""
    sources = [
        SourceOut(n=s.n, label=_cut(s.label, 500),
                  url=s.url if s.url and SEC_URL.match(s.url) and len(s.url) <= 500 else None)
        for s in result.sources[:30]
    ]
    steps = [
        StepOut(tool=_cut(c.name, 60),
                args=_cut(", ".join(f"{k}={v!r}" for k, v in c.arguments.items()), 500),
                result=_cut(c.result, 800))
        for c in result.tool_calls[:30]
    ]
    return AskResponse(
        answer=_cut(result.text, 12_000),
        sources=sources,
        steps=steps,
        usage=UsageOut(llm_calls=llm_calls, prompt_tokens=result.prompt_tokens,
                       completion_tokens=result.completion_tokens),
        # An answer citing nothing is a refusal (off-topic, or a company that isn't indexed).
        # Refusals that cite what *is* available ("FY2027 isn't out; FY2025 was ... [1]") count
        # as answers.
        refused=not result.cited,
    )


# --- App -----------------------------------------------------------------------------------


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _authorized(request: Request) -> bool:
    secret = config.api_secret()
    header = request.headers.get("authorization", "")
    if not secret or not header.startswith("Bearer "):
        return False
    # Constant-time comparison, so response timing doesn't reveal how much of a guess matched.
    return hmac.compare_digest(header[len("Bearer "):].encode(), secret.encode())


def _warm_up() -> None:
    """Load everything the first question needs, so no visitor pays for it."""
    from edgar_rag import companies
    from edgar_rag.embeddings import embed_query
    from edgar_rag.index import TABLE_NAME, connect
    from edgar_rag.tools import _facts

    embed_query("warm up")
    connect().open_table(TABLE_NAME).count_rows()
    for c in companies.indexed_companies():
        _facts(c.ticker)


def create_app(warm_up: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not config.demo_mode():
            log.warning("DEMO_MODE is off: tools can reach the SEC and any ticker")
        if not config.api_secret():
            log.warning("EDGAR_API_SECRET is not set: every /ask request will get 401")
        if warm_up:
            started = time.monotonic()
            await asyncio.to_thread(_warm_up)
            log.info(json.dumps({"event": "warm_up", "seconds": round(time.monotonic() - started, 1)}))
        yield

    app = FastAPI(title="EDGAR Analyst demo API", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.in_flight = 0

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        return _error(400, "Invalid request: send {\"question\": \"5-400 characters\"}.")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
    async def ask(request: Request):
        if not _authorized(request):
            return _error(401, "Unauthorized.")
        try:
            body = AskRequest.model_validate(await request.json())
        except ValueError:  # bad JSON, or a question outside 5-400 characters (pydantic's error is a ValueError)
            return _error(400, "Invalid request: send {\"question\": \"5-400 characters\"}.")
        if app.state.in_flight >= MAX_IN_FLIGHT:
            return _error(429, "Busy, try again shortly.")

        from edgar_rag.agent import aask

        app.state.in_flight += 1
        started = time.monotonic()
        status, record = 500, {"question_chars": len(body.question)}
        try:
            with tracing.span("api.ask", "CHAIN", **{"edgar.question_chars": len(body.question)}):
                trace_id = tracing.current_trace_id()
                result = await asyncio.wait_for(aask(body.question), REQUEST_TIMEOUT_SECONDS)
            usage = tracing.COLLECTOR.pop(trace_id)
            response = to_response(result, usage.llm_calls)
            status = 200
            record |= {"tool_calls": len(result.tool_calls), "llm_calls": usage.llm_calls,
                       "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                       "cost_usd": round(usage.cost_usd, 6) if usage.cost_known else None,
                       "refused": response.refused}
            return response
        except TimeoutError:
            record["error"] = "timeout"
            return _error(500, "The analyst took too long.")
        except Exception as e:  # never send internals to the client
            record["error"] = type(e).__name__
            log.exception("ask failed")
            return _error(500, "The analyst couldn't answer that.")
        finally:
            app.state.in_flight -= 1
            # Question text, the secret and API keys are never logged.
            record |= {"status": status, "seconds": round(time.monotonic() - started, 2)}
            log.info(json.dumps({"event": "ask", **record}))

    return app


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one JSON line per question, not per HTTP call
app = create_app(warm_up=os.getenv("EDGAR_WARM_UP", "1") != "0")
