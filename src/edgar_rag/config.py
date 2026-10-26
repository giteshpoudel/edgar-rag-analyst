"""Settings loaded from environment variables (and a local .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def sec_user_agent() -> str:
    """The SEC requires a User-Agent with a name and contact email."""
    value = os.getenv("SEC_USER_AGENT", "").strip()
    if not value or "your@email.com" in value:
        raise RuntimeError(
            "Set SEC_USER_AGENT in .env, e.g. SEC_USER_AGENT=\"Jane Doe jane@example.com\""
        )
    return value


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env (see .env.example)")
    return value


def llm_api_key() -> str:
    return _required("DEEPSEEK_API_KEY")


def llm_model() -> str:
    return _required("LLM_MODEL")


def judge_model() -> str:
    """Model that grades eval answers. Defaults to the answering model."""
    return os.getenv("JUDGE_MODEL", "").strip() or llm_model()


def llm_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "https://api.deepseek.com")


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data"))


# --- Retrieval (W2). Defaults chosen by the ablation (evals/results/v2/retrieval_test.md):
# hybrid beat vector by +0.23 recall@5 and +4.0 points end-to-end on test (both significant);
# the bge reranker added no accuracy over hybrid (+0.0, CI [-1.1, +1.1]) for +1.7 s p50, so it
# stays off by default.


RETRIEVAL_MODES = ("vector", "keyword", "hybrid")
RERANKERS = ("none", "minilm", "bge")


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Read a setting and reject unknown values immediately: a typo should stop a run before
    the first question, not quietly break every search (this happened once: 40 minutes of
    eval runs with RETRIEVAL_MODE="hybrid none")."""
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise RuntimeError(f"{name}={value!r} is not valid; use one of: {', '.join(allowed)}")
    return value


def retrieval_mode() -> str:
    """vector (meaning), keyword (BM25 full-text) or hybrid (both, fused by reciprocal rank)."""
    return _choice("RETRIEVAL_MODE", "hybrid", RETRIEVAL_MODES)


def reranker() -> str:
    """none, minilm or bge: a cross-encoder that re-orders the top candidates."""
    return _choice("RERANKER", "none", RERANKERS)


def grounding_guard() -> str:
    """on: if an answer states numbers no tool result supports, the agent gets one repair pass."""
    return _choice("GROUNDING_GUARD", "on", ("on", "off"))


def rerank_candidates() -> int:
    """How many first-stage results the reranker re-orders."""
    return int(os.getenv("RERANK_CANDIDATES", "30"))


# --- Tracing and cost (W4) ------------------------------------------------------------------

# USD per 1M tokens: (input cache hit, input cache miss, output). From
# https://api-docs.deepseek.com/quick_start/pricing (checked 2026-10-22). DeepSeek charges half
# these rates off-peak; reports use the full (peak) rate so runs at different hours compare.
# Reasoning tokens are part of completion_tokens and billed as output.
PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.006, 0.30, 1.20),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
}


def price(model: str) -> tuple[float, float, float] | None:
    """Per-1M-token prices for a model, or None if it isn't in the table (cost is then unknown,
    never silently zero)."""
    return PRICES.get(model)


def tracing() -> str:
    """off: spans are still recorded in-process (eval cost and latency come from them) but not
    exported. phoenix: also send them to a Phoenix server (PHOENIX_ENDPOINT)."""
    return _choice("TRACING", "off", ("off", "phoenix"))


def phoenix_endpoint() -> str:
    return os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")


# --- Public demo (website backend) ----------------------------------------------------------


def demo_mode() -> bool:
    """DEMO_MODE=1 for the public demo server: tools only touch the 15 indexed companies, no
    live SEC requests (cached data only), extra prompt rules and a per-request token budget.
    Off by default so the CLI, evals and CI gate behave exactly as before."""
    return _choice("DEMO_MODE", "0", ("0", "1")) == "1"


def request_token_budget() -> int | None:
    """Most prompt + completion tokens one question may use before the agent must answer.
    Default 100,000 in demo mode (eval p99 was ~53,000, max ~79,000), none otherwise."""
    raw = os.getenv("REQUEST_TOKEN_BUDGET", "").strip()
    if raw:
        return int(raw)
    return 100_000 if demo_mode() else None


def api_secret() -> str | None:
    """Bearer secret the website sends; the server refuses every /ask call when it's unset."""
    return os.getenv("EDGAR_API_SECRET", "").strip() or None
