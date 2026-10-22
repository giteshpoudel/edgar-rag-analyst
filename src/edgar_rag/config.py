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


# --- Retrieval (W2). Defaults keep v1 behaviour until the ablation picks a winner. ---------


def retrieval_mode() -> str:
    """vector (meaning), keyword (BM25 full-text) or hybrid (both, fused by reciprocal rank)."""
    return os.getenv("RETRIEVAL_MODE", "vector").strip().lower()


def reranker() -> str:
    """none, minilm or bge: a cross-encoder that re-orders the top candidates."""
    return os.getenv("RERANKER", "none").strip().lower()


def rerank_candidates() -> int:
    """How many first-stage results the reranker re-orders."""
    return int(os.getenv("RERANK_CANDIDATES", "30"))
