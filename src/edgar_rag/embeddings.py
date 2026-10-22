"""Local embedding model (runs on CPU, no API needed)."""

import threading

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dimensional vectors, max 512 tokens

# bge models are trained to embed search queries with this prefix (documents get none).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Loading or running torch models from several threads at once can crash the process (seen as
# a segfault during parallel evals, and as a Metal "command encoder" abort when the embedder and
# reranker ran together on the Apple GPU). One lock is shared by every model in the process
# (rerank.py imports it). Model calls take milliseconds; the slow LLM calls still run in parallel.
MODEL_LOCK = threading.Lock()
_lock = MODEL_LOCK
_loaded = None


def _model():
    global _loaded
    if _loaded is None:
        # Imported lazily: loading torch takes a few seconds, and only some commands need it.
        import os

        os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # hide the "unauthenticated" notice
        from sentence_transformers import SentenceTransformer

        _loaded = SentenceTransformer(MODEL_NAME)
    return _loaded


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed passages. Vectors are normalized, so dot product == cosine similarity."""
    with _lock:
        return _model().encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 32
        )


def embed_query(question: str) -> np.ndarray:
    with _lock:
        return _model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
