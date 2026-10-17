"""Local embedding model (runs on CPU, no API needed)."""

from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dimensional vectors, max 512 tokens

# bge models are trained to embed search queries with this prefix (documents get none).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    # Imported lazily: loading torch takes a few seconds, and only some commands need it.
    import os

    os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # hide the "unauthenticated" notice
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed passages. Vectors are normalized, so dot product == cosine similarity."""
    return _model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 32
    )


def embed_query(question: str) -> np.ndarray:
    return _model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
