"""Cross-encoder reranking of first-stage search results.

A bi-encoder (the embedding model) turns the question and each passage into vectors separately,
so it is fast but compares them only through those vectors. A cross-encoder reads the question
and one passage together and outputs a relevance score, which is more accurate but needs a model
run per pair. So it only re-orders the top few dozen candidates from the fast first stage.
"""

import os
import threading

MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 22M parameters: fast on CPU
    "bge": "BAAI/bge-reranker-base",  # 278M parameters: stronger, slower
}

_lock = threading.Lock()  # torch models aren't safe to load or run from several threads at once
_loaded: dict = {}


def _model(name: str):
    if name not in MODELS:
        raise ValueError(f"Unknown reranker '{name}'. Choose from: {', '.join(MODELS)}")
    if name not in _loaded:
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        from sentence_transformers import CrossEncoder

        _loaded[name] = CrossEncoder(MODELS[name], max_length=512)
    return _loaded[name]


def rerank_scores(name: str, question: str, passages: list[str]) -> list[float]:
    """Relevance score for each passage (higher = more relevant); same order as the input."""
    if not passages:
        return []
    with _lock:
        scores = _model(name).predict([(question, p) for p in passages], batch_size=16, show_progress_bar=False)
    return [float(s) for s in scores]
