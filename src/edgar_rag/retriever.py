"""Search over indexed chunks: vector, keyword or hybrid, with optional reranking and filters.

    vector   embeddings + cosine similarity: finds passages by meaning
    keyword  BM25 full-text search: finds exact terms ("deferred profit", "EPYC")
    hybrid   both, merged with reciprocal rank fusion (RRF)
    + rerank a cross-encoder re-orders the top candidates of any mode

The mode and reranker come from config (RETRIEVAL_MODE, RERANKER) unless passed explicitly, so
the agent's tools don't change when the retrieval setup does.
"""

from pydantic import BaseModel

from edgar_rag import config
from edgar_rag.chunking import Chunk
from edgar_rag.embeddings import embed_query
from edgar_rag.index import TABLE_NAME, connect

RRF_K = 60  # standard constant: damps the advantage of rank 1 over rank 2 and so on


class SearchResult(BaseModel):
    chunk: Chunk
    # Relevance, higher is better. Its scale depends on the mode: cosine similarity (vector),
    # BM25 (keyword), RRF (hybrid) or the reranker's score. Compare ranks, not values, across modes.
    score: float


def _where(ticker: str | None, item: str | None) -> str | None:
    filters = []
    if ticker:
        filters.append(f"ticker = '{ticker.upper()}'")
    if item:
        filters.append(f"item = '{item.upper()}'")
    return " AND ".join(filters) or None


def _rows_to_results(rows: list[dict], score_field: str, transform=lambda s: s) -> list[SearchResult]:
    results = []
    for row in rows:
        score = transform(row.pop(score_field))
        for extra in ("vector", "_distance", "_score", "_relevance_score"):
            row.pop(extra, None)
        results.append(SearchResult(chunk=Chunk(**row), score=score))
    return results


def _vector(question: str, n: int, where: str | None) -> list[SearchResult]:
    query = connect().open_table(TABLE_NAME).search(embed_query(question)).metric("cosine")
    if where:
        # prefilter: apply the filter before ranking, so we still get n results.
        query = query.where(where, prefilter=True)
    return _rows_to_results(query.limit(n).to_list(), "_distance", lambda d: 1 - d)


def _keyword(question: str, n: int, where: str | None) -> list[SearchResult]:
    query = connect().open_table(TABLE_NAME).search(question, query_type="fts")
    if where:
        query = query.where(where, prefilter=True)
    return _rows_to_results(query.limit(n).to_list(), "_score")


def _rrf(*ranked_lists: list[SearchResult]) -> list[SearchResult]:
    """Reciprocal rank fusion: each list contributes 1 / (RRF_K + rank) per passage. Only ranks
    are used, so cosine and BM25 scores (very different scales) never need comparing."""
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in ranked_lists:
        for rank, r in enumerate(results, 1):
            fused[r.chunk.chunk_id] = fused.get(r.chunk.chunk_id, 0.0) + 1 / (RRF_K + rank)
            chunks[r.chunk.chunk_id] = r.chunk
    order = sorted(fused, key=fused.get, reverse=True)
    return [SearchResult(chunk=chunks[i], score=fused[i]) for i in order]


def search(
    question: str,
    k: int = 5,
    ticker: str | None = None,
    item: str | None = None,
    mode: str | None = None,
    reranker: str | None = None,
    candidates: int | None = None,
) -> list[SearchResult]:
    """Return the k most relevant chunks for the question."""
    mode = mode or config.retrieval_mode()
    reranker = reranker or config.reranker()
    use_reranker = reranker != "none"
    # The first stage fetches more than k when a reranker will choose among them.
    n = max(k, candidates or config.rerank_candidates()) if use_reranker else k
    where = _where(ticker, item)

    if mode == "vector":
        results = _vector(question, n, where)
    elif mode == "keyword":
        results = _keyword(question, n, where)
    elif mode == "hybrid":
        results = _rrf(_vector(question, n, where), _keyword(question, n, where))[:n]
    else:
        raise ValueError(f"Unknown retrieval mode '{mode}': use vector, keyword or hybrid")

    if use_reranker and results:
        from edgar_rag.rerank import rerank_scores

        scores = rerank_scores(reranker, question, [r.chunk.text for r in results])
        results = [SearchResult(chunk=r.chunk, score=s) for r, s in zip(results, scores)]
        results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
