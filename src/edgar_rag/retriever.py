"""Semantic search over indexed chunks, with optional metadata filters."""

from pydantic import BaseModel

from edgar_rag.chunking import Chunk
from edgar_rag.embeddings import embed_query
from edgar_rag.index import TABLE_NAME, connect


class SearchResult(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity: 1.0 = same meaning, ~0 = unrelated


def search(
    question: str,
    k: int = 5,
    ticker: str | None = None,
    item: str | None = None,
) -> list[SearchResult]:
    """Return the k chunks most similar in meaning to the question."""
    query = connect().open_table(TABLE_NAME).search(embed_query(question)).metric("cosine")

    filters = []
    if ticker:
        filters.append(f"ticker = '{ticker.upper()}'")
    if item:
        filters.append(f"item = '{item.upper()}'")
    if filters:
        # prefilter: apply the filter before ranking, so we still get k results.
        query = query.where(" AND ".join(filters), prefilter=True)

    results = []
    for row in query.limit(k).to_list():
        distance = row.pop("_distance")
        row.pop("vector")
        results.append(SearchResult(chunk=Chunk(**row), score=1 - distance))
    return results
