"""RAG answer pipeline: retrieve -> build a cited prompt -> generate -> check citations."""

import re

from pydantic import BaseModel

from edgar_rag.llm import LlmResponse, chat
from edgar_rag.prompts import answer_messages
from edgar_rag.retriever import SearchResult, search

CITATION_RE = re.compile(r"\[(\d+)\]")


class Answer(BaseModel):
    question: str
    text: str
    sources: list[SearchResult]
    cited: list[int]  # source numbers the answer cites, in order of first use
    invalid_citations: list[int]  # numbers that don't match any source
    llm: LlmResponse


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    seen: list[int] = []
    for match in CITATION_RE.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    valid = [n for n in seen if 1 <= n <= n_sources]
    invalid = [n for n in seen if not 1 <= n <= n_sources]
    return valid, invalid


def answer(
    question: str, k: int = 5, ticker: str | None = None, item: str | None = None
) -> Answer:
    results = search(question, k=k, ticker=ticker, item=item)
    response = chat(answer_messages(question, results))
    cited, invalid = parse_citations(response.text, len(results))
    return Answer(
        question=question,
        text=response.text,
        sources=results,
        cited=cited,
        invalid_citations=invalid,
        llm=response,
    )
