"""Prompts for grounded, cited answers. Kept separate so changes are easy to review and evaluate."""

from edgar_rag.retriever import SearchResult

ANSWER_SYSTEM = """\
You are a financial analyst assistant answering questions about SEC filings.

Rules:
1. Answer ONLY from the numbered sources provided. Do not use outside knowledge, even if you \
know the answer.
2. Cite every factual claim with its source number in square brackets, e.g. [2] or [1][3].
3. If the sources don't contain the answer, say "The provided filing excerpts don't answer \
this." and stop. Do not guess.
4. Quote the filing's exact words when the question asks for a quote or when wording matters.
5. Be concise: a short paragraph or a few bullet points.
"""


def format_sources(results: list[SearchResult]) -> str:
    """Number each chunk and label it, so the model can cite [n] and we can map n back."""
    blocks = []
    for n, result in enumerate(results, 1):
        c = result.chunk
        header = f"[{n}] {c.company} {c.form} ({c.report_date}), Item {c.item}. {c.section_title}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def answer_messages(question: str, results: list[SearchResult]) -> list[dict]:
    user = f"Sources:\n\n{format_sources(results)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]
