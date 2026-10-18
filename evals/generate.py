"""Build the v2 eval set (evals/golden_v2.jsonl).

    python evals/generate.py migrate     # v1's 58 questions -> regression split, with evidence quotes
    python evals/generate.py draft       # LLM-drafted candidates from sampled chunks -> candidates_v2.jsonl
    python evals/generate.py templates   # code-built near-miss, ambiguous and hard numeric questions
    python evals/generate.py verify      # mechanical checks on every candidate
    python evals/generate.py assemble    # verified questions + dev/test split -> golden_v2.jsonl

Drafting uses JUDGE_MODEL (deepseek-v4-pro), a different model from the one being evaluated.
Nothing an LLM writes is trusted as is: quotes must be exact substrings of the source chunk,
expected numbers must appear in a quote or come from XBRL, and every question is reviewed.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from edgar_rag import config
from edgar_rag.chunking import Chunk
from edgar_rag.llm import complete
from edgar_rag.scoring import Evidence, normalize_text

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Build the v2 eval set."""
EVALS = Path(__file__).resolve().parent
V1 = EVALS / "golden.jsonl"
MIGRATED = EVALS / "v2_regression.jsonl"

# Company names as they appear in questions -> tickers.
NAMES = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "texas instruments": "TXN",
    "lam research": "LRCX", "applied materials": "AMAT", "qualcomm": "QCOM", "intel": "INTC",
    "broadcom": "AVGO", "micron": "MU", "analog devices": "ADI", "kla": "KLAC",
    "marvell": "MRVL", "microchip": "MCHP", "amd": "AMD", "advanced micro devices": "AMD",
}
SEMIS = ["ADI", "AMAT", "AMD", "AVGO", "INTC", "KLAC", "LRCX", "MCHP", "MRVL", "MU", "NVDA", "QCOM", "TXN"]


# --- helpers -------------------------------------------------------------------------------


def companies_in(question: str) -> list[str]:
    q = question.lower()
    found = [t for name, t in NAMES.items() if re.search(rf"\b{re.escape(name)}\b", q)]
    if "semiconductor" in q and not found:
        found = SEMIS
    return sorted(set(found))


def chunks_for(ticker: str) -> list[Chunk]:
    path = next((config.data_dir() / "processed" / ticker).glob("*.chunks.jsonl"))
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines()]


def ask_json(system: str, user: str) -> dict | None:
    response = complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=6000,
        model=config.judge_model(),
    )
    raw = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def exact_quote(quote: str, chunk: Chunk) -> bool:
    return bool(quote.strip()) and normalize_text(quote) in normalize_text(chunk.text)


MIN_QUOTE_CHARS = 30  # shorter spans ("As of September 27, 2025") are rarely specific
MAX_QUOTE_CHUNKS = 4  # real disclosures repeat across 2-3 sections; generic filler appears in dozens


def distinctive(quote: str, company_chunks: list[Chunk]) -> bool:
    """A gold quote must be specific: long enough, and in at most a couple of chunks. Otherwise
    any retrieved chunk would 'contain the evidence' and recall@k would mean nothing."""
    q = normalize_text(quote)
    hits = sum(q in normalize_text(c.text) for c in company_chunks)
    return len(q) >= MIN_QUOTE_CHARS and 1 <= hits <= MAX_QUOTE_CHUNKS


WORD_RE = re.compile(r"[a-z0-9][a-z0-9.,%$-]*[a-z0-9%]|[a-z0-9]")
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "by", "with", "from", "at", "as",
    "is", "are", "was", "were", "be", "its", "it", "this", "that", "which", "what", "how", "does",
    "did", "do", "has", "have", "had", "their", "company", "companies", "fiscal", "year", "years",
}


def keyword_candidates(text: str, company_chunks: list[Chunk], items: list[str], n: int = 8) -> list[Chunk]:
    """Pick candidate chunks by keyword overlap, weighting rare words higher (a simple IDF).

    Deliberately independent of the vector retriever: gold evidence found with the system
    under test would be biased toward what that system already finds."""
    import math

    terms = {w for w in WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 2}
    docs = [set(WORD_RE.findall(normalize_text(c.text))) for c in company_chunks]
    df = {t: sum(t in d for d in docs) for t in terms}
    idf = {t: math.log((1 + len(docs)) / (1 + df[t])) for t in terms if df[t]}

    def score(i: int) -> float:
        bonus = 1.2 if items and company_chunks[i].item in items else 1.0
        return bonus * sum(w for t, w in idf.items() if t in docs[i])

    ranked = sorted(range(len(company_chunks)), key=score, reverse=True)
    return [company_chunks[i] for i in ranked[:n]]


# --- migrate -------------------------------------------------------------------------------

# Evidence found by hand where the drafting model didn't return a usable quote. Still checked
# for exactness and distinctiveness like any other quote.
MANUAL_EVIDENCE: dict[str, list[tuple[str, str]]] = {
    "nar-18": [(
        "LRCX",
        "represented approximately 34%, 34%, and 42% of our total revenue for fiscal years 2026, 2025, and 2024",
    )],
}

QUOTE_SYSTEM = """\
You find supporting evidence in SEC filing excerpts. For each key point, copy the shortest \
exact span (one clause or sentence, under 200 characters) from ONE passage that supports it. \
Copy it character for character; do not paraphrase, fix typos or join text from two places.

Return JSON only: {"quotes": [{"point": <key point number>, "passage": <passage number>, \
"quote": "<exact text>"}]}. Skip a key point if no passage supports it.
"""


def find_evidence(
    question: str, key_points: list[str], ticker: str, items: list[str], query: str | None = None
) -> list[Evidence]:
    company_chunks = chunks_for(ticker)
    candidates = keyword_candidates(query or " ".join([question, *key_points]), company_chunks, items)
    passages = "\n\n".join(f"[{i + 1}] Item {c.item}\n{c.text}" for i, c in enumerate(candidates))
    points = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(key_points))
    data = ask_json(QUOTE_SYSTEM, f"Question: {question}\n\nKey points:\n{points}\n\nPassages:\n{passages}")
    evidence = []
    for q in (data or {}).get("quotes", []):
        try:
            chunk = candidates[int(q["passage"]) - 1]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        quote = q.get("quote", "")
        if (
            exact_quote(quote, chunk)
            and distinctive(quote, company_chunks)
            and all(quote.strip() != e.quote for e in evidence)
        ):
            evidence.append(Evidence(ticker=ticker, quote=quote.strip()))
    return drop_contained(evidence)


def drop_contained(evidence: list[Evidence]) -> list[Evidence]:
    """Drop a quote that is part of a longer quote: both would mark the same chunk as relevant."""
    norm = [normalize_text(e.quote) for e in evidence]
    return [e for i, e in enumerate(evidence) if not any(i != j and norm[i] in norm[j] for j in range(len(norm)))]


@app.command()
def migrate(workers: int = 6) -> None:
    """Copy v1's questions into the regression split, adding companies and evidence quotes."""
    items = [json.loads(line) for line in V1.read_text().splitlines() if line]
    # Keep evidence that already passed the checks; only look again where there is none.
    previous = {}
    if MIGRATED.exists():
        previous = {q["id"]: q for q in map(json.loads, MIGRATED.read_text().splitlines())}

    def one(q: dict) -> dict:
        q = {**q, "split": "regression", "source": "v1", "companies": companies_in(q["question"])}
        needs_text = q["category"] in ("narrative", "mixed") and q.get("key_points")
        kept = previous.get(q["id"], {}).get("evidence")
        if kept and q["id"] not in MANUAL_EVIDENCE:
            q["evidence"] = [e.model_dump() for e in drop_contained([Evidence(**e) for e in kept])]
        elif q["id"] in MANUAL_EVIDENCE:
            ev = []
            for ticker, quote in MANUAL_EVIDENCE[q["id"]]:
                company_chunks = chunks_for(ticker)
                if any(exact_quote(quote, c) for c in company_chunks) and distinctive(quote, company_chunks):
                    ev.append(Evidence(ticker=ticker, quote=quote))
                else:
                    raise ValueError(f"manual evidence for {q['id']} is not an exact, distinctive quote")
            q["evidence"] = [e.model_dump() for e in ev]
        elif needs_text and len(q["companies"]) == 1:
            ev = []
            # Retry once with only the key points as the query (the drafting model occasionally
            # returns nothing usable, and mixed questions' numbers can crowd out the narrative).
            for query in (None, " ".join(q["key_points"])):
                ev = find_evidence(
                    q["question"], q["key_points"], q["companies"][0], q.get("expected_items", []), query
                )
                if ev:
                    break
            q["evidence"] = [e.model_dump() for e in ev]
        # v1 questions were checked by hand when written (Phases 3-5).
        q["verified"] = True
        q["notes"] = "v1 question, verified against filing text and XBRL when written"
        return q

    with ThreadPoolExecutor(max_workers=workers) as pool:
        migrated = list(pool.map(one, items))
    MIGRATED.write_text("".join(json.dumps(q) + "\n" for q in migrated))

    text_qs = [q for q in migrated if q["category"] in ("narrative", "mixed")]
    with_ev = [q for q in text_qs if q.get("evidence")]
    typer.echo(f"Migrated {len(migrated)} questions -> {MIGRATED.name}")
    typer.echo(f"Text questions with evidence quotes: {len(with_ev)}/{len(text_qs)}")
    for q in text_qs:
        if not q.get("evidence"):
            typer.echo(f"  no evidence found: {q['id']} {q['question'][:70]}")


if __name__ == "__main__":
    app()
