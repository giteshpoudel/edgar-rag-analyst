"""Scoring functions for evals: numbers in text, retrieval, and an LLM judge.

Each scorer is small and independent so it can be unit-tested and reasoned about on its own.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel

from edgar_rag import config
from edgar_rag.llm import complete

# --- Numbers -------------------------------------------------------------------------------

SCALES = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}
NUMBER_RE = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"  # 416,161 or 416.2 or 7.46
    r"\s*(%|percentage points?\b|percent\b|pp\b|basis points?\b|bps\b|"
    r"trillion\b|billion\b|million\b|bn\b|b\b|m\b)?",
    re.IGNORECASE,
)

# Allowed error per kind. Answers round ($416.2 billion, 6.4%), so exact match would be wrong.
USD_REL_TOL = 0.005  # 0.5%
PERCENT_ABS_TOL = 0.15  # percentage points
PER_SHARE_ABS_TOL = 0.011  # about one cent


class Expected(BaseModel):
    value: float
    # "count" is a plain quantity from text, e.g. 166,000 employees.
    kind: Literal["usd", "percent", "usd_per_share", "count"]
    # Optional override: relative for usd/count, percentage points for percent, absolute for
    # usd_per_share. For figures the filing itself rounds (text tables, "approximately").
    tolerance: float | None = None


def extract_numbers(text: str) -> tuple[list[float], list[float]]:
    """Return (plain numbers with any scale word applied, percentages) found in text."""
    plain, percents = [], []
    text = re.sub(r"\[\d+\]", " ", text)  # citation markers like [2] are not values
    for match in NUMBER_RE.finditer(text):
        value = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").lower()
        if unit in ("%", "percent", "pp") or unit.startswith("percentage point"):
            percents.append(value)
        elif unit in ("bps",) or unit.startswith("basis point"):
            percents.append(value / 100)  # 60 bps = 0.6 percentage points
        else:
            plain.append(value * SCALES.get(unit, 1))
    return plain, percents


def number_found(expected: Expected, text: str) -> bool:
    plain, percents = extract_numbers(text)
    tol = expected.tolerance
    if expected.kind == "percent":
        # Magnitude only: a decline of -37.5% is usually written "fell 37.5%". The judge's key
        # points check the direction ("fell"), so the number check doesn't double-count it.
        return any(abs(abs(p) - abs(expected.value)) <= (tol or PERCENT_ABS_TOL) for p in percents)
    if expected.kind == "usd_per_share":
        return any(abs(p - expected.value) <= (tol or PER_SHARE_ABS_TOL) for p in plain)
    # Dollar amounts may be written without a scale word inside tables ("416,161" in millions).
    # Compare magnitudes: a loss of -267M is usually written "a net loss of $267 million", and
    # dividing by a negative expected value would make every comparison pass.
    target = abs(expected.value)
    scales = (1, 1e3, 1e6, 1e9) if expected.kind == "count" else (1, 1e6, 1e9)
    return any(
        abs(abs(p) * scale - target) / target <= (tol or USD_REL_TOL)
        for p in plain
        for scale in scales
    )


# --- Evidence (gold passages for recall@k) -------------------------------------------------


class Evidence(BaseModel):
    """A short exact quote from a filing that answers (part of) a question."""

    ticker: str
    quote: str


def normalize_text(text: str) -> str:
    """Make quote matching robust to curly quotes, dashes and whitespace from HTML extraction."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def quotes_in_chunk(chunk_text: str, chunk_ticker: str, evidence: list[Evidence]) -> set[int]:
    """Indexes of the gold quotes (for this chunk's company) that the chunk contains."""
    body = normalize_text(chunk_text)
    return {
        i for i, e in enumerate(evidence)
        if e.ticker == chunk_ticker and normalize_text(e.quote) in body
    }


def recall_at_k(found_per_rank: list[set[int]], n_evidence: int, k: int) -> float:
    """Share of distinct gold quotes found in the top k results. Overlapping chunks can hold the
    same quote, so quotes are counted once, not chunks."""
    if n_evidence == 0:
        return 0.0
    found = set().union(*found_per_rank[:k]) if found_per_rank[:k] else set()
    return len(found) / n_evidence


def reciprocal_rank(found_per_rank: list[set[int]]) -> float:
    """1/rank of the first result containing any gold quote; 0 if none does."""
    return next((1 / r for r, found in enumerate(found_per_rank, 1) if found), 0.0)


# --- Retrieval -----------------------------------------------------------------------------


def retrieval_scores(retrieved_items: list[str], expected_items: list[str]) -> tuple[bool, float]:
    """(hit, reciprocal rank): was any expected section retrieved, and how high was the first one.

    Reciprocal rank is 1 if the first result is relevant, 1/2 if the second is, ..., 0 if none.
    """
    for rank, item in enumerate(retrieved_items, 1):
        if item in expected_items:
            return True, 1 / rank
    return False, 0.0


# --- LLM judge -----------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You grade answers from a financial Q&A system against a reference. Be strict and literal.

Return JSON only:
{"key_points": [true/false for each key point, in order], "refused": true/false}

- A key point is true only if the answer clearly states that fact (wording may differ; \
numbers must match after rounding).
- "refused" is true if the answer does NOT provide the specific information the question asks \
for, and instead says it isn't available, isn't disclosed, or can't be answered. It is still \
true when the answer adds related facts as context (e.g. "unit sales aren't disclosed, but \
iPhone revenue was $X"). It is false when the answer provides the asked-for information, even \
with caveats.
"""


class Judgement(BaseModel):
    key_points: list[bool]
    refused: bool
    error: str = ""  # set when the judge gave no usable verdict; never silently scored as wrong
    raw: str = ""


def judge(question: str, answer: str, key_points: list[str]) -> Judgement:
    points = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(key_points)) or "(none)"
    user = f"Question: {question}\n\nKey points:\n{points}\n\nAnswer to grade:\n{answer}"
    # The judge model reasons first; its reasoning tokens need room (300 was not enough).
    # A different, stronger model than the one being graded judges more consistently (D-41).
    response = complete(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=4000,
        model=config.judge_model(),
        span_name="judge",  # kept apart from the system's cost in reports
    )
    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or len(data.get("key_points", [])) != len(key_points):
        reason = "truncated" if choice.finish_reason == "length" else "unparseable verdict"
        return Judgement(key_points=[False] * len(key_points), refused=False, error=reason, raw=raw)
    covered = [bool(x) for x in data["key_points"]]
    return Judgement(key_points=covered, refused=bool(data.get("refused", False)), raw=raw)
