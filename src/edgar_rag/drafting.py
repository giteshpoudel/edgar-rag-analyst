"""Helpers for drafting and checking v3 eval questions from raw filing text (v3 Phase 5a).

Questions must come from the filings themselves, not from the claims database, or the eval would
favour the system that built that database. These helpers only read the 10-K HTML:

    section_text(ticker, fy)         Items 1A and 7 as plain text (the same blocks the labels use)
    sentence_diff(ticker, fy)        sentences added and removed versus the year before
    mentions(ticker, phrase)         which fiscal years' Items 1A/7 contain a phrase
    quote_years(ticker, quote)       which fiscal years contain an exact gold quote

`mentions` and `quote_years` are what verify "first appeared" and "dropped" answers: a phrase that
first appears in FY2023 must be absent from every earlier year in the data.
"""

import re
from functools import cache

from edgar_rag import companies, config
from edgar_rag.labeling import blocks_for_filing, filing_html, normalize

SECTIONS = ("1A", "7")
SENTENCE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(“\"])")


@cache
def section_text(ticker: str, fiscal_year: int) -> dict[str, str]:
    _, html = filing_html(ticker.upper(), fiscal_year)
    blocks = blocks_for_filing(html)
    return {s: normalize(" ".join(b.text for b in blocks if b.item == s)) for s in SECTIONS}


def years(ticker: str) -> list[int]:
    """Fiscal years with a 10-K in the multi-year data."""
    from edgar_rag.pipeline import MULTI_DIR

    folder = config.data_dir() / MULTI_DIR / ticker.upper()
    found = sorted({int(m.group(1)) for p in folder.glob("*.health.json")
                    if (m := re.search(r"FY(\d{4})", p.name))}) if folder.exists() else []
    if found:
        return found
    latest = next((c.fiscal_year for c in companies.indexed_companies() if c.ticker == ticker.upper()), None)
    return list(range(latest - 4, latest + 1)) if latest else []


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text) if len(s.strip()) >= 40]


def sentence_diff(ticker: str, fiscal_year: int, section: str) -> tuple[list[str], list[str]]:
    """(added, removed): sentences in this year's section but not last year's, and the reverse.
    Whitespace and quote marks are normalised; anything else that differs counts as a change."""
    now = sentences(section_text(ticker, fiscal_year)[section])
    before = sentences(section_text(ticker, fiscal_year - 1)[section])
    key = lambda s: re.sub(r"\s+", " ", s.lower())
    now_keys, before_keys = {key(s) for s in now}, {key(s) for s in before}
    return [s for s in now if key(s) not in before_keys], [s for s in before if key(s) not in now_keys]


def mentions(ticker: str, phrase: str, section: str | None = None) -> dict[int, int]:
    """fiscal year -> number of times the phrase appears (case-insensitive, whole words)."""
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(normalize(phrase).lower())}(?![a-z0-9])")
    out = {}
    for fy in years(ticker):
        try:
            texts = section_text(ticker, fy)
        except FileNotFoundError:
            continue
        out[fy] = sum(len(pattern.findall(t.lower())) for s, t in texts.items() if section in (None, s))
    return out


def quote_years(ticker: str, quote: str) -> list[int]:
    """Fiscal years whose Items 1A/7 contain the exact (normalised) quote."""
    q = normalize(quote)
    return [fy for fy in years(ticker) if any(q in t for t in section_text(ticker, fy).values())]
