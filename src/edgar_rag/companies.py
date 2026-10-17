"""The set of indexed companies, read from data/processed (one entry per company)."""

from functools import lru_cache

from pydantic import BaseModel

from edgar_rag import config
from edgar_rag.chunking import Chunk


class Company(BaseModel):
    ticker: str
    name: str
    form: str
    report_date: str  # last day of the latest indexed fiscal year

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])


@lru_cache(maxsize=1)
def indexed_companies() -> tuple[Company, ...]:
    companies = {}
    for path in sorted((config.data_dir() / "processed").glob("*/*.chunks.jsonl")):
        with path.open() as f:
            first = f.readline()
        if not first:
            continue
        c = Chunk.model_validate_json(first)
        latest = companies.get(c.ticker)
        if not latest or c.report_date > latest.report_date:
            companies[c.ticker] = Company(
                ticker=c.ticker, name=c.company, form=c.form, report_date=c.report_date
            )
    return tuple(sorted(companies.values(), key=lambda x: x.ticker))


def find(query: str) -> list[Company]:
    """Match a ticker exactly, or any word sequence of the company name, case-insensitively."""
    q = query.strip().lower()
    exact = [c for c in indexed_companies() if c.ticker.lower() == q]
    return exact or [c for c in indexed_companies() if q in c.name.lower()]
