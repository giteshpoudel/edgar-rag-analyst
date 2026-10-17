"""Batch ingestion: download -> chunk -> index, with a health report per filing.

At more than a handful of companies nobody reads every filing, so each ingest checks itself
for the failure signs seen so far: missing core sections (the heading parser didn't match),
one section swallowing the document, and XBRL metrics the tag mapping doesn't cover.
"""

from collections import Counter

from pydantic import BaseModel

from edgar_rag import config
from edgar_rag.chunking import chunk_filing
from edgar_rag.sec_client import SecClient
from edgar_rag.xbrl import PUBLIC_METRICS, CompanyFacts, FactNotFound

CORE_ITEMS = ["1", "1A", "7", "8"]  # Business, Risk Factors, MD&A, Financial Statements
MIN_SECTIONS = 15  # a 10-K has ~20-23 Items; far fewer means headings were missed
MAX_SECTION_SHARE = 0.6  # one section holding most chunks suggests a missed boundary


class Health(BaseModel):
    ticker: str
    company: str = ""
    fiscal_year: int | None = None
    chunks: int = 0
    sections: int = 0
    missing_core: list[str] = []
    largest_section: str = ""
    largest_share: float = 0.0
    metrics_found: int = 0
    metrics_missing: list[str] = []
    warnings: list[str] = []
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.warnings


def ingest_one(ticker: str, client: SecClient | None = None) -> Health:
    """Download, chunk and index one company's latest 10-K, then check the result."""
    from edgar_rag.index import index_chunks  # loads the embedding model; import lazily

    client = client or SecClient()
    health = Health(ticker=ticker.upper())
    try:
        filing = client.latest_filing(ticker, "10-K")
        html_path = client.download(filing)
        chunks = chunk_filing(html_path.read_text(encoding="utf-8"), filing)

        out_dir = config.data_dir() / "processed" / filing.ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{html_path.stem}.chunks.jsonl").write_text(
            "".join(c.model_dump_json() + "\n" for c in chunks)
        )
        if chunks:
            index_chunks(chunks)
    except Exception as e:  # noqa: BLE001 - report the failure and move on to the next company
        health.error = f"{type(e).__name__}: {e}"
        return health

    health.company = filing.company
    health.fiscal_year = int(filing.report_date[:4])
    health.chunks = len(chunks)
    per_item = Counter(c.item for c in chunks)
    health.sections = len(per_item)
    health.missing_core = [i for i in CORE_ITEMS if i not in per_item]
    if per_item:
        item, count = per_item.most_common(1)[0]
        health.largest_section, health.largest_share = item, count / len(chunks)

    try:
        facts = CompanyFacts(ticker, client)
        for metric in PUBLIC_METRICS:
            try:
                facts.get(metric, health.fiscal_year)
                health.metrics_found += 1
            except FactNotFound:
                health.metrics_missing.append(metric)
    except Exception as e:  # noqa: BLE001
        health.warnings.append(f"XBRL facts unavailable: {e}")

    if health.missing_core:
        health.warnings.append(f"missing core sections {health.missing_core}")
    if health.sections < MIN_SECTIONS:
        health.warnings.append(f"only {health.sections} sections found")
    if health.largest_share > MAX_SECTION_SHARE:
        health.warnings.append(
            f"Item {health.largest_section} holds {health.largest_share:.0%} of chunks"
        )
    if health.metrics_missing:
        health.warnings.append(f"XBRL metrics missing: {', '.join(health.metrics_missing)}")
    return health


def ingest(tickers: list[str]) -> list[Health]:
    client = SecClient()  # one client so request throttling covers the whole batch
    return [ingest_one(t, client) for t in tickers]
