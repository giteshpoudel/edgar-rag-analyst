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


# --- v3: several fiscal years per company --------------------------------------------------

V3_ITEMS = ["1A", "7"]  # Risk Factors and MD&A: what v3 extracts claims from
MULTI_DIR = "processed_multi"  # kept apart from data/processed, which v2's eval and tools read


class YearHealth(BaseModel):
    """Health of one filing in a multi-year ingest."""

    ticker: str
    fiscal_year: int | None = None
    report_date: str = ""
    accession: str = ""
    chunks: int = 0
    sections: int = 0
    item_chars: dict[str, int] = {}  # characters of text in Items 1A and 7
    reused: bool = False  # already in chunks_multi, so not embedded again
    matches_v2: bool | None = None  # latest year only: same chunks as v2's file
    warnings: list[str] = []
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.warnings


def v2_latest(ticker: str) -> tuple[str, list]:
    """The accession number and chunks of the 10-K v2 indexed for a company."""
    from edgar_rag.index import load_chunks

    (path,) = sorted((config.data_dir() / "processed" / ticker.upper()).glob("*.chunks.jsonl"))
    chunks = load_chunks(path)
    return chunks[0].accession_number, chunks


def ingest_years_one(ticker: str, years: int, client: SecClient) -> list[YearHealth]:
    """Download, chunk and index `years` 10-Ks for one company, ending at v2's latest one."""
    from edgar_rag.index import MULTI_TABLE_NAME, index_chunks, indexed_count

    ticker = ticker.upper()
    try:
        latest_accession, v2_chunks = v2_latest(ticker)
        filings = client.filings(ticker, "10-K", years, latest_accession=latest_accession)
    except Exception as e:  # noqa: BLE001
        return [YearHealth(ticker=ticker, error=f"{type(e).__name__}: {e}")]

    report: list[YearHealth] = []
    for filing in filings:
        h = YearHealth(ticker=ticker, fiscal_year=int(filing.report_date[:4]),
                       report_date=filing.report_date, accession=filing.accession_number)
        try:
            html_path = client.download(filing)
            chunks = chunk_filing(html_path.read_text(encoding="utf-8"), filing)
            out_dir = config.data_dir() / MULTI_DIR / ticker
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{html_path.stem}.chunks.jsonl").write_text(
                "".join(c.model_dump_json() + "\n" for c in chunks))
            if chunks and indexed_count(filing.accession_number, MULTI_TABLE_NAME) == len(chunks):
                h.reused = True  # chunking is deterministic, so the stored rows are the same
            elif chunks:
                index_chunks(chunks, MULTI_TABLE_NAME, extra={"fiscal_year": h.fiscal_year},
                             keyword_index=False)
        except Exception as e:  # noqa: BLE001
            h.error = f"{type(e).__name__}: {e}"
            report.append(h)
            continue

        h.chunks = len(chunks)
        h.sections = len({c.item for c in chunks})
        for item in V3_ITEMS:
            h.item_chars[item] = sum(len(c.text) for c in chunks if c.item == item)
            if not h.item_chars[item]:
                h.warnings.append(f"Item {item} not found")
        if h.sections < MIN_SECTIONS:
            h.warnings.append(f"only {h.sections} sections found")
        if filing.accession_number == latest_accession:
            h.matches_v2 = [(c.chunk_id, c.text) for c in chunks] == [(c.chunk_id, c.text) for c in v2_chunks]
            if not h.matches_v2:
                h.warnings.append("latest year's chunks differ from v2's")
        report.append(h)
    _check_years(report, years)
    return report


def _check_years(report: list[YearHealth], years: int) -> None:
    """Fiscal years must be unique and consecutive: a gap or a repeat means a filing was
    skipped, or a 52/53-week year ended in the first days of January and got the wrong name."""
    found = [h.fiscal_year for h in report if h.fiscal_year]
    if not found:
        return
    expected = list(range(max(found), max(found) - years, -1))
    if found != expected:
        report[0].warnings.append(f"fiscal years {found}, expected {expected}")
    # Older filings shrinking sharply usually means a section boundary was missed.
    latest = report[0].item_chars
    for h in report[1:]:
        for item in V3_ITEMS:
            if latest.get(item) and h.item_chars.get(item, 0) < 0.4 * latest[item]:
                h.warnings.append(f"Item {item} is {h.item_chars.get(item, 0):,} chars vs "
                                  f"{latest[item]:,} in the latest year")


def ingest_years(tickers: list[str], years: int = 5) -> list[YearHealth]:
    """v3 multi-year ingest into data/processed_multi and the chunks_multi table."""
    from edgar_rag.index import MULTI_TABLE_NAME, build_keyword_index

    client = SecClient()
    report = [h for t in tickers for h in ingest_years_one(t, years, client)]
    if any(h.chunks for h in report):
        build_keyword_index(MULTI_TABLE_NAME)  # once for the batch, not once per filing
    return report
