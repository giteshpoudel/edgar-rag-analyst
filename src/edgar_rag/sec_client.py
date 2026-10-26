"""Download filings from SEC EDGAR, with throttling and a local cache.

SEC fair-access rules: send a descriptive User-Agent and stay under 10 requests/second.
Everything downloaded is cached under DATA_DIR so re-runs don't hit the network.
"""

import json
import threading
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

from edgar_rag import config, retry

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

MIN_SECONDS_BETWEEN_REQUESTS = 0.15  # ~6-7 req/s, safely under the SEC limit

# One throttle for the whole process, not one per client: CompanyFacts creates its own client,
# and concurrent eval questions run in several threads, so a per-client limit could be exceeded.
_throttle_lock = threading.Lock()
_last_request = 0.0


def _wait_turn() -> None:
    global _last_request
    with _throttle_lock:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


class OfflineError(RuntimeError):
    """A request that would need the network while live SEC access is disabled."""


class Filing(BaseModel):
    """Metadata for one filing. Saved next to the downloaded document."""

    ticker: str
    company: str
    cik: int
    form: str
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )


class SecClient:
    def __init__(self, cache_dir: Path | None = None, transport: httpx.BaseTransport | None = None) -> None:
        self.cache_dir = cache_dir or config.data_dir() / "raw"
        self._transport = transport  # tests pass a fake
        self._client: httpx.Client | None = None

    @property
    def _http(self) -> httpx.Client:
        """Created on the first real request, so reading cached files needs no SEC_USER_AGENT
        (CI runs on a frozen data snapshot and never contacts the SEC)."""
        if self._client is None:
            self._client = httpx.Client(
                transport=self._transport,
                headers={"User-Agent": config.sec_user_agent(), "Accept-Encoding": "gzip, deflate"},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    def _get(self, url: str) -> httpx.Response:
        """GET with the shared throttle, retrying timeouts, 429s and 5xx with backoff (the SEC
        answers 429 or 503 when it thinks a client is going too fast)."""

        if config.demo_mode():
            # The public demo serves only the frozen snapshot: a cache miss must never turn a
            # visitor's question into an SEC download (the agent did fetch TSMC and ASML facts
            # during an eval when asked about companies that aren't indexed).
            raise OfflineError(f"no cached copy of {url} and live SEC requests are off in demo mode")

        def attempt() -> httpx.Response:
            _wait_turn()
            response = self._http.get(url)
            response.raise_for_status()
            return response

        return retry.call(attempt)

    def _get_json_cached(self, url: str, path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text())
        data = self._get(url).json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    def lookup_cik(self, ticker: str) -> tuple[int, str]:
        """Return (CIK, company name) for a stock ticker like 'AAPL'."""
        tickers = self._get_json_cached(TICKERS_URL, self.cache_dir / "company_tickers.json")
        for row in tickers.values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"]), row["title"]
        raise ValueError(f"Unknown ticker: {ticker}")

    def latest_filing(self, ticker: str, form: str = "10-K") -> Filing:
        """Find the most recent filing of the given form type for a ticker."""
        cik, company = self.lookup_cik(ticker)
        submissions = self._get_json_cached(
            SUBMISSIONS_URL.format(cik=cik),
            self.cache_dir / ticker.upper() / "submissions.json",
        )
        recent = submissions["filings"]["recent"]
        # `recent` is column-oriented and sorted newest first.
        for i, form_type in enumerate(recent["form"]):
            if form_type == form:
                return Filing(
                    ticker=ticker.upper(),
                    company=company,
                    cik=cik,
                    form=form_type,
                    accession_number=recent["accessionNumber"][i],
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    primary_document=recent["primaryDocument"][i],
                )
        raise ValueError(f"No {form} found for {ticker}")

    def company_facts(self, ticker: str) -> dict:
        """All XBRL facts the company has reported (every filing), as one JSON document."""
        cik, _ = self.lookup_cik(ticker)
        return self._get_json_cached(
            COMPANY_FACTS_URL.format(cik=cik),
            self.cache_dir / ticker.upper() / "companyfacts.json",
        )

    def download(self, filing: Filing) -> Path:
        """Download a filing's primary document (HTML) unless it's already cached."""
        folder = self.cache_dir / filing.ticker
        stem = f"{filing.form}_{filing.report_date}_{filing.accession_number}"
        html_path = folder / f"{stem}.html"
        meta_path = folder / f"{stem}.json"
        if not html_path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            html_path.write_bytes(self._get(filing.url).content)
            meta_path.write_text(filing.model_dump_json(indent=2))
        return html_path
