"""v3 multi-year ingest: picking the filings and checking the years. Offline: cached SEC files."""

import json

import pytest

from edgar_rag.pipeline import YearHealth, _check_years
from edgar_rag.sec_client import SecClient

# Newest first, as the SEC lists them. A 10-K/A and other forms are mixed in.
FILINGS = [
    ("10-K", "0001-26-000001", "2026-01-25"),  # filed after v2 was indexed
    ("4", "0001-25-000099", ""),
    ("10-K", "0001-25-000001", "2025-01-26"),  # v2's latest
    ("10-K/A", "0001-24-000050", "2024-01-28"),
    ("10-K", "0001-24-000001", "2024-01-28"),
    ("10-Q", "0001-23-000070", "2023-10-29"),
    ("10-K", "0001-23-000001", "2023-01-29"),
    ("10-K", "0001-22-000001", "2022-01-30"),
]


@pytest.fixture
def client(tmp_path):
    (tmp_path / "company_tickers.json").write_text(json.dumps({"0": {"ticker": "NVDA", "cik_str": 1045810,
                                                                     "title": "NVIDIA CORP"}}))
    recent = {"form": [f for f, _, _ in FILINGS], "accessionNumber": [a for _, a, _ in FILINGS],
              "reportDate": [d for _, _, d in FILINGS], "filingDate": ["2026-01-01"] * len(FILINGS),
              "primaryDocument": ["doc.htm"] * len(FILINGS)}
    (tmp_path / "NVDA").mkdir()
    (tmp_path / "NVDA" / "submissions.json").write_text(json.dumps({"filings": {"recent": recent}}))
    return SecClient(cache_dir=tmp_path)


def test_filings_start_at_v2s_latest_and_skip_amendments(client):
    found = client.filings("NVDA", count=3, latest_accession="0001-25-000001")
    assert [f.accession_number for f in found] == ["0001-25-000001", "0001-24-000001", "0001-23-000001"]
    assert all(f.form == "10-K" for f in found)


def test_filings_without_anchor_start_at_the_newest(client):
    assert client.filings("NVDA", count=1)[0].accession_number == "0001-26-000001"


def test_too_few_filings_fails_loudly(client):
    with pytest.raises(ValueError, match="only 4"):
        client.filings("NVDA", count=5, latest_accession="0001-25-000001")


def test_unknown_anchor_fails(client):
    with pytest.raises(ValueError, match="not in the submissions list"):
        client.filings("NVDA", count=2, latest_accession="0009-99-999999")


def year(fy, a=100_000, b=50_000):
    return YearHealth(ticker="NVDA", fiscal_year=fy, item_chars={"1A": a, "7": b})


def test_consecutive_years_pass():
    report = [year(2026), year(2025), year(2024)]
    _check_years(report, 3)
    assert all(h.ok for h in report)


def test_gap_or_repeat_in_fiscal_years_is_flagged():
    report = [year(2026), year(2026), year(2024)]  # e.g. two years ending in the same calendar year
    _check_years(report, 3)
    assert "fiscal years" in report[0].warnings[0]


def test_older_section_much_smaller_than_latest_is_flagged():
    report = [year(2026), year(2025, a=20_000)]
    _check_years(report, 2)
    assert report[1].warnings and "Item 1A" in report[1].warnings[0]
