"""XBRL fact selection. Offline: small synthetic company-facts documents."""

import pytest

from edgar_rag.xbrl import CompanyFacts, FactNotFound


def fact(start, end, val, filed, form="10-K", accn="0000000000-00-000001"):
    f = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn}
    if start:
        f["start"] = start
    return f


class FakeClient:
    def __init__(self, gaap: dict):
        self.gaap = gaap

    def company_facts(self, ticker):
        return {"entityName": f"{ticker} Inc.", "cik": 1, "facts": {"us-gaap": self.gaap}}


def facts(gaap: dict) -> CompanyFacts:
    return CompanyFacts("TEST", client=FakeClient(gaap))


def usd(*items):
    return {"label": "x", "units": {"USD": list(items)}}


# A January year-end, like NVIDIA: the year ending 2026-01-25 runs almost entirely through 2025.
JANUARY_YEAR_END = {
    "Revenues": usd(
        fact("2024-01-29", "2025-01-26", 130.5, "2025-02-26"),
        fact("2025-01-27", "2026-01-25", 215.9, "2026-02-25"),
    ),
    "NetIncomeLoss": usd(fact("2025-01-27", "2026-01-25", 120.1, "2026-02-25")),
}


def test_fiscal_year_is_named_by_the_year_it_ends_in():
    # FY2026 ends 2026-01-25, matching the company's own label and the golden questions.
    # Naming years by their start date (the demo PR #2 change) breaks this.
    f = facts(JANUARY_YEAR_END)
    assert f.get("revenue", 2026).value == 215.9
    assert f.get("revenue", 2025).value == 130.5
    assert f.get("revenue", 2026).period_end == "2026-01-25"


def test_latest_filing_wins_so_restatements_are_used():
    f = facts({"Revenues": usd(
        fact("2024-09-29", "2025-09-27", 400.0, "2025-10-31"),
        fact("2024-09-29", "2025-09-27", 416.2, "2026-10-30"),  # restated in the next 10-K
    )})
    assert f.get("revenue", 2025).value == 416.2


def test_quarterly_and_10q_values_are_ignored():
    f = facts({"Revenues": usd(
        fact("2025-06-29", "2025-09-27", 102.5, "2025-10-31"),  # one quarter, filed in a 10-K
        fact("2024-09-29", "2025-09-27", 416.2, "2025-10-31"),
        fact("2024-09-29", "2025-09-27", 999.0, "2025-08-01", form="10-Q"),
    )})
    assert f.get("revenue", 2025).value == 416.2


def test_gross_profit_is_derived_and_labelled_when_not_reported():
    f = facts({
        "Revenues": usd(fact("2024-09-29", "2025-09-27", 416.2, "2025-10-31")),
        "CostOfRevenue": usd(fact("2024-09-29", "2025-09-27", 221.0, "2025-10-31")),
    })
    gp = f.get("gross_profit", 2025)
    assert gp.value == pytest.approx(195.2) and gp.tag.startswith("derived")


def test_balance_sheet_values_are_taken_at_year_end():
    f = facts({
        "Revenues": usd(fact("2024-09-29", "2025-09-27", 416.2, "2025-10-31")),
        "Assets": usd(fact(None, "2025-09-27", 359.2, "2025-10-31"), fact(None, "2024-09-28", 365.0, "2025-10-31")),
    })
    assert f.get("total_assets", 2025).value == 359.2


def test_missing_year_lists_what_is_available():
    with pytest.raises(FactNotFound, match=r"Available years: \[2025, 2026\]"):
        facts(JANUARY_YEAR_END).get("revenue", 2027)
