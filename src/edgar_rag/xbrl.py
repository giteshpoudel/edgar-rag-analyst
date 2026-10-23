"""Exact financial figures from SEC XBRL company facts.

Facts are reported per filing, so the same number appears in several filings and every 10-K
repeats prior years. We select by the period's dates (not the `fy` field, which is the filing's
fiscal year) and prefer the most recently filed value, which includes any restatement.
"""

from datetime import date

from pydantic import BaseModel

from edgar_rag.sec_client import SecClient

# Friendly metric name -> (XBRL tags to try in order, unit, kind).
# "duration" = an amount over the fiscal year; "instant" = a balance on the fiscal year-end date.
# Companies (and the same company over time) use different tags for the same concept.
METRICS: dict[str, tuple[list[str], str, str]] = {
    "revenue": (
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        "USD",
        "duration",
    ),
    "cost_of_revenue": (["CostOfGoodsAndServicesSold", "CostOfRevenue"], "USD", "duration"),
    "gross_profit": (["GrossProfit"], "USD", "duration"),
    "research_and_development": (["ResearchAndDevelopmentExpense"], "USD", "duration"),
    "operating_income": (["OperatingIncomeLoss"], "USD", "duration"),
    # ProfitLoss includes non-controlling interests; used only when NetIncomeLoss is absent (Broadcom).
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD", "duration"),
    "eps_diluted": (["EarningsPerShareDiluted"], "USD/shares", "duration"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities"], "USD", "duration"),
    "total_assets": (["Assets"], "USD", "instant"),
    "total_liabilities": (["Liabilities"], "USD", "instant"),
    "stockholders_equity": (
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "USD",
        "instant",
    ),
    "cash_and_equivalents": (["CashAndCashEquivalentsAtCarryingValue"], "USD", "instant"),
    # Building blocks for derived metrics; not offered to the LLM (leading underscore).
    "_liabilities_and_equity": (["LiabilitiesAndStockholdersEquity"], "USD", "instant"),
    "_total_equity": (
        ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
        "USD",
        "instant",
    ),
}

# Metrics the LLM can ask for.
PUBLIC_METRICS = [m for m in METRICS if not m.startswith("_")]

# Many filers don't report every line: balance sheets often show only "total liabilities and
# equity" (AMD, Intel, ADI, Microchip), and some income statements have no gross-profit line
# (Qualcomm, KLA). When no tag is reported, compute metric = a - b from reported facts, and
# label the result as derived so answers stay honest about the source.
DERIVED: dict[str, tuple[str, str]] = {
    "gross_profit": ("revenue", "cost_of_revenue"),
    "total_liabilities": ("_liabilities_and_equity", "_total_equity"),
}


class Fact(BaseModel):
    ticker: str
    cik: int
    company: str
    metric: str
    tag: str
    label: str
    fiscal_year: int
    period_start: str | None  # None for instant (balance sheet) values
    period_end: str
    value: float
    unit: str
    form: str
    accession_number: str
    filed: str


class FactNotFound(Exception):
    pass


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _is_annual(raw: dict) -> bool:
    return "start" in raw and 350 <= _days(raw["start"], raw["end"]) <= 380


class CompanyFacts:
    def __init__(self, ticker: str, client: SecClient | None = None) -> None:
        self.ticker = ticker.upper()
        data = (client or SecClient()).company_facts(self.ticker)
        self.company: str = data["entityName"]
        self.cik: int = int(data["cik"])
        self._gaap: dict = data["facts"].get("us-gaap", {})
        self._year_ends = self._fiscal_year_ends()

    def _annual_facts(self, tag: str, unit: str) -> list[dict]:
        """Full-year values from annual reports, one per period (latest filing wins)."""
        raw = self._gaap.get(tag, {}).get("units", {}).get(unit, [])
        by_period: dict[tuple, dict] = {}
        for f in raw:
            if f.get("form") not in ("10-K", "10-K/A"):
                continue
            key = (f.get("start"), f["end"])
            if key not in by_period or f["filed"] > by_period[key]["filed"]:
                by_period[key] = f
        return list(by_period.values())

    def _fiscal_year_ends(self) -> dict[int, str]:
        """Map fiscal year -> its last day, using annual periods of common income tags.

        A fiscal year is named by the calendar year it starts in, the year most of it falls in
        (NVIDIA's year ending 2026-01-25 is mostly 2025, so it maps to 2025).
        """
        ends: dict[int, str] = {}
        for tag in [*METRICS["net_income"][0], *METRICS["revenue"][0]]:
            for f in self._annual_facts(tag, "USD"):
                if _is_annual(f):
                    ends.setdefault(int(f["start"][:4]), f["end"])
        return ends

    def available_years(self, metric: str) -> list[int]:
        return sorted(y for y in self._year_ends if self._find_or_derive(metric, y))

    def _find_or_derive(self, metric: str, fiscal_year: int) -> bool:
        if self._find(metric, fiscal_year):
            return True
        parts = DERIVED.get(metric)
        return bool(parts) and all(self._find(p, fiscal_year) for p in parts)

    def _find(self, metric: str, fiscal_year: int) -> tuple[dict, str] | None:
        tags, unit, kind = METRICS[metric]
        year_end = self._year_ends.get(fiscal_year)
        if not year_end:
            return None
        for tag in tags:
            for f in self._annual_facts(tag, unit):
                if f["end"] != year_end:
                    continue
                if kind == "duration" and _is_annual(f):
                    return f, tag
                if kind == "instant" and "start" not in f:
                    return f, tag
        return None

    def _fact(self, metric: str, f: dict, tag: str, label: str, fiscal_year: int, value: float) -> Fact:
        return Fact(
            ticker=self.ticker,
            cik=self.cik,
            company=self.company,
            metric=metric,
            tag=tag,
            label=label,
            fiscal_year=fiscal_year,
            period_start=f.get("start"),
            period_end=f["end"],
            value=value,
            unit=METRICS[metric][1],
            form=f["form"],
            accession_number=f["accn"],
            filed=f["filed"],
        )

    def get(self, metric: str, fiscal_year: int) -> Fact:
        if metric not in METRICS:
            raise FactNotFound(f"Unknown metric '{metric}'. Known: {', '.join(PUBLIC_METRICS)}")
        found = self._find(metric, fiscal_year)
        if found:
            f, tag = found
            return self._fact(metric, f, tag, self._gaap[tag]["label"], fiscal_year, f["val"])

        parts = DERIVED.get(metric)
        if parts:
            a, b = (self._find(p, fiscal_year) for p in parts)
            if a and b:
                (fa, tag_a), (fb, tag_b) = a, b
                tag = f"derived: {tag_a} - {tag_b}"
                label = f"Derived ({self.company} reports no {metric} line): {tag_a} minus {tag_b}"
                return self._fact(metric, fa, tag, label, fiscal_year, fa["val"] - fb["val"])

        years = self.available_years(metric)
        raise FactNotFound(f"No {metric} for {self.ticker} FY{fiscal_year}. Available years: {years}")
