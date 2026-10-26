"""Tools the LLM can call. Each tool = a Pydantic args model + a Python function.

The Pydantic model does three jobs: it generates the JSON schema we send to the LLM, it validates
the arguments the LLM sends back, and it documents the tool in one place.
"""

from collections.abc import Callable
from functools import cache
from typing import Literal

from pydantic import BaseModel, Field

from edgar_rag import companies, config
from edgar_rag.retriever import search
from edgar_rag.sec_client import OfflineError
from edgar_rag.xbrl import METRICS, PUBLIC_METRICS, CompanyFacts, Fact, FactNotFound

MetricName = Literal[tuple(PUBLIC_METRICS)]  # type: ignore[valid-type]


class Source(BaseModel):
    """Something an answer can cite as [n]."""

    n: int
    kind: Literal["filing", "xbrl"]
    label: str
    ref: str  # chunk id, or accession number + XBRL tag
    url: str
    score: float | None = None  # search similarity, for filing passages


class Sources:
    """Numbers every source the tools return, across all tool calls in one question."""

    def __init__(self) -> None:
        self.items: list[Source] = []

    def add(self, kind: str, label: str, ref: str, url: str, score: float | None = None) -> int:
        for s in self.items:  # same source returned twice keeps its number
            if s.ref == ref:
                return s.n
        n = len(self.items) + 1
        self.items.append(Source(n=n, kind=kind, label=label, ref=ref, url=url, score=score))
        return n

    def cite(self, kind: str, label: str, ref: str, url: str, score: float | None = None) -> str:
        """Add a source and return its citation marker for the tool's text, e.g. "[3]"."""
        return f"[{self.add(kind, label, ref, url, score)}]"


class StagedSources(Sources):
    """Sources for one tool call that runs in parallel with others (W5).

    If parallel calls numbered sources as they went, the numbers would depend on which call
    finished first and change between runs. Instead each call's citations are placeholders
    ("⟦1⟧") until commit(), which numbers them in the order the model requested the calls.
    """

    def cite(self, kind: str, label: str, ref: str, url: str, score: float | None = None) -> str:
        return f"⟦{self.add(kind, label, ref, url, score)}⟧"

    def commit(self, text: str, into: Sources) -> str:
        """Add this call's sources to the question's list and put the final numbers in text."""
        for s in self.items:
            n = into.add(s.kind, s.label, s.ref, s.url, s.score)
            text = text.replace(f"⟦{s.n}⟧", f"[{n}]")  # "⟦1⟧" can't match inside "⟦11⟧"
        return text


# --- Argument models (these become the JSON schemas the LLM sees) ---


class SearchFilingsArgs(BaseModel):
    query: str = Field(description="What to look for, phrased as a question or topic.")
    ticker: str | None = Field(None, description="Limit to one company, e.g. 'AAPL'.")
    item: str | None = Field(
        None,
        description="Limit to one 10-K section: '1' Business, '1A' Risk Factors, "
        "'3' Legal Proceedings, '7' MD&A, '8' Financial Statements.",
    )


class GetMetricArgs(BaseModel):
    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'.")
    metric: MetricName = Field(description="Which financial metric.")
    fiscal_year: int = Field(description="Fiscal year, named by the year it ends in, e.g. 2025.")


class FindCompanyArgs(BaseModel):
    query: str = Field(description="Company name or ticker, e.g. 'Lam Research' or 'LRCX'.")


class CompareMetricArgs(BaseModel):
    metric: MetricName = Field(description="Metric to compare.")
    divide_by: MetricName | None = Field(
        None,
        description="Optional denominator: the result is metric / divide_by as a percentage, "
        "e.g. gross_profit divided by revenue = gross margin.",
    )
    tickers: list[str] | None = Field(
        None, description="Companies to include. Omit to include every indexed company."
    )
    fiscal_year: int | None = Field(
        None, description="Fiscal year. Omit to use each company's latest indexed year."
    )


class CalculateArgs(BaseModel):
    operation: Literal["percent_change", "difference", "ratio", "sum"] = Field(
        description="percent_change = (b - a) / a * 100; difference = b - a; ratio = a / b; sum = a + b."
    )
    a: float = Field(description="First value: the earlier/base value, or the numerator.")
    b: float = Field(description="Second value: the later value, or the denominator.")


# --- Implementations: take validated args, return text for the LLM ---


def _not_allowed(ticker: str | None) -> str | None:
    """In demo mode, an ERROR for any ticker outside the indexed companies (None if allowed)."""
    if ticker is None or not config.demo_mode():
        return None
    indexed = {c.ticker for c in companies.indexed_companies()}
    if ticker.upper() in indexed:
        return None
    return (f"ERROR: {ticker.upper()} is not one of the indexed companies. "
            f"Available: {', '.join(sorted(indexed))}.")



def search_filings(args: SearchFilingsArgs, sources: Sources) -> str:
    if error := _not_allowed(args.ticker):
        return error
    results = search(args.query, k=5, ticker=args.ticker, item=args.item)
    if not results:
        return "No matching passages."
    blocks = []
    for r in results:
        c = r.chunk
        label = f"{c.company} {c.form} ({c.report_date}), Item {c.item}. {c.section_title}"
        cite = sources.cite("filing", label, c.chunk_id, c.source_url, score=r.score)
        blocks.append(f"{cite} {label}\n{c.text}")
    return "\n\n".join(blocks)


@cache
def _facts(ticker: str) -> CompanyFacts:
    """Parsing a company's facts JSON takes a moment; reuse it across tool calls."""
    return CompanyFacts(ticker)


def _cite_fact(fact: Fact, sources: Sources) -> str:
    label = f"SEC XBRL {fact.tag}, {fact.ticker} FY{fact.fiscal_year} ({fact.form} filed {fact.filed})"
    url = f"https://www.sec.gov/Archives/edgar/data/{fact.cik}/{fact.accession_number.replace('-', '')}/"
    return sources.cite("xbrl", label, f"{fact.accession_number}:{fact.tag}:{fact.period_end}", url)


def find_company(args: FindCompanyArgs, sources: Sources) -> str:
    matches = companies.find(args.query)
    if not matches:
        return f"ERROR: no indexed company matches '{args.query}'. {len(companies.indexed_companies())} companies are indexed."
    return "\n".join(
        f"{c.ticker}: {c.name}, latest {c.form} fiscal year {c.fiscal_year} (ends {c.report_date})"
        for c in matches
    )


def compare_metric(args: CompareMetricArgs, sources: Sources) -> str:
    tickers = [t.upper() for t in args.tickers] if args.tickers else [c.ticker for c in companies.indexed_companies()]
    latest = {c.ticker: c.fiscal_year for c in companies.indexed_companies()}
    rows, missing = [], []
    for t in tickers:
        if _not_allowed(t):
            missing.append(f"{t} (not indexed)")
            continue
        year = args.fiscal_year or latest.get(t)
        if year is None:
            missing.append(f"{t} (not indexed)")
            continue
        try:
            fact = _facts(t).get(args.metric, year)
            refs = [_cite_fact(fact, sources)]
            derived = [fact.metric] if fact.tag.startswith("derived") else []
            value = fact.value
            if args.divide_by:
                denom = _facts(t).get(args.divide_by, year)
                refs.append(_cite_fact(denom, sources))
                if denom.tag.startswith("derived"):
                    derived.append(denom.metric)
                if denom.value == 0:
                    missing.append(f"{t} ({args.divide_by} is zero)")
                    continue
                value = fact.value / denom.value * 100
            rows.append((value, t, year, fact.period_end, refs, derived))
        except (FactNotFound, ValueError, OfflineError) as e:
            missing.append(f"{t} ({e})")

    rows.sort(key=lambda r: r[0], reverse=True)
    what = f"{args.metric} / {args.divide_by} (%)" if args.divide_by else f"{args.metric} ({METRICS[args.metric][1]})"
    lines = [f"{what}, highest first:"]
    for rank, (value, t, year, end, refs, derived) in enumerate(rows, 1):
        shown = f"{value:.2f}%" if args.divide_by else f"{value:,.2f}"
        cites = "".join(refs)
        note = f" (derived {', '.join(derived)}: company reports no such line)" if derived else ""
        lines.append(f"{rank}. {t} FY{year} (year ended {end}): {shown} {cites}{note}")
    if missing:
        lines.append("Not available: " + "; ".join(missing))
    return "\n".join(lines)


def get_financial_metric(args: GetMetricArgs, sources: Sources) -> str:
    if error := _not_allowed(args.ticker):
        return error
    try:
        fact = _facts(args.ticker.upper()).get(args.metric, args.fiscal_year)
    except FactNotFound as e:
        return f"ERROR: {e}"
    except ValueError as e:  # unknown ticker
        return f"ERROR: {e}"
    except OfflineError:
        return f"ERROR: no data for {args.ticker.upper()} (live SEC access is off)."
    cite = _cite_fact(fact, sources)
    period = f"{fact.period_start} to {fact.period_end}" if fact.period_start else f"as of {fact.period_end}"
    # The LLM only sees this text, so a derived value must be labelled here, not just in the source.
    derived = f" [{fact.label}]" if fact.tag.startswith("derived") else ""
    return (
        f"{cite} {fact.company} {fact.metric} FY{fact.fiscal_year} ({period}): "
        f"{fact.value:,.2f} {fact.unit}{derived}"
    )


def calculate(args: CalculateArgs, sources: Sources) -> str:
    a, b = args.a, args.b
    if (args.operation == "percent_change" and a == 0) or (args.operation == "ratio" and b == 0):
        return "ERROR: division by zero"
    result = {
        "percent_change": lambda: (b - a) / a * 100,
        "difference": lambda: b - a,
        "ratio": lambda: a / b,
        "sum": lambda: a + b,
    }[args.operation]()
    suffix = "%" if args.operation == "percent_change" else ""
    return f"{args.operation}(a={a:,.2f}, b={b:,.2f}) = {result:,.4f}{suffix}"


class Tool(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel, Sources], str]

    def schema_for_llm(self) -> dict:
        """The OpenAI-style function definition sent with every request."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            name="search_filings",
            description="Search 10-K filing text for passages about a topic. Use for explanations, "
            "risks, strategy, legal matters, and anything descriptive. Returns numbered sources.",
            args_model=SearchFilingsArgs,
            fn=search_filings,
        ),
        Tool(
            name="get_financial_metric",
            description="Get an exact reported figure from the company's SEC XBRL data for one "
            "fiscal year. Use for ANY financial number. Returns a numbered source.",
            args_model=GetMetricArgs,
            fn=get_financial_metric,
        ),
        Tool(
            name="compare_metric",
            description="Compare one metric across many companies in one call, ranked highest "
            "first, each with a numbered source. With divide_by it computes a percentage "
            "(e.g. gross_profit / revenue = gross margin). Use for rankings and peer comparisons "
            "instead of many get_financial_metric calls.",
            args_model=CompareMetricArgs,
            fn=compare_metric,
        ),
        Tool(
            name="find_company",
            description="Look up indexed companies by name or ticker. Returns tickers and each "
            "company's latest fiscal year. Use when you don't know a ticker.",
            args_model=FindCompanyArgs,
            fn=find_company,
        ),
        Tool(
            name="calculate",
            description="Do arithmetic exactly. Use for growth rates, differences, margins and "
            "ratios instead of calculating yourself.",
            args_model=CalculateArgs,
            fn=calculate,
        ),
    ]
}
