"""Tools the LLM can call. Each tool = a Pydantic args model + a Python function.

The Pydantic model does three jobs: it generates the JSON schema we send to the LLM, it validates
the arguments the LLM sends back, and it documents the tool in one place.
"""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from edgar_rag.retriever import search
from edgar_rag.xbrl import METRICS, CompanyFacts, FactNotFound

MetricName = Literal[tuple(METRICS)]  # type: ignore[valid-type]


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


class CalculateArgs(BaseModel):
    operation: Literal["percent_change", "difference", "ratio", "sum"] = Field(
        description="percent_change = (b - a) / a * 100; difference = b - a; ratio = a / b; sum = a + b."
    )
    a: float = Field(description="First value: the earlier/base value, or the numerator.")
    b: float = Field(description="Second value: the later value, or the denominator.")


# --- Implementations: take validated args, return text for the LLM ---


def search_filings(args: SearchFilingsArgs, sources: Sources) -> str:
    results = search(args.query, k=5, ticker=args.ticker, item=args.item)
    if not results:
        return "No matching passages."
    blocks = []
    for r in results:
        c = r.chunk
        label = f"{c.company} {c.form} ({c.report_date}), Item {c.item}. {c.section_title}"
        n = sources.add("filing", label, c.chunk_id, c.source_url, score=r.score)
        blocks.append(f"[{n}] {label}\n{c.text}")
    return "\n\n".join(blocks)


def get_financial_metric(args: GetMetricArgs, sources: Sources) -> str:
    try:
        fact = CompanyFacts(args.ticker).get(args.metric, args.fiscal_year)
    except FactNotFound as e:
        return f"ERROR: {e}"
    except ValueError as e:  # unknown ticker
        return f"ERROR: {e}"
    label = f"SEC XBRL {fact.tag}, {fact.ticker} FY{fact.fiscal_year} ({fact.form} filed {fact.filed})"
    url = f"https://www.sec.gov/Archives/edgar/data/{fact.cik}/{fact.accession_number.replace('-', '')}/"
    n = sources.add("xbrl", label, f"{fact.accession_number}:{fact.tag}:{fact.period_end}", url)
    period = f"{fact.period_start} to {fact.period_end}" if fact.period_start else f"as of {fact.period_end}"
    return (
        f"[{n}] {fact.company} {fact.metric} FY{fact.fiscal_year} ({period}): "
        f"{fact.value:,.2f} {fact.unit}"
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
            name="calculate",
            description="Do arithmetic exactly. Use for growth rates, differences, margins and "
            "ratios instead of calculating yourself.",
            args_model=CalculateArgs,
            fn=calculate,
        ),
    ]
}
