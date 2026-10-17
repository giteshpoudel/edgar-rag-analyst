"""Command-line interface: `edgar <command>`."""

import typer

from edgar_rag import config
from edgar_rag.chunking import chunk_filing
from edgar_rag.sec_client import Filing, SecClient

app = typer.Typer(help="Ask questions about SEC filings.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """EDGAR RAG Analyst."""


@app.command()
def download(ticker: str, form: str = "10-K") -> None:
    """Download and cache the latest filing for TICKER (e.g. AAPL)."""
    client = SecClient()
    filing = client.latest_filing(ticker, form)
    path = client.download(filing)
    size_mb = path.stat().st_size / 1_000_000
    typer.echo(f"{filing.company} {filing.form} for period {filing.report_date}")
    typer.echo(f"Filed {filing.filing_date} · accession {filing.accession_number}")
    typer.echo(f"Source: {filing.url}")
    typer.echo(f"Saved:  {path} ({size_mb:.1f} MB)")


@app.command()
def chunk(ticker: str) -> None:
    """Split each downloaded filing for TICKER into chunks (saved as JSONL)."""
    raw_dir = config.data_dir() / "raw" / ticker.upper()
    out_dir = config.data_dir() / "processed" / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(raw_dir.glob("*.html"))
    if not html_files:
        raise typer.BadParameter(f"No filings in {raw_dir}. Run `edgar download {ticker}` first.")

    for html_path in html_files:
        filing = Filing.model_validate_json(html_path.with_suffix(".json").read_text())
        chunks = chunk_filing(html_path.read_text(encoding="utf-8"), filing)
        out_path = out_dir / f"{html_path.stem}.chunks.jsonl"
        out_path.write_text("".join(c.model_dump_json() + "\n" for c in chunks))

        sizes = [len(c.text) for c in chunks]
        sections = len({c.item for c in chunks})
        typer.echo(f"{html_path.name}: {len(chunks)} chunks across {sections} sections")
        typer.echo(f"  chars per chunk: min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)}")
        typer.echo(f"  saved: {out_path}")


@app.command()
def index(ticker: str) -> None:
    """Embed TICKER's chunks and store them in the vector database."""
    from edgar_rag.index import index_chunks, load_chunks

    files = sorted((config.data_dir() / "processed" / ticker.upper()).glob("*.chunks.jsonl"))
    if not files:
        raise typer.BadParameter(f"No chunks for {ticker}. Run `edgar chunk {ticker}` first.")
    for path in files:
        count = index_chunks(load_chunks(path))
        typer.echo(f"Indexed {count} chunks from {path.name}")


@app.command()
def search(
    question: str,
    k: int = typer.Option(5, help="Number of results"),
    ticker: str | None = typer.Option(None, help="Only this company, e.g. AAPL"),
    item: str | None = typer.Option(None, help="Only this section, e.g. 1A"),
) -> None:
    """Show the chunks most relevant to QUESTION (retrieval only, no LLM)."""
    from edgar_rag.retriever import search as run_search

    for rank, result in enumerate(run_search(question, k=k, ticker=ticker, item=item), 1):
        c = result.chunk
        preview = c.text[:240].replace("\n", " ")
        typer.echo(f"\n#{rank}  score {result.score:.3f}  {c.ticker} {c.form} {c.report_date}")
        typer.echo(f"    Item {c.item}. {c.section_title}  [{c.chunk_id}]")
        typer.echo(f"    {preview}...")


@app.command()
def ask(
    question: str,
    simple: bool = typer.Option(False, help="Phase 1 RAG only: one search, no tools"),
    k: int = typer.Option(5, help="(--simple) Number of chunks to retrieve"),
    ticker: str | None = typer.Option(None, help="(--simple) Only this company, e.g. AAPL"),
    item: str | None = typer.Option(None, help="(--simple) Only this section, e.g. 1A"),
) -> None:
    """Answer QUESTION from the filings, with citations."""
    if simple:
        _ask_simple(question, k, ticker, item)
        return

    from edgar_rag.agent import ask as run_agent

    result = run_agent(question)
    typer.echo("\nTool calls:")
    for call in result.tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
        first_line = call.result.splitlines()[0] if call.result else ""
        typer.echo(f"  step {call.step}: {call.name}({args})")
        typer.echo(f"          -> {first_line[:110]}")
    if not result.tool_calls:
        typer.echo("  (none)")

    typer.echo(f"\n{result.text}\n")
    typer.echo("Sources:")
    for s in result.sources:
        mark = "*" if s.n in result.cited else " "
        typer.echo(f" {mark}[{s.n}] {s.label[:95]}")
    typer.echo("  (* = cited in the answer)")
    if result.invalid_citations:
        typer.echo(f"  WARNING: answer cites sources that don't exist: {result.invalid_citations}")
    typer.echo(
        f"\n{result.steps} LLM calls · {result.prompt_tokens} prompt + "
        f"{result.completion_tokens} completion tokens"
    )


@app.command(name="eval")
def run_eval(
    mode: str = typer.Option("both", help="agent, simple, or both"),
    only: str | None = typer.Option(None, help="Comma-separated question ids, e.g. num-01,nar-02"),
    workers: int = typer.Option(4, help="Questions answered in parallel"),
) -> None:
    """Run the golden question set, score it, and write evals/results/."""
    from edgar_rag import evaluation

    items = evaluation.load_golden()
    if only:
        wanted = set(only.split(","))
        items = [i for i in items if i.id in wanted]
    modes = ["agent", "simple"] if mode == "both" else [mode]

    summaries = {}
    for m in modes:
        typer.echo(f"Running {len(items)} questions in {m} mode...")
        records = evaluation.run(m, items, workers=workers)
        path = evaluation.save(records, m)
        summaries[m] = evaluation.summarize(records)
        failed = [r.id for r in records if not r.passed]
        typer.echo(f"  {sum(r.passed for r in records)}/{len(records)} passed · saved {path}")
        if failed:
            typer.echo(f"  failed: {', '.join(failed)}")

    table = evaluation.summary_markdown(summaries, config.llm_model())
    if not only:  # partial runs don't overwrite the committed summary
        (evaluation.EVALS_DIR / "results" / "summary.md").write_text(table)
    typer.echo("\n" + table)


def _ask_simple(question: str, k: int, ticker: str | None, item: str | None) -> None:
    from edgar_rag.answer import answer as run_answer

    result = run_answer(question, k=k, ticker=ticker, item=item)
    typer.echo(f"\n{result.text}\n")
    typer.echo("Sources:")
    for n, source in enumerate(result.sources, 1):
        c = source.chunk
        mark = "*" if n in result.cited else " "
        typer.echo(
            f" {mark}[{n}] {c.ticker} {c.form} {c.report_date} · Item {c.item}. "
            f"{c.section_title[:50]} · score {source.score:.3f} · {c.chunk_id}"
        )
    typer.echo("  (* = cited in the answer)")
    if result.invalid_citations:
        typer.echo(f"  WARNING: answer cites sources that don't exist: {result.invalid_citations}")
    typer.echo(
        f"\n{result.llm.model} · {result.llm.prompt_tokens} prompt + "
        f"{result.llm.completion_tokens} completion tokens"
    )


if __name__ == "__main__":
    app()
