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


if __name__ == "__main__":
    app()
