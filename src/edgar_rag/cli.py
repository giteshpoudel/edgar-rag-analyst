"""Command-line interface: `edgar <command>`."""

from pathlib import Path

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
def ingest(tickers: str = typer.Argument(..., help="Comma-separated, e.g. AMD,INTC,QCOM")) -> None:
    """Download, chunk and index the latest 10-K for each ticker, with a health report."""
    from edgar_rag.pipeline import ingest as run_ingest

    results = run_ingest([t.strip() for t in tickers.split(",") if t.strip()])
    typer.echo(f"\n{'ticker':6} {'FY':>4} {'chunks':>6} {'sect':>4} {'metrics':>7}  status")
    for h in results:
        if h.error:
            typer.echo(f"{h.ticker:6} {'':>4} {'':>6} {'':>4} {'':>7}  ERROR {h.error[:80]}")
            continue
        status = "ok" if h.ok else "; ".join(h.warnings)
        typer.echo(
            f"{h.ticker:6} {h.fiscal_year:>4} {h.chunks:>6} {h.sections:>4} "
            f"{h.metrics_found:>4}/{h.metrics_found + len(h.metrics_missing):<2}  {status}"
        )
    typer.echo(f"\n{sum(h.ok for h in results)}/{len(results)} filings passed all checks")


@app.command(name="ingest-multi")
def ingest_multi(
    tickers: str = typer.Argument("all", help="Comma-separated, or 'all' for every v2 company"),
    years: int = typer.Option(5, help="Fiscal years per company, ending at the 10-K v2 indexed"),
    report: str = typer.Option("evals/results/v3/ingest_health.md", help="Where to write the health report"),
) -> None:
    """v3: index several years of 10-Ks per company into chunks_multi (v2's table is untouched)."""
    from pathlib import Path

    from edgar_rag import companies
    from edgar_rag.pipeline import ingest_years

    names = ([c.ticker for c in companies.indexed_companies()] if tickers == "all"
             else [t.strip().upper() for t in tickers.split(",") if t.strip()])
    results = ingest_years(names, years)
    lines = [f"Multi-year ingest health ({len(names)} companies × {years} fiscal years)", "",
             "| Ticker | FY | Report date | Chunks | Sections | Item 1A chars | Item 7 chars | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for h in results:
        status = f"ERROR {h.error[:100]}" if h.error else ("ok" if h.ok else "; ".join(h.warnings))
        if h.matches_v2:
            status += " (= v2)"
        lines.append(f"| {h.ticker} | {h.fiscal_year or ''} | {h.report_date} | {h.chunks} | {h.sections} | "
                     f"{h.item_chars.get('1A', 0):,} | {h.item_chars.get('7', 0):,} | {status} |")
    lines += ["", (f"{sum(h.ok for h in results)}/{len(results)} filings passed all checks; "
              f"{sum(h.reused for h in results)} reused from an earlier run.")]
    text = "\n".join(lines) + "\n"
    typer.echo(text)
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(text)


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
    import time

    from edgar_rag import tracing

    start = time.monotonic()
    with tracing.span("ask", "CHAIN", **{"input.value": question, "edgar.simple": simple}):
        trace_id = tracing.current_trace_id()
        if simple:
            _ask_simple(question, k, ticker, item)
        else:
            _ask_agent(question)
    _print_usage(tracing.COLLECTOR.pop(trace_id), time.monotonic() - start, tracing.trace_id_hex(trace_id))


def _print_usage(usage, seconds: float, trace_id: str) -> None:
    cost = f"${usage.cost_usd:.4f}" if usage.cost_known else "cost unknown (model not in price table)"
    llm = usage.seconds.get("LLM", 0.0)
    typer.echo(
        f"\n{usage.llm_calls} LLM calls · {usage.prompt_tokens:,} prompt ({usage.cached_prompt_tokens:,} cached) + "
        f"{usage.completion_tokens:,} completion tokens ({usage.reasoning_tokens:,} reasoning)"
    )
    typer.echo(f"{cost} at list price · {seconds:.1f} s total, {llm:.1f} s in LLM calls")
    if config.tracing() == "phoenix":
        typer.echo(f"Trace {trace_id} in Phoenix ({config.phoenix_endpoint().removesuffix('/v1/traces')})")


def _ask_agent(question: str) -> None:
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


def _load_items(golden: str, split: str | None, only: str | None):
    from edgar_rag import evaluation

    if golden not in evaluation.GOLDEN_SETS:
        raise typer.BadParameter(f"golden must be one of {', '.join(evaluation.GOLDEN_SETS)}")
    items = evaluation.load_golden(evaluation.GOLDEN_SETS[golden])
    if split:
        items = [i for i in items if i.split in split.split(",")]
    if only:
        wanted = set(only.split(","))
        items = [i for i in items if i.id in wanted]
    return items


@app.command(name="eval")
def run_eval(
    mode: str = typer.Option("both", help="agent, simple, or both"),
    golden: str = typer.Option("v1", help="Question set: v1 (58) or v2 (198)"),
    split: str | None = typer.Option(None, help="v2 only: dev, test, regression (comma-separated)"),
    only: str | None = typer.Option(None, help="Comma-separated question ids, e.g. num-01,nar-02"),
    workers: int = typer.Option(16, "--concurrency", "--workers", help="Questions in flight at once"),
    repeat: int = typer.Option(1, help="Runs per mode; >1 reports mean, range and flip rate"),
) -> None:
    """Run a golden question set, score it, and write the results."""
    from edgar_rag import evaluation

    items = _load_items(golden, split, only)
    modes = ["agent", "simple"] if mode == "both" else [mode]
    out_dir = evaluation.results_dir(golden)
    tag = f"_{split.replace(',', '+')}" if split else ""
    # Name results after a non-default retrieval config, so runs never overwrite the baseline.
    retrieval = config.retrieval_mode() + ("+" + config.reranker() if config.reranker() != "none" else "")
    if retrieval != "hybrid":  # the default since W2; v1-style results are tagged "_vector"
        tag += f"_{retrieval}"
    if config.grounding_guard() == "off":  # the guard is on by default since W3
        tag += "_noguard"
    typer.echo(f"Retrieval: {retrieval}")
    if repeat > 1:
        _repeat_runs(modes, items, workers, repeat, out_dir, tag)
        return

    import time

    summaries, timings = {}, {}
    for m in modes:
        typer.echo(f"Running {len(items)} questions in {m} mode...")
        started = time.monotonic()
        records = evaluation.run(m, items, workers=workers)
        wall = time.monotonic() - started
        path = evaluation.save(records, m, out_dir, tag)
        summaries[m] = evaluation.summarize(records)
        timings[m] = f"{wall:.0f} s wall time at concurrency {workers}"
        typer.echo(f"  {timings[m]} ({len(records) / wall * 60:.0f} questions/minute)")
        failed = [r.id for r in records if not r.passed]
        typer.echo(f"  {sum(r.passed for r in records)}/{len(records)} passed · saved {path}")
        if failed:
            typer.echo(f"  failed: {', '.join(failed)}")

    table = evaluation.summary_markdown(summaries, config.llm_model(), config.judge_model(),
                                        "; ".join(f"{m}: {s}" for m, s in timings.items()))
    if not only:  # partial runs don't overwrite the committed summary
        (out_dir / f"summary{tag}.md").write_text(table)
    typer.echo("\n" + table)


def _repeat_runs(modes, items, workers, repeat, out_dir, tag) -> None:
    import json
    import time

    from edgar_rag import evaluation
    from edgar_rag.stats import summarize_runs

    for m in modes:
        runs, pooled = [], []
        for k in range(1, repeat + 1):
            typer.echo(f"{m} run {k}/{repeat}: {len(items)} questions...")
            started = time.monotonic()
            records = evaluation.run(m, items, workers=workers)
            wall = time.monotonic() - started
            evaluation.save(records, m, out_dir, f"{tag}_run{k}")
            runs.append([json.loads(r.model_dump_json()) for r in records])
            pooled += records
            typer.echo(f"  {sum(r.passed for r in records)}/{len(records)} passed · {wall:.0f} s wall · "
                       f"{evaluation.cost_line(records)}")
        s = summarize_runs(runs)
        typer.echo(
            f"{m}: mean {s.mean:.1%} (range {s.low:.1%}-{s.high:.1%}) over {s.runs} runs; "
            f"{s.flip_rate:.1%} of questions changed result between runs"
        )
        typer.echo(f"  all runs: {evaluation.cost_line(pooled)}")
        report = {**s.model_dump(), **evaluation.cost_and_latency(pooled)}
        (out_dir / f"{m}{tag}_runs.json").write_text(json.dumps(report, indent=2))


@app.command()
def compare(
    a: str = typer.Argument(..., help="Result files for config A, a glob such as 'agent_dev+test_run*'"),
    b: str = typer.Argument(..., help="Result files for config B"),
    golden: str = typer.Option("v2", help="Which results folder"),
    split: str | None = typer.Option(None, help="Only these splits (dev, test, ...)"),
) -> None:
    """Paired bootstrap 95% CI for the pass-rate difference B - A."""
    import json

    from edgar_rag import evaluation
    from edgar_rag.stats import paired_bootstrap, summarize_runs

    out_dir = evaluation.results_dir(golden)
    keep = None
    if split:
        keep = {i.id for i in _load_items(golden, split, None)}

    def load(pattern):
        files = sorted(out_dir.glob(pattern + ("" if pattern.endswith(".jsonl") else ".jsonl")))
        if not files:
            raise typer.BadParameter(f"no result files match {pattern} in {out_dir}")
        runs = [[json.loads(line) for line in f.read_text().splitlines() if line] for f in files]
        return [[r for r in run if keep is None or r["id"] in keep] for run in runs], files

    runs_a, files_a = load(a)
    runs_b, files_b = load(b)
    for name, runs, files in (("A", runs_a, files_a), ("B", runs_b, files_b)):
        s = summarize_runs(runs)
        records = [evaluation.Record.model_validate(r) for run in runs for r in run]
        typer.echo(f"{name}: {', '.join(f.name for f in files)}: mean {s.mean:.1%} "
                   f"(range {s.low:.1%}-{s.high:.1%}), flip rate {s.flip_rate:.1%} · "
                   f"{evaluation.cost_line(records)}")
    c = paired_bootstrap(runs_a, runs_b)
    verdict = "significant" if c.significant else "not significant (could be noise)"
    typer.echo(f"B - A = {c.diff:+.1%}, 95% CI [{c.ci_low:+.1%}, {c.ci_high:+.1%}] over {c.questions} questions: {verdict}")


@app.command(name="make-baseline")
def make_baseline(
    runs: str = typer.Argument(..., help="Repeat-run result files, a glob such as 'agent_dev_run*'"),
    golden: str = typer.Option("v2", help="Which results folder"),
    split: str = typer.Option("dev", help="The split those runs cover (the gate's smoke set)"),
) -> None:
    """Build evals/baseline.json for the CI gate from repeated runs of the same code."""
    from edgar_rag import evaluation, gate

    files = sorted(evaluation.results_dir(golden).glob(runs + ("" if runs.endswith(".jsonl") else ".jsonl")))
    if len(files) < 3:
        raise typer.BadParameter("need at least 3 runs to measure run-to-run spread")
    loaded = [[evaluation.Record.model_validate_json(line) for line in f.read_text().splitlines() if line]
              for f in files]
    cfg = {"model": config.llm_model(), "judge": config.judge_model(),
           "retrieval": config.retrieval_mode(), "reranker": config.reranker(),
           "grounding_guard": config.grounding_guard(), "files": [f.name for f in files]}
    baseline = gate.make_baseline(loaded, golden, split, cfg)
    gate.save_baseline(baseline)
    typer.echo(f"Wrote {gate.BASELINE_PATH}: pass rate {baseline.pass_rate:.1%} over {baseline.runs} runs "
               f"{[f'{r:.1%}' for r in baseline.pass_rates]}, tolerance {baseline.tolerance:.1%}, "
               f"number accuracy floor {baseline.number_accuracy_floor:.1%}")


@app.command(name="gate")
def run_gate(
    concurrency: int = typer.Option(16, help="Questions in flight at once"),
    out: str = typer.Option("evals/results/ci", help="Where to save this run's records"),
) -> None:
    """Run the smoke set and compare with evals/baseline.json; exit 1 if it regressed (CI)."""
    import os
    from pathlib import Path

    from edgar_rag import evaluation, gate

    baseline = gate.load_baseline()
    items = _load_items(baseline.golden, baseline.split, None)
    if len(items) != baseline.questions:
        typer.echo(f"Note: smoke set has {len(items)} questions, baseline was built on {baseline.questions}")
    records = evaluation.run("agent", items, workers=concurrency)
    evaluation.save(records, "agent", Path(out), "_gate")
    result = gate.check(records, baseline)
    report = gate.markdown(result, baseline, records)
    typer.echo(report)
    summary = os.getenv("GITHUB_STEP_SUMMARY")  # GitHub Actions shows this file on the run page
    if summary:
        with open(summary, "a") as f:
            f.write(report)
    raise typer.Exit(0 if result.passed else 1)


@app.command(name="label-pack")
def label_pack(
    ticker: str = typer.Argument(..., help="e.g. NVDA"),
    later_year: int = typer.Argument(..., help="The later fiscal year of the pair, e.g. 2026"),
    split: str = typer.Option(..., help="dev or test"),
    share: float = typer.Option(0.35, help="Share of later-year blocks to label (1 = everything)"),
    seed: int = typer.Option(3, help="Random seed for the block sample"),
    pack_dir: str = typer.Option("data/v3/label_packs", help="Where to write the readable packs"),
) -> None:
    """v3: write label packs and empty label CSVs for one company's two consecutive years."""
    import json

    from edgar_rag import labeling

    ticker = ticker.upper()
    out = labeling.make_pair_packs(ticker, later_year, share, seed, Path(pack_dir))
    manifest = json.loads(labeling.MANIFEST.read_text()) if labeling.MANIFEST.exists() else {"pairs": []}
    manifest["pairs"] = [p for p in manifest["pairs"] if not (p["ticker"] == ticker and p["later_year"] == later_year)]
    manifest["pairs"].append({"ticker": ticker, "later_year": later_year, "split": split, "share": share,
                              "seed": seed, "years": {str(k): v for k, v in out.items()}})
    labeling.MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    for fy, v in out.items():
        typer.echo(f"{ticker} FY{fy}: {len(v['chosen'])}/{v['blocks']} blocks, {v['words']:,} of "
                   f"{v['all_words']:,} words -> {v['pack']}, labels in {v['labels']}")


@app.command(name="label-check")
def label_check() -> None:
    """v3: check every labelled filing in evals/v3/labels/manifest.json."""
    import json

    from edgar_rag import labeling

    manifest = json.loads(labeling.MANIFEST.read_text())
    bad = 0
    for p in manifest["pairs"]:
        t, later = p["ticker"], p["later_year"]
        rows = {}
        for fy in (later, later - 1):
            path = Path(p["years"][str(fy)]["labels"])
            rows[fy] = labeling.read_labels(path) if path.exists() else []
        targets = {r.prior_claim_id for r in rows[later] if r.prior_claim_id}
        for fy, prior in ((later, rows[later - 1]), (later - 1, None)):
            _, html = labeling.filing_html(t, fy)
            blocks = labeling.blocks_for_filing(html)
            if len(blocks) != p["years"][str(fy)]["blocks"]:
                typer.echo(f"{t} FY{fy}: block count changed since the pack was made; regenerate it")
            problems = labeling.check_labels(t, fy, set(p["years"][str(fy)]["chosen"]), blocks, rows[fy],
                                             prior, targets if prior is None else frozenset())
            bad += len(problems)
            typer.echo(f"{t} FY{fy} ({p['split']}): {len(rows[fy])} claims, {len(problems)} problems")
            for x in problems:
                typer.echo(f"   - {x}")
    raise typer.Exit(1 if bad else 0)


@app.command(name="extract")
def extract(
    targets: str = typer.Argument(..., help="TICKER:FY pairs, comma-separated (e.g. MU:2026), or 'all'"),
    peak: bool = typer.Option(False, help="Allow running in DeepSeek's peak hours (double price)"),
    approved: bool = typer.Option(False, help="The owner approved a run estimated over $1"),
    out_tokens: int = typer.Option(3000, help="Assumed output tokens per window, for the estimate"),
) -> None:
    """v3: extract claims from Items 1A and 7, with a cost estimate first (cached windows are free)."""
    import asyncio

    from edgar_rag import budget, claims, companies, config
    from edgar_rag.labeling import blocks_for_filing, filing_html
    from edgar_rag.llm_cache import LlmCache, sha256_text

    if targets == "all":
        pairs = [(c.ticker, c.fiscal_year - k) for c in companies.indexed_companies() for k in range(5)]
    else:
        pairs = [(t.split(":")[0].upper(), int(t.split(":")[1])) for t in targets.split(",")]
    cache, version = LlmCache(), claims.extractor_version()
    _, miss, out = config.price(claims.MODEL)
    est, n_windows, n_cached = 0.0, 0, 0
    for t, fy in pairs:
        _, html = filing_html(t, fy)
        for w in claims.windows(blocks_for_filing(html)):
            part = f"{w[0].item}:{w[0].number}-{w[-1].number}"
            n_windows += 1
            if cache.get(sha256_text(html), version, claims.MODEL, part) is not None:
                n_cached += 1
                continue
            prompt_tokens = (len(claims.window_text(w)) + len(claims.system_prompt())) / 4
            est += (prompt_tokens * miss + out_tokens * out) / 1e6
    offpeak = budget.is_offpeak()
    typer.echo(f"Extractor {version[:12]}: {len(pairs)} filings, {n_windows} windows ({n_cached} cached). "
               f"Estimate ${est:.2f} at list price (${est / 2:.2f} off-peak); now is "
               f"{'off-peak' if offpeak else 'PEAK'}.")
    if est > 1.0 and not approved:
        typer.echo("Over $1: needs the owner's go-ahead (--approved).")
        raise typer.Exit(1)
    if not offpeak and not peak and est > 0:
        typer.echo("DeepSeek peak hours (01-04 and 06-10 UTC weekdays): run later or pass --peak.")
        raise typer.Exit(1)

    async def run_all():
        return [await claims.extract_filing(t, fy, cache) for t, fy in pairs]

    for x in asyncio.run(run_all()):
        path = claims.save(x)
        typer.echo(f"{x.ticker} FY{x.fiscal_year}: {len(x.claims)} claims from {x.windows} windows "
                   f"({x.cached_windows} cached); dropped {x.dropped_quote} quote, {x.dropped_invalid} invalid, "
                   f"{x.dropped_duplicate} duplicate; {x.truncated_windows} truncated; ${x.cost_usd:.4f} -> {path}")


@app.command(name="extract-eval")
def extract_eval(
    split: str = typer.Option("dev", help="dev or test"),
    extractor: str = typer.Option(None, help="Extractor version (default: the current prompt's)"),
    out: str = typer.Option("evals/results/v3", help="Where to write the report"),
) -> None:
    """v3: precision and recall of extracted claims against the hand labels, in sampled blocks."""
    import json

    from edgar_rag import claims, extraction_eval

    version = extractor or claims.extractor_version()[:12]
    report = extraction_eval.evaluate(split, version, lambda t, fy: claims.load(t, fy, version).claims)
    text = extraction_eval.markdown(report)
    typer.echo(text)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"extraction_{split}_{version}.md").write_text(text)
    (Path(out) / f"extraction_{split}_{version}.json").write_text(json.dumps(report, indent=1, default=str))


@app.command(name="budget")
def show_budget(
    add: float = typer.Option(None, help="Record spend the ledger couldn't see (e.g. a CI run), in USD"),
    note: str = typer.Option("", help="With --add: what it was, e.g. 'CI gate, PR #6'"),
    purpose: str = typer.Option("ci", help="With --add: the purpose label"),
) -> None:
    """Show LLM spend against EDGAR_BUDGET_USD, by purpose, model and run."""
    from edgar_rag import budget

    if add is not None:
        budget.add_manual(add, purpose, note)
        typer.echo(f"Recorded ${add:.4f} ({purpose}: {note})")
    s = budget.summary()
    typer.echo(f"Ledger: {budget.ledger_path()}{'' if budget.enabled() else '  (EDGAR_LEDGER=off)'}")
    typer.echo(f"Spent ${s.spent_usd:.4f} of ${s.cap_usd:.2f} cap, ${s.reserved_usd:.4f} reserved, "
               f"${s.remaining_usd:.4f} left · {s.calls} calls, {s.failed} failed (list prices)")
    for title, rows in (("By purpose", s.by_purpose), ("By model", s.by_model)):
        typer.echo(f"\n{title}:")
        for name, n, cost in rows:
            typer.echo(f"  {name:<16} {n:>6} calls  ${cost:.4f}")
    typer.echo("\nRecent runs:")
    for run, started, n, cost in s.by_run:
        typer.echo(f"  {run:<14} {started}  {n:>6} calls  ${cost:.4f}")


@app.command(name="retrieval-eval")
def run_retrieval_eval(
    golden: str = typer.Option("v2", help="Question set with evidence quotes"),
    split: str | None = typer.Option(None, help="dev, test, regression (comma-separated)"),
    configs: str = typer.Option(
        "vector", help="Comma-separated: vector, keyword, hybrid, each optionally +minilm or +bge"
    ),
    by_category: bool = typer.Option(False, help="Also break each config down by category"),
) -> None:
    """Retrieval only (no LLM): recall@5/10/20, MRR and latency against the gold evidence."""
    from edgar_rag import evaluation

    items = _load_items(golden, split, None)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    from edgar_rag.stats import bootstrap_scores

    rows, per_config = [], {}
    for cfg in [c.strip() for c in configs.split(",") if c.strip()]:
        mode, _, reranker = cfg.partition("+")
        recs = evaluation.retrieval_eval(items, mode=mode, reranker=reranker or "none")
        per_config[cfg] = recs
        if not recs:
            typer.echo("No questions with evidence quotes in this selection.")
            return
        secs = [r.seconds for r in recs]
        rows.append((cfg, len(recs), mean([r.recall[5] for r in recs]), mean([r.recall[10] for r in recs]),
                     mean([r.recall[20] for r in recs]), mean([r.reciprocal_rank for r in recs]),
                     pct(secs, 0.5) * 1000, pct(secs, 0.95) * 1000))
        typer.echo(f"  done: {cfg}")
        if by_category:
            cats: dict[str, list] = {}
            for r in recs:
                cats.setdefault(r.category, []).append(r)
            for cat, rs in sorted(cats.items()):
                typer.echo(f"    {cat:13} n={len(rs):>2} R@5 {mean([r.recall[5] for r in rs]):.2f} "
                           f"R@10 {mean([r.recall[10] for r in rs]):.2f} MRR {mean([r.reciprocal_rank for r in rs]):.2f}")

    lines = [f"Retrieval ablation ({golden}, split: {split or 'all'}, {rows[0][1]} questions with gold evidence)", "",
             "| Config | R@5 | R@10 | R@20 | MRR | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|"]
    for cfg, _, r5, r10, r20, mrr, p50, p95 in rows:
        lines.append(f"| {cfg} | {r5:.3f} | {r10:.3f} | {r20:.3f} | {mrr:.3f} | {p50:.0f} | {p95:.0f} |")
    # Paired bootstrap CIs against the first config (the baseline), per question.
    base_name = rows[0][0]
    base = per_config[base_name]
    lines += ["", f"Change vs {base_name} (paired bootstrap 95% CI over questions):", "",
              "| Config | ΔR@5 | ΔMRR |", "|---|---|---|"]
    for cfg, recs in list(per_config.items())[1:]:
        cells = []
        for key in ("r5", "mrr"):
            def score(rs, key=key):
                return {r.id: (r.recall[5] if key == "r5" else r.reciprocal_rank) for r in rs}
            c = bootstrap_scores(score(base), score(recs))
            mark = " *" if c.significant else ""
            cells.append(f"{c.diff:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}]{mark}")
        lines.append(f"| {cfg} | {cells[0]} | {cells[1]} |")
    lines += ["", "* = the CI excludes 0"]
    table = "\n".join(lines) + "\n"
    out = evaluation.results_dir(golden) / f"retrieval_{(split or 'all').replace(',', '+')}.md"
    out.write_text(table)
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


if __name__ == "__main__":
    app()
