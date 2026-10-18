"""Hard round for the v2 eval set: questions the first round was too easy to ask.

    python evals/hard_round.py draft    # -> candidates_v2_hard.jsonl
    python evals/hard_round.py verify   # -> candidates_v2_hard.checked.jsonl

The first round drafted each question from one chunk, so questions reused the chunk's wording and
search found them easily (v1 scored 96%). This round adds four kinds of question:

- paraphrased: worded like someone who hasn't read the filing (checked for word overlap)
- table_calc: a growth rate, change or share computed from table rows; the drafting model picks
  the operands and this script computes the answer
- cross_company: one topic across three companies, with evidence from every company
- enumeration: a complete list of 4+ items with values from one table

Then run `python evals/generate.py assemble` to rebuild golden_v2.jsonl.
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import generate as base  # evals/generate.py: shared, already-tested helpers
import typer

from edgar_rag import config
from edgar_rag.chunking import Chunk
from edgar_rag.scoring import Evidence, normalize_text

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Hard round of the v2 eval set."""


EVALS = Path(__file__).resolve().parent
HARD = EVALS / "candidates_v2_hard.jsonl"
HARD_CHECKED = EVALS / "candidates_v2_hard.checked.jsonl"
LABELS = "ABC"
MAX_OVERLAP = 0.5  # paraphrased: at most half the question's content words may come from the passage

PARAPHRASE_RULE = (
    " Word the question the way an analyst who has NOT read this filing would ask it: don't reuse "
    "distinctive words or phrases from the passage; describe the concept in your own words (e.g. "
    "'how much would a stronger dollar hurt earnings?' rather than 'adverse 10% foreign exchange "
    "rate change')."
)

CALC_SYSTEM = """\
You write evaluation questions for a financial Q&A system over SEC 10-K filings.
From ONE table row (or two related rows) in the passage, write ONE question whose answer needs a \
calculation the filing does not state directly: a growth rate between two years, a change, one \
line's share of a total, or a ratio of two lines.

Rules:
- Name the company. Never mention "passage", "excerpt", "table" or Item numbers.
- Use the passage's column headers to identify the years; ask only about years the header names.
- Do NOT use total revenue, net income, gross profit, operating income, R&D, EPS, cash flow, total \
assets/liabilities, equity or cash (another tool covers those). Segment, product, geographic, \
expense-line or balance-sheet-line figures are good.
- operands: the two numbers exactly as written in the row, with a label each. Order matters: \
percent_change and difference use [earlier, later]; ratio_percent uses [part, whole].
- evidence: the exact row text you used (copied character for character), passage "A".
- key_points: state the result in words, e.g. "China net sales fell about 3.8% from 2024 to 2025".
- If no row supports a good calculation, return {"skip": true}.

Return JSON only: {"question": "...", "operation": "percent_change" | "difference" | \
"ratio_percent", "operands": [{"as_written": "...", "multiplier": 1 | 1000 | 1000000 | \
1000000000, "label": "..."}, {...}], "evidence": [{"passage": "A", "quote": "..."}], \
"key_points": ["..."]}
"""

CROSS_SYSTEM = """\
You write evaluation questions for a financial Q&A system over SEC 10-K filings.
Write ONE question that compares the companies in the passages on the topic below, answerable \
only by reading every passage. Topic: {topic}

Rules:
- Name every company. Never mention "passage", "excerpt" or Item numbers.
- key_points: one fact per company (with its figure if there is one), plus one comparison \
(e.g. "Lam Research has the highest share").
- evidence: for each company's fact, an exact quote copied character for character (under 200 \
characters) and its passage letter.
- expected_numbers: each company's figure as {{"as_written": exactly as in the quote, \
"multiplier": 1, 1000, 1000000 or 1000000000, "kind": "usd" | "percent" | "count"}}; [] if not numeric.
- If the passages don't let you compare them on this topic, return {{"skip": true}}.

Return JSON only: {{"question": "...", "key_points": ["..."], "evidence": [{{"passage": "A", \
"quote": "..."}}], "expected_numbers": [...]}}
"""

ENUM_SYSTEM = """\
You write evaluation questions for a financial Q&A system over SEC 10-K filings.
Write ONE question asking for a complete list from one table in the passage: every row of one \
group (at least 4 items) with its value for the most recent year in the header, e.g. "List each \
of Apple's reportable segments with its fiscal 2025 net sales."

Rules:
- Name the company and the year. Never mention "passage", "excerpt", "table" or Item numbers.
- Don't list total revenue, net income or other single company-wide totals.
- key_points: one per item, "<item>: <value>".
- evidence: the exact row text for each item (copied character for character), passage "A".
- expected_numbers: every item's value as {"as_written": exactly as in the row, "multiplier": \
1, 1000, 1000000 or 1000000000 (per the table's units), "kind": "usd" | "percent" | "count"}.
- If the passage has no suitable group of 4+ rows with a clear year header, return {"skip": true}.

Return JSON only: {"question": "...", "key_points": ["..."], "evidence": [{"passage": "A", \
"quote": "..."}], "expected_numbers": [...]}
"""

# Topics for cross-company questions, each with a pattern that finds the disclosure in a filing.
TOPICS = {
    "share of revenue from customers in China": r"China[^.|]{0,160}\d+%",
    "number of employees": r"(approximately|about) [\d,]+ (full-time )?(employees|people)",
    "share of revenue from the largest customer(s)": r"(one|single|largest) (direct )?customer[^.]{0,100}\d+%",
    "order backlog": r"backlog[^.]{0,120}\$[\d.,]+ (billion|million)",
    "sensitivity of earnings to foreign exchange rates": r"10%[^.]{0,80}(foreign|currency|exchange)[^.]{0,160}\$[\d.,]+ (million|billion)",
    "sensitivity to interest-rate changes": r"(100 basis point|1%)[^.]{0,120}interest[^.]{0,160}\$[\d.,]+ (million|billion)",
}

YEAR_CELL = re.compile(r"^(fy\s*|fiscal\s+(year\s+)?)?20[12]\d$", re.IGNORECASE)


def has_year_header(chunk: Chunk) -> bool:
    """A table header row whose cells are years ("2025 | 2024 | 2023"), so columns are known.
    Rows that merely mention a year (exhibit lists: "10.4 | ... dated 2025") don't count."""
    for line in chunk.text.split("\n"):
        cells = [c.strip() for c in line.split("|")]
        if sum(bool(YEAR_CELL.match(c)) for c in cells) >= 2:
            return True
    return False


def content_words(text: str, drop: set[str]) -> set[str]:
    words = {w for w in base.WORD_RE.findall(normalize_text(text)) if len(w) > 3 and not re.match(r"^[\d.,%$-]+$", w)}
    return words - base.STOP - drop


def overlap(question: str, passages: list[Chunk]) -> float:
    """Share of the question's content words (minus company names) that also appear in the source."""
    drop = set()
    for c in passages:
        drop |= set(base.WORD_RE.findall(normalize_text(c.company)))
    for name in base.NAMES:
        drop |= set(name.split())
    q = content_words(question, drop)
    if not q:
        return 0.0
    src = set(base.WORD_RE.findall(normalize_text(" ".join(c.text for c in passages))))
    return len(q & src) / len(q)


# --- sampling ------------------------------------------------------------------------------


def sample(counts: dict[str, int], seed: int, exclude: set[str]) -> list[dict]:
    rng = random.Random(seed)
    by_ticker = {t: base.chunks_for(t) for t in base.ALL_TICKERS}
    used = set(exclude)
    jobs = []

    def take(pool: list[Chunk]) -> Chunk | None:
        pool = [c for c in pool if c.chunk_id not in used]
        if not pool:
            return None
        c = rng.choice(pool)
        used.add(c.chunk_id)
        return c

    tickers = base.ALL_TICKERS[:]
    # Paraphrased versions of the first-round categories, from chunks not used before.
    for category, n in (("long_tail", counts["paraphrased"] // 3), ("narrative", counts["paraphrased"] // 3),
                        ("multi_hop", counts["paraphrased"] - 2 * (counts["paraphrased"] // 3))):
        rng.shuffle(tickers)
        for t in (tickers * 3)[: n * 3]:
            if sum(j["category"] == category and "paraphrased" in j["tags"] for j in jobs) >= n:
                break
            cs = by_ticker[t]
            if category == "long_tail":
                first = take([c for c in cs if c.item in base.LONG_TAIL_ITEMS and len(c.text) > 500])
                passages = [first] if first else None
            elif category == "narrative":
                first = take([c for c in cs if c.item in base.NARRATIVE_ITEMS and len(c.text) > 700 and base.table_rows(c) < 2])
                passages = [first] if first else None
            else:
                first = take([c for c in cs if c.item == "7" and len(c.text) > 700 and base.table_rows(c) < 2])
                partner = base.keyword_candidates(first.text, [c for c in cs if c.item == "1A"], [], n=1) if first else []
                passages = [first, partner[0]] if first and partner else None
            if passages:
                jobs.append({"category": category, "tags": ["paraphrased"], "passages": passages})

    # Table calculations and enumerations: table chunks with a year header row.
    for category, n, min_rows in (("table_calc", counts["table_calc"], 3), ("enumeration", counts["enumeration"], 5)):
        rng.shuffle(tickers)
        made = 0
        for t in tickers * 3:
            if made >= n:
                break
            c = take([c for c in by_ticker[t] if c.item in base.TABLE_ITEMS and base.table_rows(c) >= min_rows and has_year_header(c)])
            if c:
                jobs.append({"category": category, "tags": [], "passages": [c]})
                made += 1

    # Cross-company: for each topic, the first matching chunk per company; groups of three.
    per_topic = max(1, counts["cross_company"] // len(TOPICS)) + 1  # topics with more matches make up for thin ones
    cross_made = 0
    for topic, pattern in TOPICS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        matches = {}
        for t in base.ALL_TICKERS:
            hit = next((c for c in by_ticker[t] if rx.search(c.text)), None)
            if hit:
                matches[t] = hit
        names = sorted(matches)
        rng.shuffle(names)
        for g in range(per_topic):
            group = names[g * 3:(g + 1) * 3]
            if len(group) == 3 and cross_made < counts["cross_company"]:
                cross_made += 1
                jobs.append({"category": "cross_company", "tags": [], "topic": topic,
                             "passages": [matches[t] for t in group]})
    return jobs


# --- drafting ------------------------------------------------------------------------------


def draft_one(job: dict) -> dict | None:
    passages = job["passages"]
    blocks = "\n\n".join(
        f"Passage {LABELS[i]} ({c.company}, {c.form} for fiscal year ending {c.report_date}; "
        f"Item {c.item}. {c.section_title}):\n{c.text}"
        for i, c in enumerate(passages)
    )
    cat = job["category"]
    if cat == "table_calc":
        system = CALC_SYSTEM
    elif cat == "enumeration":
        system = ENUM_SYSTEM
    elif cat == "cross_company":
        system = CROSS_SYSTEM.format(topic=job["topic"])
    else:
        system = base.DRAFT_SYSTEM.format(rule=base.CATEGORY_RULES[cat] + PARAPHRASE_RULE)
    data = base.ask_json(system, blocks)
    if not data or data.get("skip") or not data.get("question"):
        return None
    return {
        "category": cat, "tags": job["tags"], "question": data["question"],
        "key_points": data.get("key_points", []), "raw_evidence": data.get("evidence", []),
        "raw_numbers": data.get("expected_numbers", []), "operation": data.get("operation"),
        "operands": data.get("operands", []), "companies": [c.ticker for c in passages],
        "source_chunks": [c.chunk_id for c in passages], "source": "generated", "topic": job.get("topic"),
    }


@app.command()
def draft(paraphrased: int = 24, table_calc: int = 16, cross_company: int = 12, enumeration: int = 12,
          seed: int = 11, workers: int = 6) -> None:
    """Draft hard-round candidates from chunks the first round didn't use."""
    first_round = [json.loads(line) for line in base.CANDIDATES.read_text().splitlines() if line]
    exclude = {i for c in first_round for i in c["source_chunks"]}
    jobs = sample({"paraphrased": paraphrased, "table_calc": table_calc, "cross_company": cross_company,
                   "enumeration": enumeration}, seed, exclude)
    typer.echo(f"Drafting {len(jobs)} hard candidates with {config.judge_model()}...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        drafted = [d for d in pool.map(draft_one, jobs) if d]
    counters: dict[str, int] = {}
    prefix = {"long_tail": "hlt", "narrative": "hnar", "multi_hop": "hmh", "table_calc": "calc",
              "cross_company": "xco", "enumeration": "enum"}
    for d in drafted:
        counters[d["category"]] = counters.get(d["category"], 0) + 1
        d["id"] = f"{prefix[d['category']]}-{counters[d['category']]:03d}"
    HARD.write_text("".join(json.dumps(d) + "\n" for d in drafted))
    typer.echo(f"Saved {len(drafted)} candidates ({len(jobs) - len(drafted)} skipped) -> {HARD.name}: "
               + ", ".join(f"{k} {v}" for k, v in sorted(counters.items())))


# --- verification --------------------------------------------------------------------------


HEADER_YEAR = re.compile(r"20[12]\d")


def column_years_ok(row_quote: str, operands: list[dict], chunk: Chunk) -> str | None:
    """Check each operand sits in the column its label says. Uses the quoted row and the nearest
    header above it with 2+ years or dates (e.g. "2026 | 2025" or "Dec 27, 2025 | Dec 28, 2024").
    Returns a problem, or None. Tables often run newest-first, so a swapped column flips the sign."""
    lines = chunk.text.split("\n")
    target = normalize_text(row_quote)
    ri = next((i for i, x in enumerate(lines) if target and target in normalize_text(x)), None)
    if ri is None:
        return None  # quote spans part of a row; can't place it, leave it to review
    hi = next((i for i in range(ri - 1, -1, -1) if " | " in lines[i] and len(HEADER_YEAR.findall(lines[i])) >= 2), None)
    if hi is None:
        return "no header above the quoted row"
    # Headers may or may not start with a label cell ("2025 | 2024" vs "Years Ended | Dec 27, 2025"),
    # so take every cell that holds a year; data rows always start with a label.
    year_cols = [HEADER_YEAR.search(c).group(0) for c in lines[hi].split("|") if HEADER_YEAR.search(c)]
    cells = [c.strip() for c in lines[ri].split("|")[1:]]
    for op in operands:
        written = str(op.get("as_written", "")).replace("$", "").strip()
        label_year = HEADER_YEAR.search(str(op.get("label", "")))
        if not label_year or written not in cells:
            continue
        col = cells.index(written)
        if col < len(year_cols) and year_cols[col] != label_year.group(0):
            return f"operand {written} is in the {year_cols[col]} column but labelled {label_year.group(0)}"
    return None


def compute(operation: str, a: float, b: float) -> float | None:
    if operation == "percent_change":
        return (b - a) / a * 100 if a else None
    if operation == "difference":
        return b - a
    if operation == "ratio_percent":
        return a / b * 100 if b else None
    return None


def check(c: dict, chunk_index: dict[str, Chunk]) -> tuple[dict, list[str]]:
    problems = []
    passages = [chunk_index[i] for i in c["source_chunks"]]

    evidence, used = [], set()
    for e in c.get("raw_evidence", []):
        if not isinstance(e, dict):
            problems.append("malformed evidence entry")
            continue
        idx = LABELS.find(str(e.get("passage", "A")).strip().upper()[:1])
        quote = str(e.get("quote", "")).strip()
        if not 0 <= idx < len(passages):
            problems.append("evidence cites unknown passage")
            continue
        p = passages[idx]
        if not base.exact_quote(quote, p):
            problems.append(f"quote not exact: {quote[:50]!r}")
            continue
        # Table rows are short and specific; allow them up to the usual distinctiveness limit.
        if not base.distinctive(quote, base.chunks_for(p.ticker)):
            problems.append(f"quote not distinctive: {quote[:50]!r}")
            continue
        evidence.append(Evidence(ticker=p.ticker, quote=quote))
        used.add(idx)
    evidence = base.drop_contained(evidence)
    if not evidence:
        problems.append("no valid evidence")
    if c["category"] == "multi_hop" and used != {0, 1}:
        problems.append("multi_hop evidence doesn't use both passages")
    if c["category"] == "cross_company" and used != set(range(len(passages))):
        problems.append("cross_company evidence doesn't cover every company")

    quotes_text = " ".join(normalize_text(e.quote) for e in evidence)
    numbers = []

    if c["category"] == "table_calc":
        ops = c.get("operands", [])
        values = []
        for op in ops[:2]:
            if not isinstance(op, dict):
                problems.append("malformed operand")
                continue
            written = str(op.get("as_written", "")).strip()
            v = base.parse_written(written)
            bare = normalize_text(written.replace("$", "").replace("%", "").strip("() "))
            if v is None or bare not in quotes_text:
                problems.append(f"operand {written!r} not in the quoted row")
                continue
            values.append(v * float(op.get("multiplier") or 1))
        if len(values) == 2:
            result = compute(c.get("operation") or "", *values)
            if result is None:
                problems.append(f"can't compute {c.get('operation')!r}")
            else:
                kind = "usd" if c.get("operation") == "difference" else "percent"
                numbers.append({"value": result, "kind": kind})
                # A calculation the filing already states isn't a calculation question.
                if kind == "percent" and re.search(rf"\b{round(result):d}%", passages[0].text) and abs(result - round(result)) < 0.05:
                    problems.append("result already stated in the passage")
        else:
            problems.append("needs two operands")
        if not has_year_header(passages[0]):
            problems.append("no year header in the passage")
        for e in evidence:
            issue = column_years_ok(e.quote, ops, passages[0])
            if issue:
                problems.append(issue)
                break
    else:
        for n in c.get("raw_numbers", []):
            if not isinstance(n, dict):
                problems.append("malformed number entry")
                continue
            written = str(n.get("as_written", "")).strip()
            value = base.parse_written(written)
            kind = n.get("kind")
            if value is None or kind not in ("usd", "percent", "count", "usd_per_share"):
                problems.append(f"bad number {n}")
                continue
            bare = normalize_text(written.replace("$", "").replace("%", "").strip("() "))
            if bare not in quotes_text:
                problems.append(f"number {written!r} not in any quote")
                continue
            mult = 1 if kind in ("percent", "usd_per_share") else float(n.get("multiplier") or 1)
            numbers.append({"value": value * mult, "kind": kind, **base.precision_tolerance(written, value * mult, kind)})
        if c["category"] == "enumeration" and len(numbers) < 4:
            problems.append(f"enumeration has {len(numbers)} values (needs 4+)")

    q = c["question"]
    if any(w in q.lower() for w in (*base.BANNED_WORDS, "table")):
        problems.append("question mentions the source text")
    ov = overlap(q, passages)
    if "paraphrased" in c.get("tags", []) and ov > MAX_OVERLAP:
        problems.append(f"not paraphrased (word overlap {ov:.0%})")
    if not 1 <= len(c.get("key_points", [])) <= 6:
        problems.append("needs 1-6 key points")

    cleaned = {
        "id": c["id"], "category": c["category"], "question": q, "expected_numbers": numbers,
        "key_points": c.get("key_points", []), "evidence": [e.model_dump() for e in evidence],
        "companies": sorted(set(c["companies"])), "source": "generated", "verified": False,
        "tags": c.get("tags", []), "source_chunks": c["source_chunks"], "notes": "",
        "word_overlap": round(ov, 2),
    }
    return cleaned, problems


@app.command()
def verify() -> None:
    chunk_index = {c.chunk_id: c for t in base.ALL_TICKERS for c in base.chunks_for(t)}
    cands = [json.loads(line) for line in HARD.read_text().splitlines() if line]
    passed, failed = [], []
    for c in cands:
        q, problems = check(c, chunk_index)
        (failed if problems else passed).append((q, problems))
    HARD_CHECKED.write_text("".join(json.dumps(q) + "\n" for q, _ in passed))
    typer.echo(f"{len(passed)}/{len(cands)} hard candidates passed -> {HARD_CHECKED.name}")
    cats: dict[str, list[int]] = {}
    for q, p in passed + failed:
        cats.setdefault(q["category"] + ("*" if "paraphrased" in q["tags"] else ""), [0, 0])[1 if p else 0] += 1
    for k, (ok, bad) in sorted(cats.items()):
        typer.echo(f"  {k:14} {ok} passed, {bad} failed")
    reasons: dict[str, int] = {}
    for _, ps in failed:
        for p in ps:
            k = re.sub(r"[:(].*", "", p).strip()
            reasons[k] = reasons.get(k, 0) + 1
    typer.echo("Failure reasons: " + ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
    typer.echo("(* = paraphrased)")


if __name__ == "__main__":
    app()
