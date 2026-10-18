"""Build the v2 eval set (evals/golden_v2.jsonl).

    python evals/generate.py migrate     # v1's 58 questions -> regression split, with evidence quotes
    python evals/generate.py draft       # LLM-drafted candidates from sampled chunks -> candidates_v2.jsonl
    python evals/generate.py templates   # code-built near-miss, ambiguous and hard numeric questions
    python evals/generate.py verify      # mechanical checks on every candidate
    python evals/generate.py assemble    # verified questions + dev/test split -> golden_v2.jsonl

Drafting uses JUDGE_MODEL (deepseek-v4-pro), a different model from the one being evaluated.
Nothing an LLM writes is trusted as is: quotes must be exact substrings of the source chunk,
expected numbers must appear in a quote or come from XBRL, and every question is reviewed.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from edgar_rag import config
from edgar_rag.chunking import Chunk
from edgar_rag.llm import complete
from edgar_rag.scoring import Evidence, normalize_text

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Build the v2 eval set."""


EVALS = Path(__file__).resolve().parent
CANDIDATES = EVALS / "candidates_v2.jsonl"
V1 = EVALS / "golden.jsonl"
MIGRATED = EVALS / "v2_regression.jsonl"

# Company names as they appear in questions -> tickers.
NAMES = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "texas instruments": "TXN",
    "lam research": "LRCX", "applied materials": "AMAT", "qualcomm": "QCOM", "intel": "INTC",
    "broadcom": "AVGO", "micron": "MU", "analog devices": "ADI", "kla": "KLAC",
    "marvell": "MRVL", "microchip": "MCHP", "amd": "AMD", "advanced micro devices": "AMD",
}
SEMIS = ["ADI", "AMAT", "AMD", "AVGO", "INTC", "KLAC", "LRCX", "MCHP", "MRVL", "MU", "NVDA", "QCOM", "TXN"]
ALL_TICKERS = ["AAPL", "MSFT", *SEMIS]


# --- helpers -------------------------------------------------------------------------------


def companies_in(question: str) -> list[str]:
    q = question.lower()
    found = [t for name, t in NAMES.items() if re.search(rf"\b{re.escape(name)}\b", q)]
    if "semiconductor" in q and not found:
        found = SEMIS
    return sorted(set(found))


def chunks_for(ticker: str) -> list[Chunk]:
    path = next((config.data_dir() / "processed" / ticker).glob("*.chunks.jsonl"))
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines()]


def ask_json(system: str, user: str) -> dict | None:
    response = complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=6000,
        model=config.judge_model(),
    )
    raw = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def exact_quote(quote: str, chunk: Chunk) -> bool:
    return bool(quote.strip()) and normalize_text(quote) in normalize_text(chunk.text)


MIN_QUOTE_CHARS = 30  # shorter spans ("As of September 27, 2025") are rarely specific
MAX_QUOTE_CHUNKS = 4  # real disclosures repeat across 2-3 sections; generic filler appears in dozens


def distinctive(quote: str, company_chunks: list[Chunk]) -> bool:
    """A gold quote must be specific: long enough, and in at most a couple of chunks. Otherwise
    any retrieved chunk would 'contain the evidence' and recall@k would mean nothing."""
    q = normalize_text(quote)
    hits = sum(q in normalize_text(c.text) for c in company_chunks)
    return len(q) >= MIN_QUOTE_CHARS and 1 <= hits <= MAX_QUOTE_CHUNKS


WORD_RE = re.compile(r"[a-z0-9][a-z0-9.,%$-]*[a-z0-9%]|[a-z0-9]")
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "by", "with", "from", "at", "as",
    "is", "are", "was", "were", "be", "its", "it", "this", "that", "which", "what", "how", "does",
    "did", "do", "has", "have", "had", "their", "company", "companies", "fiscal", "year", "years",
}


def keyword_candidates(text: str, company_chunks: list[Chunk], items: list[str], n: int = 8) -> list[Chunk]:
    """Pick candidate chunks by keyword overlap, weighting rare words higher (a simple IDF).

    Deliberately independent of the vector retriever: gold evidence found with the system
    under test would be biased toward what that system already finds."""
    import math

    terms = {w for w in WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 2}
    docs = [set(WORD_RE.findall(normalize_text(c.text))) for c in company_chunks]
    df = {t: sum(t in d for d in docs) for t in terms}
    idf = {t: math.log((1 + len(docs)) / (1 + df[t])) for t in terms if df[t]}

    def score(i: int) -> float:
        bonus = 1.2 if items and company_chunks[i].item in items else 1.0
        return bonus * sum(w for t, w in idf.items() if t in docs[i])

    ranked = sorted(range(len(company_chunks)), key=score, reverse=True)
    return [company_chunks[i] for i in ranked[:n]]


# --- migrate -------------------------------------------------------------------------------

# Evidence found by hand where the drafting model didn't return a usable quote. Still checked
# for exactness and distinctiveness like any other quote.
MANUAL_EVIDENCE: dict[str, list[tuple[str, str]]] = {
    "nar-18": [(
        "LRCX",
        "represented approximately 34%, 34%, and 42% of our total revenue for fiscal years 2026, 2025, and 2024",
    )],
}

QUOTE_SYSTEM = """\
You find supporting evidence in SEC filing excerpts. For each key point, copy the shortest \
exact span (one clause or sentence, under 200 characters) from ONE passage that supports it. \
Copy it character for character; do not paraphrase, fix typos or join text from two places.

Return JSON only: {"quotes": [{"point": <key point number>, "passage": <passage number>, \
"quote": "<exact text>"}]}. Skip a key point if no passage supports it.
"""


def find_evidence(
    question: str, key_points: list[str], ticker: str, items: list[str], query: str | None = None
) -> list[Evidence]:
    company_chunks = chunks_for(ticker)
    candidates = keyword_candidates(query or " ".join([question, *key_points]), company_chunks, items)
    passages = "\n\n".join(f"[{i + 1}] Item {c.item}\n{c.text}" for i, c in enumerate(candidates))
    points = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(key_points))
    data = ask_json(QUOTE_SYSTEM, f"Question: {question}\n\nKey points:\n{points}\n\nPassages:\n{passages}")
    evidence = []
    for q in (data or {}).get("quotes", []):
        try:
            chunk = candidates[int(q["passage"]) - 1]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        quote = q.get("quote", "")
        if (
            exact_quote(quote, chunk)
            and distinctive(quote, company_chunks)
            and all(quote.strip() != e.quote for e in evidence)
        ):
            evidence.append(Evidence(ticker=ticker, quote=quote.strip()))
    return drop_contained(evidence)


def drop_contained(evidence: list[Evidence]) -> list[Evidence]:
    """Drop a quote that is part of a longer quote: both would mark the same chunk as relevant."""
    norm = [normalize_text(e.quote) for e in evidence]
    return [e for i, e in enumerate(evidence) if not any(i != j and norm[i] in norm[j] for j in range(len(norm)))]


@app.command()
def migrate(workers: int = 6) -> None:
    """Copy v1's questions into the regression split, adding companies and evidence quotes."""
    items = [json.loads(line) for line in V1.read_text().splitlines() if line]
    # Keep evidence that already passed the checks; only look again where there is none.
    previous = {}
    if MIGRATED.exists():
        previous = {q["id"]: q for q in map(json.loads, MIGRATED.read_text().splitlines())}

    def one(q: dict) -> dict:
        q = {**q, "split": "regression", "source": "v1", "companies": companies_in(q["question"])}
        needs_text = q["category"] in ("narrative", "mixed") and q.get("key_points")
        kept = previous.get(q["id"], {}).get("evidence")
        if kept and q["id"] not in MANUAL_EVIDENCE:
            q["evidence"] = [e.model_dump() for e in drop_contained([Evidence(**e) for e in kept])]
        elif q["id"] in MANUAL_EVIDENCE:
            ev = []
            for ticker, quote in MANUAL_EVIDENCE[q["id"]]:
                company_chunks = chunks_for(ticker)
                if any(exact_quote(quote, c) for c in company_chunks) and distinctive(quote, company_chunks):
                    ev.append(Evidence(ticker=ticker, quote=quote))
                else:
                    raise ValueError(f"manual evidence for {q['id']} is not an exact, distinctive quote")
            q["evidence"] = [e.model_dump() for e in ev]
        elif needs_text and len(q["companies"]) == 1:
            ev = []
            # Retry once with only the key points as the query (the drafting model occasionally
            # returns nothing usable, and mixed questions' numbers can crowd out the narrative).
            for query in (None, " ".join(q["key_points"])):
                ev = find_evidence(
                    q["question"], q["key_points"], q["companies"][0], q.get("expected_items", []), query
                )
                if ev:
                    break
            q["evidence"] = [e.model_dump() for e in ev]
        # v1 questions were checked by hand when written (Phases 3-5).
        q["verified"] = True
        q["notes"] = "v1 question, verified against filing text and XBRL when written"
        return q

    with ThreadPoolExecutor(max_workers=workers) as pool:
        migrated = list(pool.map(one, items))
    MIGRATED.write_text("".join(json.dumps(q) + "\n" for q in migrated))

    text_qs = [q for q in migrated if q["category"] in ("narrative", "mixed")]
    with_ev = [q for q in text_qs if q.get("evidence")]
    typer.echo(f"Migrated {len(migrated)} questions -> {MIGRATED.name}")
    typer.echo(f"Text questions with evidence quotes: {len(with_ev)}/{len(text_qs)}")
    for q in text_qs:
        if not q.get("evidence"):
            typer.echo(f"  no evidence found: {q['id']} {q['question'][:70]}")


# --- draft ---------------------------------------------------------------------------------

DRAFT_SYSTEM = """\
You write evaluation questions for a financial Q&A system over SEC 10-K filings.
Write ONE question an equity analyst might ask that is answerable from the passage(s) below and \
nothing else.

Rules:
- Name the company in the question. Never mention "passage", "excerpt", "the text" or Item numbers.
- Be specific: the answer is a fact, number or short list stated in the text, not an opinion.
- {rule}
- key_points: 1-3 short facts a correct answer must contain.
- evidence: for each key point, an exact quote copied character for character (one clause or \
sentence, under 200 characters) and the passage it comes from ("A" or "B").
- expected_numbers: every number a correct answer must state, as {{"as_written": the number \
exactly as it appears in a quote (e.g. "8,529" or "34%"), "multiplier": 1, 1000, 1000000 or \
1000000000 (apply the table's "in millions" etc.; 1 for percentages), "kind": "usd", "percent", \
"count" or "usd_per_share"}}. Use [] if the answer isn't numeric.
- If the passage(s) can't support a good question, return {{"skip": true}}.

Return JSON only: {{"question": "...", "key_points": ["..."], "evidence": [{{"passage": "A", \
"quote": "..."}}], "expected_numbers": [...]}}
"""

CATEGORY_RULES = {
    "long_tail": "Ask about what this section specifically discloses: a legal matter, a market-risk "
    "sensitivity or exposure, or a conclusion about internal controls.",
    "multi_hop": "The question must need BOTH passages: at least one key point from A and at least "
    "one from B. For example, a result described in A and the risk or driver behind it in B.",
    "text_table": "The answer is a specific figure from a table row in the passage. Do NOT ask for "
    "total revenue, cost of revenue, gross profit, R&D expense, operating income, net income, EPS, "
    "operating cash flow, total assets, total liabilities, equity, or cash and equivalents (another "
    "tool covers those). Good targets: segment, product or geographic revenue, share repurchases, "
    "specific expense lines, backlog, sensitivities.",
    "narrative": "Ask about one specific disclosed fact: who, what, how many, or which.",
}
LONG_TAIL_ITEMS = ["3", "7A", "9A"]
TABLE_ITEMS = ["7", "8", "15", "16"]
NARRATIVE_ITEMS = ["1", "1A", "1C", "7"]


def table_rows(chunk: Chunk) -> int:
    return sum(1 for line in chunk.text.split("\n") if " | " in line and re.search(r"\d", line))


def sample_jobs(per_category: dict[str, int], seed: int) -> list[dict]:
    """Choose source chunks for each category, spread across companies, reproducibly."""
    import random

    rng = random.Random(seed)
    by_ticker = {t: chunks_for(t) for t in ALL_TICKERS}
    jobs: list[dict] = []

    def pick(category: str, pool_fn, n: int) -> None:
        tickers = ALL_TICKERS[:]
        rng.shuffle(tickers)
        made = 0
        while made < n:
            progress = False
            for t in tickers:
                if made >= n:
                    break
                pool = [c for c in pool_fn(by_ticker[t]) if all(c.chunk_id not in j["ids"] for j in jobs)]
                if not pool:
                    continue
                passages = pool_fn.pair(t, rng.choice(pool), by_ticker[t]) if hasattr(pool_fn, "pair") \
                    else [rng.choice(pool)]
                if not passages:
                    continue
                jobs.append({"category": category, "ticker": t, "ids": [c.chunk_id for c in passages],
                             "passages": passages})
                made += 1
                progress = True
            if not progress:
                break

    def long_tail(cs):
        return [c for c in cs if c.item in LONG_TAIL_ITEMS and len(c.text) > 500]

    def text_table(cs):
        return [c for c in cs if c.item in TABLE_ITEMS and table_rows(c) >= 4]

    def narrative(cs):
        return [c for c in cs if c.item in NARRATIVE_ITEMS and len(c.text) > 700 and table_rows(c) < 2]

    def mdna(cs):
        return [c for c in cs if c.item == "7" and len(c.text) > 700 and table_rows(c) < 2]

    def pair_with_risk(ticker, first, cs):
        # Partner chunk from Risk Factors chosen by keyword overlap (not the vector retriever).
        risks = [c for c in cs if c.item == "1A"]
        best = keyword_candidates(first.text, risks, [], n=1)
        return [first, best[0]] if best else None

    mdna.pair = pair_with_risk
    for name, fn in [("long_tail", long_tail), ("text_table", text_table), ("narrative", narrative),
                     ("multi_hop", mdna)]:
        pick(name, fn, per_category.get(name, 0))
    return jobs


def draft_one(job: dict) -> dict | None:
    labels = "AB"
    c0 = job["passages"][0]
    header = f"Company: {c0.company} ({job['ticker']}), {c0.form} for fiscal year ending {c0.report_date}"
    passages = "\n\n".join(
        f"Passage {labels[i]} (Item {c.item}. {c.section_title}):\n{c.text}" for i, c in enumerate(job["passages"])
    )
    system = DRAFT_SYSTEM.format(rule=CATEGORY_RULES[job["category"]])
    data = ask_json(system, f"{header}\n\n{passages}")
    if not data or data.get("skip") or not data.get("question"):
        return None
    return {
        "category": job["category"],
        "question": data["question"],
        "key_points": data.get("key_points", []),
        "raw_evidence": data.get("evidence", []),
        "raw_numbers": data.get("expected_numbers", []),
        "companies": [job["ticker"]],
        "source_chunks": job["ids"],
        "source": "generated",
    }


@app.command()
def draft(
    long_tail: int = 28, multi_hop: int = 28, text_table: int = 28, narrative: int = 22,
    seed: int = 7, workers: int = 6,
) -> None:
    """Draft candidate questions from sampled chunks (over-generates; verify filters)."""
    jobs = sample_jobs(
        {"long_tail": long_tail, "multi_hop": multi_hop, "text_table": text_table, "narrative": narrative}, seed
    )
    typer.echo(f"Drafting {len(jobs)} candidates with {config.judge_model()}...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        drafted = list(pool.map(draft_one, jobs))
    kept = [d for d in drafted if d]
    counters: dict[str, int] = {}
    for d in kept:
        counters[d["category"]] = counters.get(d["category"], 0) + 1
        d["id"] = f"{d['category'].replace('_', '')[:4]}-{counters[d['category']]:03d}"
    CANDIDATES.write_text("".join(json.dumps(d) + "\n" for d in kept))
    typer.echo(f"Saved {len(kept)} candidates ({len(jobs) - len(kept)} skipped by the model) -> {CANDIDATES.name}")
    typer.echo("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counters.items())))


# --- verify --------------------------------------------------------------------------------

VERIFIED = EVALS / "candidates_v2.checked.jsonl"
BANNED_WORDS = ("passage", "excerpt", "the text", "item 1", "item 7", "item 3", "item 9")


def parse_written(as_written: str) -> float | None:
    s = str(as_written).replace("$", "").replace(",", "").replace("%", "").strip()
    s = s.strip("()")  # (16) in tables is a negative number; magnitude is what matters
    try:
        return float(s)
    except ValueError:
        return None


def precision_tolerance(written: str, value: float, kind: str) -> dict:
    """A figure written rounded ("$2.8 billion") is only known to half its last digit, so an answer
    giving the exact "$2.782 billion" is right. Widen the tolerance to that precision when it is
    looser than the default; never tighten it."""
    from edgar_rag.scoring import PERCENT_ABS_TOL, USD_REL_TOL

    digits = written.replace("$", "").replace(",", "").replace("%", "").strip("() ")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    half_unit = 0.5 * 10 ** -decimals
    if kind == "percent":
        return {"tolerance": round(half_unit, 6)} if half_unit > PERCENT_ABS_TOL else {}
    if value and kind in ("usd", "count"):
        rel = half_unit / abs(value)
        return {"tolerance": round(rel, 6)} if rel > USD_REL_TOL else {}
    return {}


def check_candidate(c: dict, chunk_index: dict[str, Chunk]) -> tuple[dict, list[str]]:
    """Mechanical checks. Returns the cleaned question and a list of problems (empty = passes)."""
    from edgar_rag.xbrl import PUBLIC_METRICS, CompanyFacts, FactNotFound

    problems = []
    ticker = c["companies"][0]
    passages = [chunk_index[i] for i in c["source_chunks"]]
    company_chunks = chunks_for(ticker)

    # Evidence: exact, distinctive quotes from the stated passage.
    evidence, from_passage = [], set()
    for e in c.get("raw_evidence", []):
        label = str(e.get("passage", "A")).strip().upper()[:1]
        idx = "AB".find(label)
        quote = str(e.get("quote", "")).strip()
        if idx < 0 or idx >= len(passages):
            problems.append(f"evidence cites unknown passage {label!r}")
            continue
        if not exact_quote(quote, passages[idx]):
            problems.append(f"quote not exact: {quote[:60]!r}")
            continue
        if not distinctive(quote, company_chunks):
            problems.append(f"quote not distinctive: {quote[:60]!r}")
            continue
        evidence.append(Evidence(ticker=ticker, quote=quote))
        from_passage.add(idx)
    evidence = drop_contained(evidence)
    if not evidence:
        problems.append("no valid evidence")
    if c["category"] == "multi_hop" and from_passage != {0, 1}:
        problems.append("multi_hop evidence doesn't use both passages")

    # Numbers: written form must appear in a quote; convert with the stated multiplier.
    quotes_text = " ".join(normalize_text(e.quote) for e in evidence)
    numbers = []
    for n in c.get("raw_numbers", []):
        written = str(n.get("as_written", "")).strip()
        value = parse_written(written)
        kind = n.get("kind")
        if value is None or kind not in ("usd", "percent", "count", "usd_per_share"):
            problems.append(f"bad number {n}")
            continue
        bare = normalize_text(written.replace("$", "").replace("%", "").strip("() "))
        if bare not in quotes_text:
            problems.append(f"number {written!r} not in any quote")
            continue
        multiplier = 1 if kind in ("percent", "usd_per_share") else float(n.get("multiplier") or 1)
        numbers.append({"value": value * multiplier, "kind": kind, **precision_tolerance(written, value, kind)})

    # Text-table questions must not just restate an XBRL metric (another tool covers those).
    if c["category"] == "text_table":
        if not numbers:
            problems.append("text_table question has no expected number")
        try:
            facts = CompanyFacts(ticker)
            year = int(passages[0].report_date[:4])
            xbrl_values = []
            for m in PUBLIC_METRICS:
                try:
                    xbrl_values.append(abs(facts.get(m, year).value))
                except FactNotFound:
                    pass
            for n in numbers:
                if n["kind"] == "usd" and any(v and abs(abs(n["value"]) - v) / v < 0.005 for v in xbrl_values):
                    problems.append(f"text_table number {n['value']:,.0f} equals an XBRL metric")
        except (ValueError, KeyError):
            pass

    q = c["question"]
    if any(w in q.lower() for w in BANNED_WORDS):
        problems.append("question mentions the source text or an Item number")
    company = passages[0].company.lower()
    name_words = [w for w in re.findall(r"[a-z]+", company) if w not in ("inc", "corp", "corporation", "co", "de", "the")]
    names = [n for n, t in NAMES.items() if t == ticker]
    if not (ticker.lower() in q.lower() or any(n in q.lower() for n in names) or any(w in q.lower() for w in name_words[:1])):
        problems.append("question doesn't name the company")
    if not 1 <= len(c.get("key_points", [])) <= 3:
        problems.append("needs 1-3 key points")

    cleaned = {
        "id": c["id"], "category": c["category"], "question": q,
        "expected_numbers": numbers, "key_points": c.get("key_points", []),
        "evidence": [e.model_dump() for e in evidence], "companies": [ticker],
        "source": "generated", "verified": False, "source_chunks": c["source_chunks"],
        "notes": "",
    }
    return cleaned, problems


@app.command()
def verify() -> None:
    """Run mechanical checks on every drafted candidate; write the ones that pass."""
    chunk_index = {c.chunk_id: c for t in ALL_TICKERS for c in chunks_for(t)}
    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    passed, failed = [], []
    for c in candidates:
        cleaned, problems = check_candidate(c, chunk_index)
        (failed if problems else passed).append((cleaned, problems))
    VERIFIED.write_text("".join(json.dumps(q) + "\n" for q, _ in passed))
    typer.echo(f"{len(passed)}/{len(candidates)} candidates passed mechanical checks -> {VERIFIED.name}")
    by_cat: dict[str, list[int]] = {}
    for q, p in passed + failed:
        by_cat.setdefault(q["category"], [0, 0])[0 if not p else 1] += 1
    for cat, (ok, bad) in sorted(by_cat.items()):
        typer.echo(f"  {cat:11} {ok} passed, {bad} failed")
    reasons: dict[str, int] = {}
    for _, problems in failed:
        for p in problems:
            key = re.sub(r":.*", "", p).replace("'", "")
            reasons[key] = reasons.get(key, 0) + 1
    typer.echo("Failure reasons: " + ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))


# --- templates -----------------------------------------------------------------------------

TEMPLATES = EVALS / "templates_v2.jsonl"


@app.command()
def templates() -> None:
    """Code-built questions whose answers can be proved: near-miss unanswerable, ambiguous names,
    and multi-step XBRL numerics. Expected values are computed from the data, not written by hand."""
    from edgar_rag.companies import indexed_companies
    from edgar_rag.xbrl import CompanyFacts, FactNotFound

    fy = {c.ticker: c.fiscal_year for c in indexed_companies()}
    cf = {t: CompanyFacts(t) for t in ALL_TICKERS}

    def v(t, m, y=None):
        return cf[t].get(m, y or fy[t]).value

    out: list[dict] = []

    def add(id_, category, question, numbers=(), key_points=(), notes="", companies=()):
        out.append({
            "id": id_, "category": category, "question": question,
            "expected_numbers": [
                {"value": n[0], "kind": n[1], **({"tolerance": n[2]} if len(n) > 2 else {})} for n in numbers
            ],
            "key_points": list(key_points), "should_refuse": category == "near_miss",
            "companies": list(companies), "source": "template", "verified": True, "notes": notes,
        })

    # Near-miss: fiscal years that end in the future or whose 10-K isn't filed yet.
    for i, (t, name, year) in enumerate([
        ("AAPL", "Apple", 2026), ("NVDA", "NVIDIA", 2027), ("MSFT", "Microsoft", 2027),
        ("INTC", "Intel", 2026), ("TXN", "Texas Instruments", 2026), ("AMD", "AMD", 2026),
    ], 1):
        try:
            cf[t].get("revenue", year)
            continue  # exists after all: not unanswerable
        except FactNotFound:
            pass
        possessive = f"{name}'" if name.endswith("s") else f"{name}'s"
        add(f"near-{i:03d}", "near_miss", f"What was {possessive} total revenue in fiscal {year}?",
            notes=f"No FY{year} revenue in {t}'s XBRL data: that fiscal year's 10-K isn't filed (checked).",
            companies=[t])

    # Near-miss: foreign 20-F filers (IFRS, no us-gaap facts).
    for i, (t, name) in enumerate([("TSM", "TSMC"), ("ASML", "ASML")], 7):
        try:
            gaap = CompanyFacts(t)._gaap
        except (ValueError, KeyError):
            gaap = {}
        if gaap and any(gaap.get(tag) for tag in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")):
            continue
        add(f"near-{i:03d}", "near_miss", f"What was {name}'s net income in its latest fiscal year?",
            notes=f"{t} files a 20-F under IFRS; it has no us-gaap revenue facts and no indexed 10-K (checked).",
            companies=[t])

    # Near-miss: filing text of companies that aren't indexed (numbers would still be fetchable
    # from XBRL, so these ask about text only).
    indexed = {c.ticker for c in indexed_companies()}
    for i, (t, q) in enumerate([
        ("AMZN", "What does Amazon's latest 10-K say about competition for AWS?"),
        ("GOOGL", "How does Alphabet's 10-K describe the risks of its AI investments?"),
        ("META", "What does Meta's latest 10-K say about Reality Labs' outlook?"),
        ("TSLA", "What legal proceedings does Tesla's latest 10-K disclose?"),
        ("ORCL", "How does Oracle's 10-K describe its cloud infrastructure strategy?"),
        ("IBM", "What does IBM's latest 10-K say about its consulting business risks?"),
        ("CSCO", "How does Cisco's latest 10-K describe its supply-chain risks?"),
    ], 9):
        if t not in indexed:
            add(f"near-{i:03d}", "near_miss", q, notes=f"{t} is not indexed, so its filing text can't be searched (checked).",
                companies=[t])

    # Ambiguous or informal names.
    adi_rev, lrcx_gm = v("ADI", "revenue"), v("LRCX", "gross_profit") / v("LRCX", "revenue") * 100
    add("amb-001", "ambiguous", "What was Analog's revenue in its latest fiscal year?",
        [(adi_rev, "usd")], ["Identifies the company as Analog Devices (ADI)"], companies=["ADI"])
    add("amb-002", "ambiguous", "What was Lam's gross margin in its latest fiscal year?",
        [(lrcx_gm, "percent")], ["Identifies the company as Lam Research (LRCX)"], companies=["LRCX"])
    add("amb-003", "ambiguous", "How much net income did the iPhone maker report for fiscal 2025?",
        [(v("AAPL", "net_income", 2025), "usd")], ["Identifies the company as Apple"], companies=["AAPL"])
    add("amb-004", "ambiguous", "What was Micro's revenue last fiscal year?",
        key_points=["Says 'Micro' is ambiguous: it could mean Microsoft, Micron, Microchip or Advanced Micro Devices, and asks which one or gives the candidates"],
        notes="Deliberately ambiguous: the right answer asks for clarification or covers the candidates.",
        companies=["MSFT", "MU", "MCHP", "AMD"])
    add("amb-005", "ambiguous", "What was Applied's operating income in fiscal 2025?",
        [(v("AMAT", "operating_income", 2025), "usd")], ["Identifies the company as Applied Materials (AMAT)"], companies=["AMAT"])
    add("amb-006", "ambiguous", "Compare Micron's and Microchip's gross margins in their latest fiscal years.",
        [(v("MU", "gross_profit") / v("MU", "revenue") * 100, "percent"),
         (v("MCHP", "gross_profit") / v("MCHP", "revenue") * 100, "percent")],
        ["Micron's gross margin is higher than Microchip's"], companies=["MU", "MCHP"])
    add("amb-007", "ambiguous", "What was MCHP's revenue in its latest fiscal year?",
        [(v("MCHP", "revenue"), "usd")], ["Identifies MCHP as Microchip Technology"], companies=["MCHP"])
    add("amb-008", "ambiguous", "What was Advanced Micro's diluted EPS for fiscal 2025?",
        [(v("AMD", "eps_diluted", 2025), "usd_per_share")], ["Identifies the company as AMD"], companies=["AMD"])
    add("amb-009", "ambiguous", "Who had more revenue in their latest fiscal year, AMD or Applied?",
        [(v("AMD", "revenue"), "usd"), (v("AMAT", "revenue"), "usd")],
        ["AMD had more revenue than Applied Materials"], companies=["AMD", "AMAT"])
    add("amb-010", "ambiguous", "How much did Texas spend on R&D in 2025?",
        [(v("TXN", "research_and_development", 2025), "usd")], ["Identifies the company as Texas Instruments"], companies=["TXN"])

    # Harder multi-step numerics, all computed from XBRL.
    add("hnum-001", "numeric", "What was AMD's operating cash flow as a percentage of revenue in fiscal 2025?",
        [(v("AMD", "operating_cash_flow", 2025) / v("AMD", "revenue", 2025) * 100, "percent")], companies=["AMD"])
    add("hnum-002", "numeric", "By what percentage did Micron's revenue grow from fiscal 2025 to fiscal 2026?",
        [((v("MU", "revenue", 2026) - v("MU", "revenue", 2025)) / v("MU", "revenue", 2025) * 100, "percent")], companies=["MU"])
    add("hnum-003", "numeric", "What was Intel's net income or loss in fiscal 2025?",
        [(v("INTC", "net_income", 2025), "usd")], ["Intel reported a net loss"], companies=["INTC"])
    add("hnum-004", "numeric", "What were Broadcom's total liabilities at the end of fiscal 2025?",
        [(v("AVGO", "total_liabilities", 2025), "usd")],
        ["Says the figure is derived (Broadcom reports no total liabilities line)"], companies=["AVGO"])
    add("hnum-005", "numeric", "By what percentage did NVIDIA's net income grow from fiscal 2025 to fiscal 2026?",
        [((v("NVDA", "net_income", 2026) - v("NVDA", "net_income", 2025)) / v("NVDA", "net_income", 2025) * 100, "percent")], companies=["NVDA"])
    add("hnum-006", "numeric", "What was KLA's gross margin in fiscal 2026?",
        [(v("KLAC", "gross_profit", 2026) / v("KLAC", "revenue", 2026) * 100, "percent")],
        ["Says gross profit is derived (KLA reports no gross profit line)"], companies=["KLAC"])
    add("hnum-007", "numeric", "How much higher was Microsoft's operating cash flow than its net income in fiscal 2026?",
        [(v("MSFT", "operating_cash_flow", 2026) - v("MSFT", "net_income", 2026), "usd")], companies=["MSFT"])
    add("hnum-008", "numeric", "What was Lam Research's diluted EPS in fiscal 2026?",
        [(v("LRCX", "eps_diluted", 2026), "usd_per_share")], companies=["LRCX"])
    add("hnum-009", "numeric", "What was Qualcomm's ratio of total liabilities to stockholders' equity at the end of fiscal 2025?",
        [(v("QCOM", "total_liabilities", 2025) / v("QCOM", "stockholders_equity", 2025), "count", 0.02)], companies=["QCOM"])
    add("hnum-010", "numeric", "What was Apple's operating margin change from fiscal 2024 to fiscal 2025, in percentage points?",
        [((v("AAPL", "operating_income", 2025) / v("AAPL", "revenue", 2025)
           - v("AAPL", "operating_income", 2024) / v("AAPL", "revenue", 2024)) * 100, "percent")], companies=["AAPL"])

    TEMPLATES.write_text("".join(json.dumps(q) + "\n" for q in out))
    import collections
    typer.echo(f"Wrote {len(out)} template questions -> {TEMPLATES.name}: "
               + ", ".join(f"{k} {n}" for k, n in collections.Counter(q["category"] for q in out).items()))


# --- assemble ------------------------------------------------------------------------------

REVIEW = EVALS / "review_v2.json"
GOLDEN_V2 = EVALS / "golden_v2.jsonl"
SPOTCHECK = EVALS / "spotcheck_v2.md"
DEV_SHARE = 0.35  # the rest is test; tune only on dev


def stable_rank(text: str, seed: str) -> str:
    import hashlib

    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


@app.command()
def assemble(seed: str = "v2", spot_checks: int = 20) -> None:
    """Combine regression + reviewed generated + template questions, split dev/test by category,
    and pick a stratified random sample for the human spot-check."""
    review = json.loads(REVIEW.read_text())
    regression = [json.loads(line) for line in MIGRATED.read_text().splitlines() if line]
    generated = []
    hard_checked = EVALS / "candidates_v2_hard.checked.jsonl"
    lines = VERIFIED.read_text().splitlines() + (hard_checked.read_text().splitlines() if hard_checked.exists() else [])
    for line in lines:
        q = json.loads(line)
        q.pop("word_overlap", None)
        if q["id"] in review["reject"]:
            continue
        fix = {k: v for k, v in review["fix"].get(q["id"], {}).items() if k != "reason"}
        q.update(fix)
        q.pop("source_chunks", None)
        q["verified"] = True
        q["notes"] = "Generated by " + config.judge_model() + "; quotes and numbers checked mechanically; " \
            "read and accepted by Claude" + (f" (fixed: {review['fix'][q['id']]['reason']})" if q["id"] in review["fix"] else "")
        generated.append(q)
    template = [json.loads(line) for line in TEMPLATES.read_text().splitlines() if line]
    new = generated + template

    # Each question's split depends only on its own id, so adding questions later never moves an
    # existing question between dev and test (no accidental tuning on test).
    by_cat: dict[str, list[dict]] = {}
    for q in new:
        by_cat.setdefault(q["category"], []).append(q)
        q["split"] = "dev" if int(stable_rank(q["id"], seed), 16) / 16 ** 64 < DEV_SHARE else "test"

    everything = regression + new
    ids = [q["id"] for q in everything]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    GOLDEN_V2.write_text("".join(json.dumps(q) + "\n" for q in everything))

    # Spot-check: once the reviewer has checked a sample, keep exactly that sample and record the
    # verdict; otherwise draw a fresh stratified sample from the questions Claude verified.
    by_id = {q["id"]: q for q in generated + template}
    done = review.get("spot_check")
    if done:
        picked = [by_id[i] for i in done["ids"] if i in by_id]
        lines = ["# v2 spot-check", "",
                 f"**Reviewed {done['date']} by {done['reviewer']}:** {done['verdict']} {done['caveat']}",
                 f"*{done['recorded_by']}.*", "",
                 (f"{len(picked)} questions sampled at random across categories from the round-1 "
                  "Claude-verified questions."), ""]
    else:
        pool = sorted(generated + [q for q in template if q["category"] != "near_miss"],
                      key=lambda q: stable_rank(q["id"], seed + "-spot"))
        picked, seen = [], {}
        for q in pool:
            if seen.get(q["category"], 0) < max(2, spot_checks // 5) and len(picked) < spot_checks:
                picked.append(q)
                seen[q["category"]] = seen.get(q["category"], 0) + 1
        lines = ["# v2 spot-check", "",
                 (f"{len(picked)} questions sampled at random across categories from the {len(generated) + len(template)} "
                  "Claude-verified questions. For each, check that the question is clear and fair, and that the expected "
                  "answer is right. Mark each **OK** or describe the problem."), ""]
    for i, q in enumerate(picked, 1):
        lines += [f"## {i}. `{q['id']}` ({q['category']}, {', '.join(q['companies'])})", "", f"**Q:** {q['question']}", ""]
        if q.get("key_points"):
            lines += ["**Must say:**"] + [f"- {p}" for p in q["key_points"]] + [""]
        if q.get("expected_numbers"):
            nums = ", ".join(
                f"{n['value']:.2f}%" if n["kind"] == "percent" else
                (f"${n['value'] / 1e6:,.1f} million" if n["kind"] == "usd" else f"{n['value']:,.4g} ({n['kind']})")
                for n in q["expected_numbers"]
            )
            lines += [f"**Expected numbers:** {nums}", ""]
        if q.get("evidence"):
            lines += ["**Evidence in the filing:**"] + [f"> {e['quote']}" for e in q["evidence"]] + [""]
        if q.get("notes"):
            lines += [f"*{q['notes']}*", ""]
        lines += [f"**Verdict:** {'OK' if done else ''}", "", "---", ""]
    SPOTCHECK.write_text("\n".join(lines))

    import collections
    splits = collections.Counter(q["split"] for q in everything)
    typer.echo(f"Wrote {len(everything)} questions -> {GOLDEN_V2.name}: " + ", ".join(f"{k} {v}" for k, v in sorted(splits.items())))
    typer.echo(f"  new questions: {len(new)} ({len(generated)} generated, {len(template)} template); rejected {len(review['reject'])}")
    for cat, qs in sorted(by_cat.items()):
        typer.echo(f"  {cat:11} {sum(q['split'] == 'dev' for q in qs)} dev / {sum(q['split'] == 'test' for q in qs)} test")
    typer.echo(f"Spot-check sample: {len(picked)} questions -> {SPOTCHECK.name}")


if __name__ == "__main__":
    app()
