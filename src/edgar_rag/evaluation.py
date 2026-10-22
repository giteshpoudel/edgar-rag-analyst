"""Run the golden question set through a mode ("agent" or "simple"), score it, and summarize."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from edgar_rag import tracing
from edgar_rag.scoring import (
    Evidence,
    Expected,
    judge,
    number_found,
    quotes_in_chunk,
    recall_at_k,
    reciprocal_rank,
    retrieval_scores,
)

Mode = Literal["agent", "simple"]
EVALS_DIR = Path("evals")
PASS_COVERAGE = 2 / 3  # narrative answers must cover at least this share of key points
CATEGORIES = (
    "numeric", "narrative", "mixed", "comparison", "trend", "sector",
    # v2: harder categories v1 never faced
    "multi_hop", "long_tail", "text_table", "ambiguous", "near_miss",
    # v2 hard round
    "table_calc", "cross_company", "enumeration",
    "unanswerable",
)
# Categories where the right answer is to decline (near_miss = plausible-looking but unanswerable).
REFUSAL_CATEGORIES = ("unanswerable", "near_miss")


class GoldenItem(BaseModel):
    id: str
    category: Literal[
        "numeric", "narrative", "mixed", "comparison", "trend", "sector",
        "multi_hop", "long_tail", "text_table", "ambiguous", "near_miss",
        "table_calc", "cross_company", "enumeration", "unanswerable",
    ]
    question: str
    expected_numbers: list[Expected] = []
    expected_items: list[str] = []
    key_points: list[str] = []
    should_refuse: bool = False
    # v2 fields (all optional, so v1's golden.jsonl still loads)
    split: Literal["regression", "dev", "test"] | None = None  # regression = v1's 58 questions
    evidence: list[Evidence] = []  # gold quotes for recall@k (D-52)
    companies: list[str] = []  # tickers the question is about
    source: Literal["v1", "generated", "template"] | None = None
    verified: bool = False  # set only after checking against the filing / XBRL
    notes: str = ""  # how it was verified, or why it's unanswerable
    tags: list[str] = []  # e.g. "paraphrased"


class Record(BaseModel):
    id: str
    category: str
    mode: str
    question: str
    answer: str
    tools_used: list[str]
    retrieved_items: list[str]  # 10-K sections of retrieved chunks, in rank order
    top_score: float | None = None  # best search similarity seen (None if no search ran)
    invalid_citations: list[int]
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    numbers_found: list[bool]
    retrieval_hit: bool | None
    reciprocal_rank: float | None
    key_points_covered: list[bool]
    refused: bool
    passed: bool
    truncated: bool = False  # the system's answer was cut off at max_tokens
    malformed: bool = False  # the model wrote tool-call markup instead of an answer
    judge_error: str = ""  # the judge gave no usable verdict (excluded from judge metrics)
    # Share of gold evidence quotes present in the passages the system actually saw
    # (None when the question has no evidence). Separates "never found it" from "found it,
    # answered wrong".
    evidence_seen: float | None = None
    # Number grounding (W3): numbers the answer states, and those no tool result supports.
    numbers_stated: int = 0
    unsupported_numbers: list[str] = []
    unsupported_before_repair: list[str] = []  # flagged by the runtime guard before its repair pass
    # Tracing and cost (W4), totalled from this question's spans. None in runs from before W4.
    trace_id: str = ""  # open the full trace in Phoenix by this id
    cost_usd: float | None = None  # the system's LLM calls, at list (peak) prices
    judge_cost_usd: float | None = None  # grading cost, reported separately
    llm_calls: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    llm_seconds: float | None = None  # time inside LLM calls (the rest is tools and overhead)
    retrieval_seconds: float | None = None
    error: str = ""


def load_golden(path: Path = EVALS_DIR / "golden.jsonl") -> list[GoldenItem]:
    return [GoldenItem.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _item_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split(":")[1]  # "<accession>:<item>:<index>"


@lru_cache(maxsize=1)
def _chunk_index() -> dict:
    from edgar_rag import config
    from edgar_rag.chunking import Chunk

    index = {}
    for path in (config.data_dir() / "processed").glob("*/*.chunks.jsonl"):
        for line in path.read_text().splitlines():
            c = Chunk.model_validate_json(line)
            index[c.chunk_id] = c
    return index


def _evidence_seen(chunk_ids: list[str], evidence: list[Evidence]) -> float | None:
    if not evidence:
        return None
    index = _chunk_index()
    found = [quotes_in_chunk(index[i].text, index[i].ticker, evidence) for i in chunk_ids if i in index]
    return recall_at_k(found, len(evidence), len(found))


# --- Retrieval-only eval (no LLM): recall@k and MRR straight from the retriever ------------


class RetrievalRecord(BaseModel):
    id: str
    category: str
    recall: dict[int, float]  # k -> recall@k
    reciprocal_rank: float
    seconds: float


def retrieval_eval(
    items: list[GoldenItem], ks: tuple[int, ...] = (5, 10, 20), mode: str | None = None,
    reranker: str | None = None,
) -> list[RetrievalRecord]:
    """Search the top max(ks) chunks for each question with gold evidence, filtered to the
    question's company when it names one (as the agent does), and score against the quotes.
    One warm-up search first, so model loading isn't counted as search latency."""
    from edgar_rag.retriever import search

    with_evidence = [i for i in items if i.evidence]
    if with_evidence:
        search(with_evidence[0].question, k=max(ks), mode=mode, reranker=reranker)
    records = []
    for item in with_evidence:
        tickers = sorted({e.ticker for e in item.evidence})
        start = time.monotonic()
        results = search(item.question, k=max(ks), ticker=tickers[0] if len(tickers) == 1 else None,
                         mode=mode, reranker=reranker)
        found = [quotes_in_chunk(r.chunk.text, r.chunk.ticker, item.evidence) for r in results]
        records.append(RetrievalRecord(
            id=item.id, category=item.category,
            recall={k: recall_at_k(found, len(item.evidence), k) for k in ks},
            reciprocal_rank=reciprocal_rank(found),
            seconds=round(time.monotonic() - start, 3),
        ))
    return records


def _run_mode(question: str, mode: Mode) -> dict:
    """Answer one question; return the parts scoring needs, independent of mode."""
    if mode == "simple":
        from edgar_rag.answer import answer

        r = answer(question)
        return {
            "answer": r.text,
            "tools_used": ["search"],
            "chunk_ids": [s.chunk.chunk_id for s in r.sources],
            "sources_text": "\n".join(s.chunk.text for s in r.sources),
            "retrieved_items": [s.chunk.item for s in r.sources],
            "top_score": max((s.score for s in r.sources), default=None),
            "invalid_citations": r.invalid_citations,
            "prompt_tokens": r.llm.prompt_tokens,
            "completion_tokens": r.llm.completion_tokens,
            "truncated": r.llm.truncated,
        }
    from edgar_rag.agent import ask

    r = ask(question)
    return {
        "answer": r.text,
        "tools_used": [c.name for c in r.tool_calls],
        "chunk_ids": [s.ref for s in r.sources if s.kind == "filing"],
        "sources_text": "\n".join(c.result for c in r.tool_calls),
        "retrieved_items": [_item_from_chunk_id(s.ref) for s in r.sources if s.kind == "filing"],
        "top_score": max((s.score for s in r.sources if s.score is not None), default=None),
        "invalid_citations": r.invalid_citations,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "truncated": r.truncated,
        "malformed": r.malformed,
        "unsupported_before_repair": r.unsupported_before_repair,
    }


def evaluate_one(item: GoldenItem, mode: Mode) -> Record:
    """Answer and score one question inside a root span, so the whole run (answer, tools,
    retrieval, judge) is one trace; then attach that trace's cost and timing to the record."""
    with tracing.span("eval.question", "CHAIN", **{
        "input.value": item.question, "edgar.eval.id": item.id,
        "edgar.eval.category": item.category, "edgar.eval.mode": mode,
    }) as span:
        trace_id = tracing.current_trace_id()
        record = _evaluate(item, mode)
        tracing.set_attributes(span, {"output.value": record.answer, "edgar.eval.passed": record.passed})
        if record.error:
            tracing.mark_error(span, record.error)
    usage = tracing.COLLECTOR.pop(trace_id)  # complete now: the root span has ended
    return record.model_copy(update={
        "trace_id": tracing.trace_id_hex(trace_id),
        "cost_usd": round(usage.cost_usd, 6) if usage.cost_known else None,
        "judge_cost_usd": round(usage.judge_cost_usd, 6),
        "llm_calls": usage.llm_calls,
        "cached_prompt_tokens": usage.cached_prompt_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "llm_seconds": round(usage.seconds.get("LLM", 0.0), 2),
        "retrieval_seconds": round(usage.seconds.get("RETRIEVER", 0.0), 2),
    })


def _evaluate(item: GoldenItem, mode: Mode) -> Record:
    start = time.monotonic()
    try:
        out = _run_mode(item.question, mode)
        error = ""
    except Exception as e:  # noqa: BLE001 - one failure shouldn't stop the run; it's scored as a fail
        out = {"answer": "", "tools_used": [], "retrieved_items": [], "invalid_citations": [],
               "prompt_tokens": 0, "completion_tokens": 0, "truncated": False, "top_score": None,
               "chunk_ids": [], "sources_text": ""}
        error = f"{type(e).__name__}: {e}"
    seconds = time.monotonic() - start

    seen = _evidence_seen(out.pop("chunk_ids"), item.evidence)
    sources_text = out.pop("sources_text")
    answer = out["answer"]
    from edgar_rag.grounding import check as ground

    grounding = ground(answer, sources_text, item.question)
    numbers = [number_found(e, answer) for e in item.expected_numbers]
    hit, rr = (None, None)
    if item.expected_items:
        hit, rr = retrieval_scores(out["retrieved_items"], item.expected_items)
    verdict = judge(item.question, answer, item.key_points) if answer else None
    covered = verdict.key_points if verdict else [False] * len(item.key_points)
    refused = verdict.refused if verdict else True

    if item.category in REFUSAL_CATEGORIES:
        passed = refused
    else:
        coverage_ok = not item.key_points or sum(covered) / len(covered) >= PASS_COVERAGE
        passed = all(numbers) and coverage_ok and not refused and not error

    return Record(
        id=item.id, category=item.category, mode=mode, question=item.question, **out,
        seconds=round(seconds, 1), numbers_found=numbers, retrieval_hit=hit,
        reciprocal_rank=rr, key_points_covered=covered, refused=refused, passed=passed,
        judge_error=verdict.error if verdict else "", evidence_seen=seen,
        numbers_stated=len(grounding.claims), unsupported_numbers=[c.raw for c in grounding.unsupported],
        error=error,
    )


def run(mode: Mode, items: list[GoldenItem], workers: int = 4) -> list[Record]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda it: evaluate_one(it, mode), items))


# --- Summary -------------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[Record]) -> dict:
    answerable = [r for r in records if r.category not in REFUSAL_CATEGORIES]
    unanswerable = [r for r in records if r.category in REFUSAL_CATEGORIES]
    with_retrieval = [r for r in records if r.retrieval_hit is not None]
    with_points = [r for r in records if r.key_points_covered and not r.judge_error]
    numbers = [n for r in records for n in r.numbers_found]
    out = {
        "questions": len(records),
        "pass_rate": _mean([r.passed for r in records]),
        **{
            f"pass_{cat}": _mean([r.passed for r in records if r.category == cat])
            for cat in CATEGORIES
        },
        "number_accuracy": _mean(numbers),
        "retrieval_hit_rate": _mean([r.retrieval_hit for r in with_retrieval]),
        "mrr": _mean([r.reciprocal_rank for r in with_retrieval]),
        "key_point_coverage": _mean([sum(r.key_points_covered) / len(r.key_points_covered) for r in with_points]),
        "refusal_when_unanswerable": _mean([r.refused for r in unanswerable]),
        "false_refusal_rate": _mean([r.refused for r in answerable]),
        "invalid_citation_rate": _mean([bool(r.invalid_citations) for r in records]),
        "avg_prompt_tokens": _mean([r.prompt_tokens for r in records]),
        "avg_completion_tokens": _mean([r.completion_tokens for r in records]),
        "avg_seconds": _mean([r.seconds for r in records]),
        "errors": sum(bool(r.error) for r in records),
        "truncated_answers": sum(r.truncated for r in records),
        "malformed_answers": sum(r.malformed for r in records),
        "judge_errors": sum(bool(r.judge_error) for r in records),
        "evidence_seen": _mean([r.evidence_seen for r in records if r.evidence_seen is not None]),
        "unsupported_number_rate": (
            sum(len(r.unsupported_numbers) for r in records) / max(1, sum(r.numbers_stated for r in records))
        ),
        "answers_with_unsupported_numbers": _mean([bool(r.unsupported_numbers) for r in records]),
        "repairs_triggered": sum(bool(r.unsupported_before_repair) for r in records),
        **cost_and_latency(records),
    }
    return out


def percentile(values: list[float], q: float) -> float | None:
    """q-th percentile (0-100) with linear interpolation between ranks."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def cost_and_latency(records: list[Record]) -> dict:
    """Cost per question and latency percentiles (W4). Cost is None when any record lacks it
    (runs from before W4, or a model missing from the price table), never a partial sum."""
    costs = [r.cost_usd for r in records]
    known = bool(records) and all(c is not None for c in costs)
    traced = [r for r in records if r.llm_seconds is not None]
    total = sum(r.seconds for r in traced)
    return {
        "cost_per_question": _mean(costs) if known else None,
        "cost_per_1000_questions": _mean(costs) * 1000 if known else None,
        "judge_cost_per_question": _mean([r.judge_cost_usd for r in traced]) if traced else None,
        "avg_llm_calls": _mean([r.llm_calls for r in traced]) if traced else None,
        "cache_hit_share": (sum(r.cached_prompt_tokens for r in traced) / max(1, sum(r.prompt_tokens for r in traced))
                            if traced else None),
        "p50_seconds": percentile([r.seconds for r in records], 50),
        "p95_seconds": percentile([r.seconds for r in records], 95),
        "llm_time_share": sum(r.llm_seconds for r in traced) / total if traced and total else None,
        "retrieval_time_share": sum(r.retrieval_seconds for r in traced) / total if traced and total else None,
    }


def cost_line(records: list[Record]) -> str:
    """One line for terminal reports, e.g. "$0.0011/question · p50 7.2 s · p95 16.0 s"."""
    s = cost_and_latency(records)
    cost = f"${s['cost_per_question']:.4f}/question" if s["cost_per_question"] is not None else "cost n/a (run predates W4)"
    p50, p95 = s["p50_seconds"], s["p95_seconds"]
    return f"{cost} · p50 {p50:.1f} s · p95 {p95:.1f} s" if p50 is not None else cost


ROWS = [
    ("Overall pass rate", "pass_rate", "pct"),
    ("Numeric questions", "pass_numeric", "pct"),
    ("Narrative questions", "pass_narrative", "pct"),
    ("Mixed questions", "pass_mixed", "pct"),
    ("Comparison questions (across companies)", "pass_comparison", "pct"),
    ("Trend questions (across years)", "pass_trend", "pct"),
    ("Sector questions (13 semiconductor companies)", "pass_sector", "pct"),
    ("Multi-hop (two sections)", "pass_multi_hop", "pct"),
    ("Long-tail sections (legal, market risk, controls)", "pass_long_tail", "pct"),
    ("Text-table facts (not in XBRL)", "pass_text_table", "pct"),
    ("Ambiguous company names", "pass_ambiguous", "pct"),
    ("Near-miss unanswerable (refused correctly)", "pass_near_miss", "pct"),
    ("Calculations on text-table figures", "pass_table_calc", "pct"),
    ("Same topic across several companies", "pass_cross_company", "pct"),
    ("Complete enumerations (4+ items)", "pass_enumeration", "pct"),
    ("Unanswerable (refused correctly)", "pass_unanswerable", "pct"),
    ("Gold evidence in passages seen", "evidence_seen", "pct"),
    ("Number accuracy", "number_accuracy", "pct"),
    ("Retrieval hit rate (right section retrieved)", "retrieval_hit_rate", "pct"),
    ("Retrieval MRR", "mrr", "num"),
    ("Key-point coverage (LLM judge)", "key_point_coverage", "pct"),
    ("False refusals on answerable questions", "false_refusal_rate", "pct"),
    ("Answers with invalid citations", "invalid_citation_rate", "pct"),
    ("Unsupported numbers (share of all numbers stated)", "unsupported_number_rate", "pct"),
    ("Answers with any unsupported number", "answers_with_unsupported_numbers", "pct"),
    ("Grounding repairs triggered", "repairs_triggered", "int"),
    ("Avg prompt tokens", "avg_prompt_tokens", "int"),
    ("Avg completion tokens", "avg_completion_tokens", "int"),
    ("Avg seconds per question", "avg_seconds", "sec"),
    ("p50 seconds per question", "p50_seconds", "sec"),
    ("p95 seconds per question", "p95_seconds", "sec"),
    ("Share of answer time in LLM calls", "llm_time_share", "pct"),
    ("Share of answer time in retrieval", "retrieval_time_share", "pct"),
    ("Avg LLM calls per question", "avg_llm_calls", "num"),
    ("Prompt tokens served from cache", "cache_hit_share", "pct"),
    ("Cost per question (USD, list price)", "cost_per_question", "usd"),
    ("Cost per 1,000 questions (USD)", "cost_per_1000_questions", "usd2"),
    ("Judge cost per question (USD, eval only)", "judge_cost_per_question", "usd"),
    ("Truncated answers (hit token limit)", "truncated_answers", "int"),
    ("Malformed answers (tool-call markup)", "malformed_answers", "int"),
    ("Judge errors (no usable verdict)", "judge_errors", "int"),
    ("Run errors", "errors", "int"),
]


def _fmt(value: float | None, style: str) -> str:
    if value is None:
        return "n/a"
    return {"pct": f"{value:.0%}", "num": f"{value:.2f}", "int": f"{value:,.0f}", "sec": f"{value:.1f}",
            "usd": f"${value:.4f}", "usd2": f"${value:,.2f}"}[style]


def summary_markdown(summaries: dict[str, dict], model: str, judge: str) -> str:
    modes = list(summaries)
    lines = [
        f"# Eval results ({datetime.now():%Y-%m-%d})",
        "",
        f"{next(iter(summaries.values()))['questions']} questions · model `{model}` · judge `{judge}`",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for label, key, style in ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(summaries[m][key], style) for m in modes) + " |")
    return "\n".join(lines) + "\n"


GOLDEN_SETS = {"v1": EVALS_DIR / "golden.jsonl", "v2": EVALS_DIR / "golden_v2.jsonl"}


def results_dir(golden: str) -> Path:
    """v1 results stay where they were; v2 results go in their own folder."""
    return EVALS_DIR / "results" if golden == "v1" else EVALS_DIR / "results" / golden


def save(records: list[Record], mode: Mode, out_dir: Path | None = None, tag: str = "") -> Path:
    out_dir = out_dir or EVALS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{mode}{tag}.jsonl"
    path.write_text("".join(json.dumps(r.model_dump()) + "\n" for r in records))
    return path
