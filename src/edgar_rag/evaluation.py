"""Run the golden question set through a mode ("agent" or "simple"), score it, and summarize."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from edgar_rag.scoring import Expected, judge, number_found, retrieval_scores

Mode = Literal["agent", "simple"]
EVALS_DIR = Path("evals")
PASS_COVERAGE = 2 / 3  # narrative answers must cover at least this share of key points
CATEGORIES = ("numeric", "narrative", "mixed", "comparison", "trend", "unanswerable")


class GoldenItem(BaseModel):
    id: str
    category: Literal["numeric", "narrative", "mixed", "comparison", "trend", "unanswerable"]
    question: str
    expected_numbers: list[Expected] = []
    expected_items: list[str] = []
    key_points: list[str] = []
    should_refuse: bool = False


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
    judge_error: str = ""  # the judge gave no usable verdict (excluded from judge metrics)
    error: str = ""


def load_golden(path: Path = EVALS_DIR / "golden.jsonl") -> list[GoldenItem]:
    return [GoldenItem.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _item_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split(":")[1]  # "<accession>:<item>:<index>"


def _run_mode(question: str, mode: Mode) -> dict:
    """Answer one question; return the parts scoring needs, independent of mode."""
    if mode == "simple":
        from edgar_rag.answer import answer

        r = answer(question)
        return {
            "answer": r.text,
            "tools_used": ["search"],
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
        "retrieved_items": [_item_from_chunk_id(s.ref) for s in r.sources if s.kind == "filing"],
        "top_score": max((s.score for s in r.sources if s.score is not None), default=None),
        "invalid_citations": r.invalid_citations,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "truncated": r.truncated,
    }


def evaluate_one(item: GoldenItem, mode: Mode) -> Record:
    start = time.monotonic()
    try:
        out = _run_mode(item.question, mode)
        error = ""
    except Exception as e:  # noqa: BLE001 - one failure shouldn't stop the run; it's scored as a fail
        out = {"answer": "", "tools_used": [], "retrieved_items": [], "invalid_citations": [],
               "prompt_tokens": 0, "completion_tokens": 0, "truncated": False, "top_score": None}
        error = f"{type(e).__name__}: {e}"
    seconds = time.monotonic() - start

    answer = out["answer"]
    numbers = [number_found(e, answer) for e in item.expected_numbers]
    hit, rr = (None, None)
    if item.expected_items:
        hit, rr = retrieval_scores(out["retrieved_items"], item.expected_items)
    verdict = judge(item.question, answer, item.key_points) if answer else None
    covered = verdict.key_points if verdict else [False] * len(item.key_points)
    refused = verdict.refused if verdict else True

    if item.category == "unanswerable":
        passed = refused
    else:
        coverage_ok = not item.key_points or sum(covered) / len(covered) >= PASS_COVERAGE
        passed = all(numbers) and coverage_ok and not refused and not error

    return Record(
        id=item.id, category=item.category, mode=mode, question=item.question, **out,
        seconds=round(seconds, 1), numbers_found=numbers, retrieval_hit=hit,
        reciprocal_rank=rr, key_points_covered=covered, refused=refused, passed=passed,
        judge_error=verdict.error if verdict else "", error=error,
    )


def run(mode: Mode, items: list[GoldenItem], workers: int = 4) -> list[Record]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda it: evaluate_one(it, mode), items))


# --- Summary -------------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[Record]) -> dict:
    answerable = [r for r in records if r.category != "unanswerable"]
    unanswerable = [r for r in records if r.category == "unanswerable"]
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
        "judge_errors": sum(bool(r.judge_error) for r in records),
    }
    return out


ROWS = [
    ("Overall pass rate", "pass_rate", "pct"),
    ("Numeric questions", "pass_numeric", "pct"),
    ("Narrative questions", "pass_narrative", "pct"),
    ("Mixed questions", "pass_mixed", "pct"),
    ("Comparison questions (across companies)", "pass_comparison", "pct"),
    ("Trend questions (across years)", "pass_trend", "pct"),
    ("Unanswerable (refused correctly)", "pass_unanswerable", "pct"),
    ("Number accuracy", "number_accuracy", "pct"),
    ("Retrieval hit rate (right section retrieved)", "retrieval_hit_rate", "pct"),
    ("Retrieval MRR", "mrr", "num"),
    ("Key-point coverage (LLM judge)", "key_point_coverage", "pct"),
    ("False refusals on answerable questions", "false_refusal_rate", "pct"),
    ("Answers with invalid citations", "invalid_citation_rate", "pct"),
    ("Avg prompt tokens", "avg_prompt_tokens", "int"),
    ("Avg completion tokens", "avg_completion_tokens", "int"),
    ("Avg seconds per question", "avg_seconds", "sec"),
    ("Truncated answers (hit token limit)", "truncated_answers", "int"),
    ("Judge errors (no usable verdict)", "judge_errors", "int"),
    ("Run errors", "errors", "int"),
]


def _fmt(value: float | None, style: str) -> str:
    if value is None:
        return "n/a"
    return {"pct": f"{value:.0%}", "num": f"{value:.2f}", "int": f"{value:,.0f}", "sec": f"{value:.1f}"}[style]


def summary_markdown(summaries: dict[str, dict], model: str) -> str:
    modes = list(summaries)
    lines = [
        f"# Eval results ({datetime.now():%Y-%m-%d})",
        "",
        f"{next(iter(summaries.values()))['questions']} questions · model `{model}` · judge `{model}`",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for label, key, style in ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(summaries[m][key], style) for m in modes) + " |")
    return "\n".join(lines) + "\n"


def save(records: list[Record], mode: Mode) -> Path:
    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{mode}.jsonl"
    path.write_text("".join(json.dumps(r.model_dump()) + "\n" for r in records))
    return path
