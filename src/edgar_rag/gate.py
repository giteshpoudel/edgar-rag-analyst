"""CI regression gate (W6): run a fixed smoke set and compare with a committed baseline.

The baseline (evals/baseline.json) is built from several repeated runs of the same code, so the
gate knows how much scores move between identical runs and only fails on drops larger than that.

Blocking checks:
    pass rate        >= baseline mean - tolerance (tolerance from the measured run-to-run spread)
    number accuracy  >= the lowest value seen across the baseline runs ("doesn't drop")
    run errors       == 0 (with retries in place, an error means something is broken)
Reported but not blocking: cost per question, p95 latency, unsupported numbers.
"""

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from edgar_rag.evaluation import Record, cost_and_latency, summarize

BASELINE_PATH = Path("evals/baseline.json")


class Baseline(BaseModel):
    golden: str
    split: str
    questions: int
    runs: int
    pass_rates: list[float]  # one per baseline run
    pass_rate: float  # mean
    tolerance: float  # allowed drop below the mean, in pass-rate points (0-1)
    tolerance_rule: str  # how the tolerance was chosen, for the reader
    number_accuracy_floor: float
    always_passed: list[str]  # passed in every baseline run: a failure here is news
    cost_per_question: float | None = None
    p95_seconds: float | None = None
    created: str
    config: dict


class Check(BaseModel):
    name: str
    value: float
    threshold: float
    ok: bool
    fmt: str = "pct"


class GateResult(BaseModel):
    checks: list[Check]
    newly_failing: list[str]
    report: dict

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)


def tolerance_for(pass_rates: list[float], questions: int) -> tuple[float, str]:
    """Allowed drop: the widest gap seen between the baseline mean and any single run, plus one
    question of slack, so an unlucky but normal run never fails the build."""
    mean = sum(pass_rates) / len(pass_rates)
    spread = max(mean - r for r in pass_rates)
    one_question = 1 / questions
    rule = (f"widest drop of a baseline run below the mean ({spread:.1%}) + one question "
            f"({one_question:.1%})")
    return spread + one_question, rule


def make_baseline(runs: list[list[Record]], golden: str, split: str, config: dict) -> Baseline:
    rates = [sum(r.passed for r in run) / len(run) for run in runs]
    questions = len(runs[0])
    tolerance, rule = tolerance_for(rates, questions)
    number_acc = [summarize(run)["number_accuracy"] for run in runs]
    passed_sets = [{r.id for r in run if r.passed} for run in runs]
    pooled = [r for run in runs for r in run]
    costs = cost_and_latency(pooled)
    return Baseline(
        golden=golden, split=split, questions=questions, runs=len(runs),
        pass_rates=[round(x, 4) for x in rates], pass_rate=round(sum(rates) / len(rates), 4),
        tolerance=round(tolerance, 4), tolerance_rule=rule,
        number_accuracy_floor=round(min(number_acc), 4),
        always_passed=sorted(set.intersection(*passed_sets)),
        cost_per_question=costs["cost_per_question"], p95_seconds=costs["p95_seconds"],
        created=date.today().isoformat(), config=config,
    )


def check(records: list[Record], baseline: Baseline) -> GateResult:
    s = summarize(records)
    costs = cost_and_latency(records)
    errors = [r.id for r in records if r.error]
    checks = [
        Check(name="Pass rate", value=s["pass_rate"], threshold=baseline.pass_rate - baseline.tolerance,
              ok=s["pass_rate"] >= baseline.pass_rate - baseline.tolerance - 1e-9),
        Check(name="Number accuracy", value=s["number_accuracy"] or 0.0,
              threshold=baseline.number_accuracy_floor,
              ok=(s["number_accuracy"] or 0.0) >= baseline.number_accuracy_floor - 1e-9),
        Check(name="Run errors", value=len(errors), threshold=0, ok=not errors, fmt="int"),
    ]
    failed_now = {r.id for r in records if not r.passed}
    report = {
        "questions": len(records),
        "cost_per_question": costs["cost_per_question"],
        "p95_seconds": costs["p95_seconds"],
        "unsupported_number_rate": s["unsupported_number_rate"],
        "errors": errors,
    }
    return GateResult(checks=checks, newly_failing=sorted(failed_now & set(baseline.always_passed)),
                      report=report)


def _fmt(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    return {"pct": f"{value:.1%}", "int": f"{value:.0f}", "usd": f"${value:.4f}", "sec": f"{value:.1f} s"}[fmt]


def markdown(result: GateResult, baseline: Baseline, records: list[Record]) -> str:
    verdict = "✅ Passed" if result.passed else "❌ Failed"
    lines = [
        f"## Eval gate: {verdict}",
        "",
        (f"{len(records)} questions ({baseline.golden} {baseline.split} split) compared with "
        f"`evals/baseline.json` ({baseline.runs} runs, {baseline.created})."),
        "",
        "| Check | Baseline | This run | Required | |",
        "|---|---|---|---|---|",
    ]
    base_values = {"Pass rate": baseline.pass_rate, "Number accuracy": baseline.number_accuracy_floor,
                   "Run errors": 0}
    for c in result.checks:
        threshold = "none" if c.name == "Run errors" else f"≥ {_fmt(c.threshold, c.fmt)}"
        lines.append(f"| {c.name} | {_fmt(base_values[c.name], c.fmt)} | **{_fmt(c.value, c.fmt)}** | "
                     f"{threshold} | {'✅' if c.ok else '❌'} |")
    r = result.report
    lines += [
        (f"| Cost per question (info) | {_fmt(baseline.cost_per_question, 'usd')} | "
        f"{_fmt(r['cost_per_question'], 'usd')} | | |"),
        f"| p95 seconds (info) | {_fmt(baseline.p95_seconds, 'sec')} | {_fmt(r['p95_seconds'], 'sec')} | | |",
        f"| Unsupported numbers (info) | | {_fmt(r['unsupported_number_rate'], 'pct')} | | |",
        "",
        f"Pass-rate tolerance: {baseline.tolerance:.1%} below the baseline mean: {baseline.tolerance_rule}.",
    ]
    failed = [x for x in records if not x.passed]
    if failed:
        lines += ["", "**Failed questions** (★ = passed in every baseline run):", ""]
        for x in failed:
            star = "★ " if x.id in result.newly_failing else ""
            why = x.error or ("refused" if x.refused else "missed numbers or key points")
            lines.append(f"- {star}`{x.id}` ({x.category}): {why[:120]}")
    return "\n".join(lines) + "\n"


def load_baseline(path: Path = BASELINE_PATH) -> Baseline:
    return Baseline.model_validate_json(path.read_text())


def save_baseline(baseline: Baseline, path: Path = BASELINE_PATH) -> None:
    path.write_text(json.dumps(baseline.model_dump(), indent=2) + "\n")
