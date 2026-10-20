"""Repeated runs and bootstrap confidence intervals.

LLM answers vary from run to run even at temperature 0, and the judge adds its own variance, so a
2-point difference between two configs can be luck. Two tools make comparisons honest:

- repeated runs: report the mean and range of a config's pass rate over several runs. The range
  is the noise floor; a change smaller than it isn't evidence of anything.
- paired bootstrap: resample the questions with replacement many times and recompute the
  difference in pass rate each time. The middle 95% of those differences is the confidence
  interval. Pairing (both configs answer the same questions) cancels per-question difficulty.
"""

import random
from collections import defaultdict

from pydantic import BaseModel


class RunSummary(BaseModel):
    runs: int
    questions: int
    pass_rates: list[float]
    mean: float
    low: float
    high: float
    # Share of questions whose pass/fail result changed between runs of the same config.
    flip_rate: float


def per_question_rate(runs: list[list[dict]]) -> dict[str, float]:
    """Average pass rate per question id across runs (1.0 = passed every run)."""
    passed = defaultdict(list)
    for run in runs:
        for r in run:
            passed[r["id"]].append(1.0 if r["passed"] else 0.0)
    return {qid: sum(v) / len(v) for qid, v in passed.items()}


def summarize_runs(runs: list[list[dict]]) -> RunSummary:
    rates = [sum(r["passed"] for r in run) / len(run) for run in runs]
    per_q = per_question_rate(runs)
    flips = sum(0 < rate < 1 for rate in per_q.values())
    return RunSummary(
        runs=len(runs), questions=len(per_q), pass_rates=[round(x, 4) for x in rates],
        mean=sum(rates) / len(rates), low=min(rates), high=max(rates),
        flip_rate=flips / len(per_q) if per_q else 0.0,
    )


class Comparison(BaseModel):
    questions: int
    rate_a: float
    rate_b: float
    diff: float  # b - a
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def paired_bootstrap(runs_a: list[list[dict]], runs_b: list[list[dict]], iterations: int = 10_000,
                     seed: int = 0) -> Comparison:
    """95% CI for (pass rate of B) - (pass rate of A), over the questions both configs answered."""
    a, b = per_question_rate(runs_a), per_question_rate(runs_b)
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    n = len(diffs)
    samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return Comparison(
        questions=n,
        rate_a=sum(a[i] for i in ids) / n,
        rate_b=sum(b[i] for i in ids) / n,
        diff=sum(diffs) / n,
        ci_low=samples[int(0.025 * iterations)],
        ci_high=samples[int(0.975 * iterations) - 1],
    )
