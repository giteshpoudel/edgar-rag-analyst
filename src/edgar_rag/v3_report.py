"""The v2-vs-v3 comparison report (v3 Phase 5b): same questions, same runs, per judge.

For one judge it reads A's and B's run files (the same test questions, answered 3 times each)
and writes: overall and per-category pass rates with paired bootstrap 95% CIs on B - A, the
run-to-run spread, cost and latency per question, how often B used the v3 tools, and the
questions where B did worse than A ("where v3 doesn't help").
"""

import json
from collections import defaultdict
from pathlib import Path

from edgar_rag.evaluation import CATEGORIES, Record, cost_and_latency, percentile
from edgar_rag.stats import paired_bootstrap, per_question_rate, summarize_runs

V3_TOOLS = {"claim_history", "compare_claims", "graph_neighbors", "find_entity"}


def load_runs(files: list[Path]) -> list[list[dict]]:
    return [[json.loads(line) for line in f.read_text().splitlines() if line.strip()] for f in files]


EPS = 1e-9  # a bootstrap bound of 4.7e-18 is zero, not "above zero"


def significant(c) -> bool:
    return c.ci_low > EPS or c.ci_high < -EPS


def _ci(c) -> str:
    mark = " *" if significant(c) else ""
    return f"{c.diff:+.1%} [{c.ci_low:+.1%}, {c.ci_high:+.1%}]{mark}"


def section(name_a: str, runs_a: list[list[dict]], name_b: str, runs_b: list[list[dict]], judge: str) -> str:
    sa, sb = summarize_runs(runs_a), summarize_runs(runs_b)
    overall = paired_bootstrap(runs_a, runs_b)
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in runs_a[0])]
    lines = [f"## Judge: `{judge}`", "",
             (f"{overall.questions} test questions × {sa.runs} runs per system. Differences are {name_b} − {name_a}, "
             "with a paired bootstrap 95% CI over questions; * marks a CI that excludes zero (a bound within 1e-9 of zero counts as zero)."), "",
             f"| | n | {name_a} | {name_b} | Difference (95% CI) |", "|---|---|---|---|---|",
             f"| **Overall** | {overall.questions} | {overall.rate_a:.1%} | {overall.rate_b:.1%} | {_ci(overall)} |"]
    for cat in cats:
        fa = [[r for r in run if r["category"] == cat] for run in runs_a]
        fb = [[r for r in run if r["category"] == cat] for run in runs_b]
        c = paired_bootstrap(fa, fb)
        lines.append(f"| {cat} | {c.questions} | {c.rate_a:.1%} | {c.rate_b:.1%} | {_ci(c)} |")
    lines += ["", "| Run-to-run | Pass rate per run | Range | Questions that flipped between runs |", "|---|---|---|---|"]
    for name, s in ((name_a, sa), (name_b, sb)):
        lines.append(f"| {name} | {', '.join(f'{x:.1%}' for x in s.pass_rates)} | {s.low:.1%}–{s.high:.1%} | {s.flip_rate:.1%} |")

    per_a, per_b = per_question_rate(runs_a), per_question_rate(runs_b)
    worse = sorted((q for q in per_a if per_b.get(q, 0) < per_a[q]), key=lambda q: per_b[q] - per_a[q])
    category = {r["id"]: r["category"] for r in runs_a[0]}
    lines += ["", f"**Where {name_b} doesn't help** (questions it passed less often than {name_a}): "
              + (", ".join(f"{q} ({category[q]}, {per_a[q]:.0%}→{per_b[q]:.0%})" for q in worse) or "none")]
    same = [c for c in cats if paired_bootstrap([[r for r in run if r["category"] == c] for run in runs_a],
                                                [[r for r in run if r["category"] == c] for run in runs_b]).diff <= 0]
    lines.append(f"Categories with no gain: {', '.join(same) or 'none'}.")
    return "\n".join(lines) + "\n"


def cost_section(name_a: str, runs_a: list[list[dict]], name_b: str, runs_b: list[list[dict]]) -> str:
    lines = ["## Cost and latency (answering; judge cost separate)", "",
             ("| | Cost per question | Judge cost per question | p50 s | p95 s | Avg prompt tokens | Avg LLM calls | "
             "Answers using a v3 tool |"), "|---|---|---|---|---|---|---|---|"]
    for name, runs in ((name_a, runs_a), (name_b, runs_b)):
        recs = [Record.model_validate(r) for run in runs for r in run]
        s = cost_and_latency(recs)
        uses = sum(bool(V3_TOOLS & set(r.tools_used)) for r in recs) / len(recs)
        cost = f"${s['cost_per_question']:.4f}" if s["cost_per_question"] is not None else "n/a"
        judge = f"${s['judge_cost_per_question']:.4f}" if s["judge_cost_per_question"] is not None else "n/a"
        lines.append(f"| {name} | {cost} | {judge} | {percentile([r.seconds for r in recs], 50):.1f} | "
                     f"{percentile([r.seconds for r in recs], 95):.1f} | {sum(r.prompt_tokens for r in recs) / len(recs):,.0f} | "
                     f"{s['avg_llm_calls']:.2f} | {uses:.0%} |")
    by_cat = defaultdict(list)
    for run in runs_b:
        for r in run:
            by_cat[r["category"]].append(bool(V3_TOOLS & set(r["tools_used"])))
    lines += ["", f"{name_b} used a v3 tool in: " + ", ".join(f"{c} {sum(v) / len(v):.0%}" for c, v in by_cat.items()) + "."]
    return "\n".join(lines) + "\n"
