"""Judge calibration (v3 Phase 6): how far can each LLM judge's grades be trusted?

Reference grades (evals/v3/judge_calibration/reference_grades.jsonl) are 80 answers graded blind
by Claude, a model independent of both judges, not by the owner, so "agreement" here means
agreement with another model's careful reading:
  60 natural answers, sampled at random from v3 dev runs (both systems) and v2-set dev runs
  20 perturbed answers written with one known error each (wrong year, wrong company list,
     wrong figure, "no change" claims, confident answers to unanswerable questions), because
     natural answers nearly all pass and kappa can't be measured on one class.

Two units are compared:
  key point   each key point covered or not (the judge's actual job)
  verdict     the judge-dependent part of the pass rule: refused for unanswerable questions,
              otherwise key-point coverage >= 2/3 and not refused

Cohen's kappa = (observed agreement - chance agreement) / (1 - chance agreement): 1 is perfect,
0 is no better than chance given each grader's base rates. A 95% CI comes from resampling answers.
"""

import asyncio
import json
import random
from pathlib import Path

from edgar_rag.evaluation import GOLDEN_SETS, PASS_COVERAGE, REFUSAL_CATEGORIES, load_golden
from edgar_rag.scoring import ajudge

DIR = Path("evals/v3/judge_calibration")
REFERENCE = DIR / "reference_grades.jsonl"
REFERENCE_NAME = "reference (Claude, blind)"


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return None if chance == 1 else (observed - chance) / (1 - chance)


def agreement(a: list[bool], b: list[bool]) -> float | None:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else None


def verdict(category: str, key_points: list[bool], refused: bool) -> bool:
    if category in REFUSAL_CATEGORIES:
        return refused
    coverage_ok = not key_points or sum(key_points) / len(key_points) >= PASS_COVERAGE
    return coverage_ok and not refused


def load_reference(path: Path = REFERENCE) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    goldens = {g: {i.id: i for i in load_golden(GOLDEN_SETS[g])} for g in {r["golden"] for r in rows}}
    stored: dict[str, dict[str, str]] = {}
    for r in rows:
        r["item"] = goldens[r["golden"]][r["id"]]
        if r["kind"] == "natural":
            f = r["answer_file"]
            if f not in stored:
                stored[f] = {json.loads(line)["id"]: json.loads(line)["answer"] for line in Path(f).read_text().splitlines()}
            r["answer"] = stored[f][r["id"]]
    return rows


async def ajudge_all(rows: list[dict], model: str, concurrency: int = 8) -> list[dict]:
    gate = asyncio.Semaphore(concurrency)

    async def one(r: dict) -> dict:
        it = r["item"]
        async with gate:
            v = await ajudge(it.question, r["answer"], it.key_points, model=model)
        return {"cid": r["cid"], "key_points": v.key_points, "refused": v.refused, "error": v.error, "raw": v.raw}

    return list(await asyncio.gather(*(one(r) for r in rows)))


def verdicts_path(model: str) -> Path:
    return DIR / f"verdicts_{model}.jsonl"


def _units(rows: list[dict], grades: dict[str, dict]) -> tuple[list[bool], list[bool]]:
    """(key-point labels, verdict labels) in row order, for one grader."""
    kps, verdicts = [], []
    for r in rows:
        g = grades[r["cid"]]
        kps += g["key_points"]
        verdicts.append(verdict(r["item"].category, g["key_points"], g["refused"]))
    return kps, verdicts


def _boot_kappa(rows, a, b, unit: int, n: int = 2000, seed: int = 7) -> tuple[float, float] | None:
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        k = kappa(*(_units(sample, g)[unit] for g in (a, b)))
        if k is not None:
            stats.append(k)
    if not stats:
        return None
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]


def compare(rows: list[dict], a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Agreement and kappa between two graders over the rows both graded without error."""
    ok = [r for r in rows if not a[r["cid"]].get("error") and not b[r["cid"]].get("error")]
    out = {"answers": len(ok), "excluded_judge_errors": len(rows) - len(ok)}
    for unit, name in ((0, "key_point"), (1, "verdict")):
        ua, ub = _units(ok, a)[unit], _units(ok, b)[unit]
        out[name] = {"n": len(ua), "agreement": agreement(ua, ub), "kappa": kappa(ua, ub),
                     "kappa_ci95": _boot_kappa(ok, a, b, unit),
                     "a_true_rate": sum(ua) / len(ua) if ua else None, "b_true_rate": sum(ub) / len(ub) if ub else None}
    for kind in ("natural", "perturbed"):  # verdict agreement on each half
        part = [r for r in ok if r["kind"] == kind]
        ua, ub = _units(part, a)[1], _units(part, b)[1]
        out[f"verdict_agreement_{kind}"] = agreement(ua, ub)
    # Perturbed answers all carry a known error: how many did the grader fail?
    pert = [r for r in ok if r["kind"] == "perturbed"]
    out["perturbed_caught_by_b"] = sum(not _units([r], b)[1][0] for r in pert) / len(pert) if pert else None
    out["disagreements"] = [r["cid"] for r in ok if _units([r], a)[1] != _units([r], b)[1]]
    return out


def report(rows: list[dict], graders: dict[str, dict[str, dict]]) -> str:
    """Markdown: every judge against the reference, then judge against judge."""
    names = list(graders)
    ref = graders[REFERENCE_NAME]
    lines = ["# Judge calibration (v3 Phase 6)", "",
             (f"{len(rows)} answers ({sum(r['kind'] == 'natural' for r in rows)} natural, "
             f"{sum(r['kind'] == 'perturbed' for r in rows)} perturbed with a known error), "
             f"{sum(len(r['key_points']) for r in rows)} key points. Reference grades are Claude's, graded blind; "
             "agreement with them is agreement with an independent model, not with a human expert."), "",
             "| Judge vs | Unit | n | Agreement | Cohen's kappa (95% CI) | Judge 'true' rate | Reference 'true' rate |",
             "|---|---|---|---|---|---|---|"]

    def row(label, c):
        for unit in ("key_point", "verdict"):
            u = c[unit]
            ci = f" ({u['kappa_ci95'][0]:.2f} to {u['kappa_ci95'][1]:.2f})" if u["kappa_ci95"] else ""
            k = f"{u['kappa']:.2f}{ci}" if u["kappa"] is not None else "n/a"
            lines.append(f"| {label} | {unit.replace('_', ' ')} | {u['n']} | {u['agreement']:.0%} | {k} | "
                         f"{u['b_true_rate']:.0%} | {u['a_true_rate']:.0%} |")

    judges = [n for n in names if n != REFERENCE_NAME]
    for j in judges:
        row(f"`{j}` vs reference", compare(rows, ref, graders[j]))
    for i, j in enumerate(judges):
        for k in judges[i + 1:]:
            row(f"`{k}` vs `{j}`", compare(rows, graders[j], graders[k]))
    lines += ["", ("| Judge | Verdict agreement, natural | Verdict agreement, perturbed | Perturbed answers failed | "
              "Judge errors | Disagreements with reference |"), "|---|---|---|---|---|---|"]
    for j in judges:
        c = compare(rows, ref, graders[j])
        lines.append(f"| `{j}` | {c['verdict_agreement_natural']:.0%} | {c['verdict_agreement_perturbed']:.0%} | "
                     f"{c['perturbed_caught_by_b']:.0%} | {c['excluded_judge_errors']} | {', '.join(c['disagreements']) or 'none'} |")
    return "\n".join(lines) + "\n"


def reference_grades(rows: list[dict]) -> dict[str, dict]:
    return {r["cid"]: {"key_points": r["key_points"], "refused": r["refused"]} for r in rows}


def load_verdicts(model: str) -> dict[str, dict]:
    return {v["cid"]: v for v in map(json.loads, verdicts_path(model).read_text().splitlines())}
