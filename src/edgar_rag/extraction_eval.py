"""Extraction eval: the claim extractor against hand labels (v3 Phase 2).

Scored only inside the sampled blocks of each labelled filing (evals/v3/labels/manifest.json),
since that's where the labels are complete. Earlier-year "link target" labels outside the
sample are used for change labels (Phase 3), not here.

Matching is deterministic, so the numbers are exact and free (no LLM judge):
    an extracted claim matches a labelled one when both are in the same block, have the same
    claim_type, and their quotes overlap by at least half of the shorter quote.
"At least half of the shorter" rather than IoU, so a correct three-sentence quote that
contains the labeller's one sentence still counts. Matches are one-to-one, best overlap first.

Precision = matched / extracted. Recall = matched / labelled. Confidence intervals come from a
bootstrap over blocks (claims in one block aren't independent: one long paragraph can hold
five), pooled across the filings of a split.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from edgar_rag import labeling
from edgar_rag.claims import Claim
from edgar_rag.labeling import LabelRow, normalize

MIN_OVERLAP = 0.5
BOOTSTRAP = 2000


def span(quote: str, block_text: str) -> tuple[int, int] | None:
    q, t = normalize(quote), normalize(block_text)
    i = t.find(q)
    return (i, i + len(q)) if i >= 0 else None


def overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1, min(a[1] - a[0], b[1] - b[0]))


@dataclass
class Unit:
    """One sampled block: what was labelled in it, what was extracted, what matched."""

    filing: str
    block: int
    gold: list[str] = field(default_factory=list)  # claim types
    pred: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)  # claim types of matched pairs
    matched_any_type: int = 0  # pairs whose quotes match whatever the type


def match_block(gold: list[LabelRow], pred: list[Claim], block_text: str, typed: bool = True) -> list[tuple[int, int]]:
    """One-to-one (gold index, pred index) matches, best overlap first."""
    g_spans = [span(g.quote, block_text) for g in gold]
    p_spans = [span(p.quote, block_text) for p in pred]
    pairs = []
    for i, (g, gs) in enumerate(zip(gold, g_spans)):
        for j, (p, ps) in enumerate(zip(pred, p_spans)):
            if gs and ps and (not typed or g.claim_type == p.claim_type):
                o = overlap(gs, ps)
                if o >= MIN_OVERLAP:
                    pairs.append((o, i, j))
    used_g, used_p, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def score_filing(ticker: str, fiscal_year: int, chosen: set[int], gold: list[LabelRow],
                 pred: list[Claim], blocks: list[labeling.Block]) -> list[Unit]:
    by_number = {b.number: b for b in blocks}
    units = []
    for n in sorted(chosen):
        g = [r for r in gold if r.block == n]
        p = [c for c in pred if c.block == n]
        text = by_number[n].text
        typed = match_block(g, p, text)
        untyped = match_block(g, p, text, typed=False)
        units.append(Unit(f"{ticker}_{fiscal_year}", n, [r.claim_type for r in g], [c.claim_type for c in p],
                          [g[i].claim_type for i, _ in typed], len(untyped)))
    return units


def metrics(units: list[Unit]) -> dict:
    gold = sum(len(u.gold) for u in units)
    pred = sum(len(u.pred) for u in units)
    tp = sum(len(u.matched) for u in units)
    any_type = sum(u.matched_any_type for u in units)
    p = tp / pred if pred else None
    r = tp / gold if gold else None
    f1 = 2 * p * r / (p + r) if p and r else None
    return {"labelled": gold, "extracted": pred, "matched": tp, "precision": p, "recall": r, "f1": f1,
            # Pairs whose quotes match regardless of type: the gap to `matched` is type confusion.
            "matched_ignoring_type": any_type}


def bootstrap(units: list[Unit], seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% intervals for precision, recall and F1, resampling blocks with replacement."""
    rng = random.Random(seed)
    samples = {"precision": [], "recall": [], "f1": []}
    for _ in range(BOOTSTRAP):
        m = metrics([rng.choice(units) for _ in units])
        for k, xs in samples.items():
            if m[k] is not None:
                xs.append(m[k])
    out = {}
    for k, xs in samples.items():
        xs.sort()
        out[k] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs else (None, None)
    return out


def per_type(units: list[Unit]) -> dict[str, dict]:
    out = {}
    for t in labeling.CLAIM_TYPES:
        g = sum(u.gold.count(t) for u in units)
        p = sum(u.pred.count(t) for u in units)
        m = sum(u.matched.count(t) for u in units)
        if g or p:
            out[t] = {"labelled": g, "extracted": p, "matched": m,
                      "precision": m / p if p else None, "recall": m / g if g else None}
    return out


def evaluate(split: str, extractor: str, load_claims) -> dict:
    """Score every labelled filing in a split. load_claims(ticker, fy) -> list[Claim]."""
    manifest = json.loads(labeling.MANIFEST.read_text())
    units: list[Unit] = []
    filings = []
    for pair in manifest["pairs"]:
        if pair["split"] != split:
            continue
        for fy_s, info in pair["years"].items():
            fy = int(fy_s)
            gold = labeling.read_labels(Path(info["labels"]))
            chosen = set(info["chosen"])
            gold = [r for r in gold if r.block in chosen]  # link targets outside the sample aren't scored
            _, html = labeling.filing_html(pair["ticker"], fy)
            blocks = labeling.blocks_for_filing(html)
            if len(blocks) != info["blocks"]:
                raise RuntimeError(f"{pair['ticker']} FY{fy}: blocks changed since labelling; scores would be wrong")
            pred = [c for c in load_claims(pair["ticker"], fy) if c.block in chosen]
            u = score_filing(pair["ticker"], fy, chosen, gold, pred, blocks)
            units += u
            filings.append({"filing": f"{pair['ticker']}_{fy}", **metrics(u)})
    return {"split": split, "extractor": extractor, "blocks": len(units), "overall": metrics(units),
            "ci95": bootstrap(units), "per_type": per_type(units), "per_filing": filings}


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def markdown(report: dict) -> str:
    o, ci = report["overall"], report["ci95"]
    lines = [(f"Extraction eval ({report['split']} split, extractor {report['extractor']}, "
             f"{report['blocks']} sampled blocks)"), "",
             "| Metric | Value | 95% CI |", "|---|---|---|"]
    for k in ("precision", "recall", "f1"):
        lo, hi = ci[k]
        lines.append(f"| {k.capitalize() if k != 'f1' else 'F1'} | {_pct(o[k])} | "
                     f"{_pct(lo)} to {_pct(hi)} |")
    lines += ["", (f"{o['matched']} matched of {o['extracted']} extracted and {o['labelled']} labelled "
              f"({o['matched_ignoring_type']} would match ignoring claim type)."), "",
              "| Claim type | Labelled | Extracted | Matched | Precision | Recall |", "|---|---|---|---|---|---|"]
    for t, m in report["per_type"].items():
        lines.append(f"| {t} | {m['labelled']} | {m['extracted']} | {m['matched']} | "
                     f"{_pct(m['precision'])} | {_pct(m['recall'])} |")
    lines += ["", "| Filing | Labelled | Extracted | Matched | Precision | Recall |", "|---|---|---|---|---|---|"]
    for f in report["per_filing"]:
        lines.append(f"| {f['filing']} | {f['labelled']} | {f['extracted']} | {f['matched']} | "
                     f"{_pct(f['precision'])} | {_pct(f['recall'])} |")
    return "\n".join(lines) + "\n"
