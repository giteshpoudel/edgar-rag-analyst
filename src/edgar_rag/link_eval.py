"""Link eval: year-to-year change labels against the labelled pairs (v3 Phase 3).

Gold, per labelled pair (Y-1, Y):
    each labelled year-Y claim has a status (new, changed, unchanged) and, unless new, the
    year Y-1 claim it links to (which may be a "link target" outside the sampled blocks);
    each labelled year Y-1 claim in the sampled blocks that no year-Y claim links to is dropped.

Gold claims are mapped onto extracted claims with the extraction eval's matching rule (same
block, same type, quote overlap >= half the shorter quote, one-to-one). Then:

    status correct   predicted status == gold status, and for changed/unchanged the predicted
                     prior claim is the extracted match of the gold prior claim
    wrong prior      changed/unchanged on both sides, but linked to a different earlier claim

Two views:
    given extracted  only gold items whose claims (and gold prior) were all extracted: measures
                     the linker by itself
    end to end       every gold item; one the extractor missed counts as wrong ("missed")

Dropped caveat: labels cover sampled blocks only. An earlier claim whose later version sits in an
unsampled block is labelled dropped although it isn't, so dropped accuracy reads low.
"""

import json
import random
from collections import Counter
from pathlib import Path

from edgar_rag import labeling
from edgar_rag.extraction_eval import match_block

STATUSES = ("new", "changed", "unchanged", "dropped")
BOOTSTRAP = 2000


def gold_to_pred(gold: list[labeling.LabelRow], pred: list, blocks: dict[int, labeling.Block]) -> dict[str, str]:
    """gold claim_id -> extracted claim uid, block by block."""
    out = {}
    for n in sorted({g.block for g in gold}):
        gs = [g for g in gold if g.block == n]
        ps = [p for p in pred if p["block"] == n]
        for i, j in match_block(gs, [_AsClaim(p) for p in ps], blocks[n].text):
            out[gs[i].claim_id] = ps[j]["uid"]
    return out


class _AsClaim:
    """match_block reads .quote, .claim_type and .block."""

    def __init__(self, row) -> None:
        self.quote, self.claim_type, self.block = row["quote"], row["claim_type"], row["block"]


def score_pair(later_gold, earlier_gold, chosen_earlier: set[int], m_later: dict, m_earlier: dict,
               links: list) -> list[dict]:
    """One item per gold later claim and per gold dropped earlier claim."""
    by_claim = {k["claim_uid"]: k for k in links if k["claim_uid"]}
    prior_linked = {k["prior_uid"] for k in links if k["claim_uid"] and k["prior_uid"]}
    items = []
    for g in later_gold:
        item = {"gold": g.change, "claim": g.claim_id, "block": g.block}
        uid = m_later.get(g.claim_id)
        gold_prior = m_earlier.get(g.prior_claim_id) if g.prior_claim_id else None
        extracted = uid is not None and (g.change == "new" or gold_prior is not None)
        item["extracted"] = extracted
        if uid is None:
            item["pred"] = "missed"
        else:
            k = by_claim[uid]
            item["pred"] = k["status"]
            if k["status"] in ("changed", "unchanged") and g.change != "new" and k["prior_uid"] != gold_prior:
                item["pred"] = "wrong prior"
        item["correct"] = item["pred"] == g.change
        items.append(item)
    linked_ids = {g.prior_claim_id for g in later_gold if g.prior_claim_id}
    for g in earlier_gold:
        if g.block not in chosen_earlier or g.claim_id in linked_ids:
            continue
        item = {"gold": "dropped", "claim": g.claim_id, "block": g.block}
        uid = m_earlier.get(g.claim_id)
        item["extracted"] = uid is not None
        item["pred"] = "missed" if uid is None else ("linked" if uid in prior_linked else "dropped")
        item["correct"] = item["pred"] == "dropped"
        items.append(item)
    return items


def summarise(items: list[dict]) -> dict:
    n = len(items)
    correct = sum(i["correct"] for i in items)
    return {"items": n, "correct": correct, "accuracy": correct / n if n else None}


def bootstrap(items: list[dict], seed: int = 0) -> tuple[float | None, float | None]:
    if not items:
        return None, None
    rng = random.Random(seed)
    xs = sorted(sum(rng.choice(items)["correct"] for _ in items) / len(items) for _ in range(BOOTSTRAP))
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


def confusion(items: list[dict]) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for i in items:
        out.setdefault(i["gold"], Counter())[i["pred"]] += 1
    return out


def evaluate(split: str, linker: str, claims_for, links_for) -> dict:
    """claims_for(ticker, fy) -> extracted claim rows (uid, block, claim_type, quote);
    links_for(ticker, later_year) -> link rows of that year."""
    manifest = json.loads(labeling.MANIFEST.read_text())
    items: list[dict] = []
    pairs = []
    for pair in manifest["pairs"]:
        if pair["split"] != split:
            continue
        t, ly = pair["ticker"], pair["later_year"]
        ey = ly - 1
        gold, mapping, chosen = {}, {}, {}
        for fy in (ey, ly):
            info = pair["years"][str(fy)]
            gold[fy] = labeling.read_labels(Path(info["labels"]))
            _, html = labeling.filing_html(t, fy)
            blocks = {b.number: b for b in labeling.blocks_for_filing(html)}
            mapping[fy] = gold_to_pred(gold[fy], claims_for(t, fy), blocks)
            chosen[fy] = set(info["chosen"])
        later_gold = [g for g in gold[ly] if g.block in chosen[ly]]
        got = score_pair(later_gold, gold[ey], chosen[ey], mapping[ly], mapping[ey], links_for(t, ly))
        for i in got:
            i["pair"] = f"{t} {ey}-{ly}"
        items += got
        pairs.append({"pair": f"{t} {ey}-{ly}", **summarise(got)})
    given = [i for i in items if i["extracted"]]
    return {"split": split, "linker": linker,
            "end_to_end": {**summarise(items), "ci95": bootstrap(items)},
            "given_extracted": {**summarise(given), "ci95": bootstrap(given)},
            "confusion_end_to_end": {k: dict(v) for k, v in confusion(items).items()},
            "confusion_given_extracted": {k: dict(v) for k, v in confusion(given).items()},
            "pairs": pairs, "items": items}


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _matrix(conf: dict[str, dict]) -> list[str]:
    cols = [*STATUSES, "wrong prior", "linked", "missed"]
    cols = [c for c in cols if any(c in row for row in conf.values())]
    lines = ["| Gold \\ predicted | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for g in STATUSES:
        if g in conf:
            lines.append(f"| {g} | " + " | ".join(str(conf[g].get(c, 0)) for c in cols) + " |")
    return lines


def markdown(r: dict) -> str:
    lines = [f"Link eval ({r['split']} split, linker {r['linker']})", "",
             "| View | Items | Correct | Accuracy | 95% CI |", "|---|---|---|---|---|"]
    for name, key in (("Given both claims extracted", "given_extracted"), ("End to end", "end_to_end")):
        v = r[key]
        lo, hi = v["ci95"]
        lines.append(f"| {name} | {v['items']} | {v['correct']} | {_pct(v['accuracy'])} | {_pct(lo)} to {_pct(hi)} |")
    lines += ["", "Confusion, given both claims extracted:", "", *_matrix(r["confusion_given_extracted"]),
              "", "Confusion, end to end:", "", *_matrix(r["confusion_end_to_end"]), "",
              "| Pair | Items | Correct | Accuracy |", "|---|---|---|---|"]
    for p in r["pairs"]:
        lines.append(f"| {p['pair']} | {p['items']} | {p['correct']} | {_pct(p['accuracy'])} |")
    lines += ["", ("\"wrong prior\": linked, but to a different earlier claim. \"linked\": a gold-dropped claim "
              "the linker linked forward. \"missed\": the extractor didn't produce the claim (or its gold prior)."),
              "Dropped items are only as complete as the sampled blocks (see link_eval.py)."]
    return "\n".join(lines) + "\n"
