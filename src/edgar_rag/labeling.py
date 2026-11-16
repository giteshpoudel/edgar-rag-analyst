"""Hand-labelling support for the v3 extraction eval (Phase 2).

The owner labels claims from the raw section text, without seeing the extractor's output.
This module:

1. Splits Items 1A and 7 of a filing into numbered blocks, starting a new block at each
   heading-like line (a risk-factor title or an MD&A subheading).
2. Optionally picks a fixed random sample of blocks to label, so labelling takes hours, not
   days. For a pair of consecutive years, the earlier year's blocks with the same heading are
   included too, so year-to-year links can still be labelled. The extractor is scored only on
   the sampled blocks, against the labels in them.
3. Writes a readable label pack (Markdown) and an empty CSV to fill in.
4. Checks a filled-in CSV: known types, quotes copied verbatim from inside sampled blocks,
   unique ids, and prior-year links that resolve.

The claim types and rules are in evals/v3/LABELING.md.
"""

import csv
import difflib
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from edgar_rag import config
from edgar_rag.chunking import html_to_lines, split_sections
from edgar_rag.sec_client import Filing

ITEMS = ("1A", "7")
HEADING_MAX_CHARS = 300
HEADING_MATCH = 0.8  # difflib ratio for "same heading" across years
# Page furniture repeated on every printed page; not headings, and not text to label.
PAGE_NOISE = re.compile(r"^(table of contents|index|\d{1,3}|page \d+)$", re.IGNORECASE)
LABEL_DIR = Path("evals/v3/labels")

CLAIM_TYPES = {
    "customer_concentration": "Reliance on particular or few customers: a named customer, or a share of revenue",
    "supplier_dependence": "Reliance on particular or few suppliers, foundries, or sole/single sources",
    "export_controls_trade": "Export restrictions, entity lists, tariffs, sanctions, trade actions",
    "geographic_exposure": "Revenue, operations or supply concentrated in a named country or region",
    "segment_change": "Reportable segments, or a change to them (new, renamed, merged, reorganised)",
    "strategic_partnership": "A named partner relationship (investment, joint development, alliance)",
    "acquisition_divestiture": "A named acquisition, merger, divestiture or spin-off, done or pending",
    "capacity_investment": "Fabs, datacenters, capacity expansion, capex plans, government incentives for them",
    "restructuring": "A specific restructuring plan, layoffs or facility closures, with charges or scope",
    "legal_regulatory": "A named investigation, lawsuit, regulator action or regulation that applies",
    "cybersecurity_incident": "A specific security incident that happened (not general cyber risk)",
    "ai_demand": "AI as a driver of demand, revenue, investment or risk for the business",
    "competition": "Named competitors, or a specific competitive development",
}
CHANGE_VALUES = ("new", "changed", "unchanged")
CSV_FIELDS = ["claim_id", "ticker", "fiscal_year", "section", "block", "claim_type", "entities",
              "statement", "quote", "prior_claim_id", "change", "notes"]


def normalize(text: str) -> str:
    """Whitespace and quote-mark differences don't make a quote "not verbatim"."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


# --- Blocks --------------------------------------------------------------------------------


@dataclass
class Block:
    number: int
    item: str
    heading: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join([self.heading, *self.lines]) if self.heading else "\n".join(self.lines)


def _is_heading(line: str, next_line: str | None) -> bool:
    """A short line, not a bullet, followed by a paragraph or a bullet list: how risk-factor
    titles and MD&A subheadings look once the HTML is flattened to lines."""
    if not line or len(line) > HEADING_MAX_CHARS or line.startswith(("•", "-", "(")):
        return False
    return next_line is not None and (len(next_line) > HEADING_MAX_CHARS or next_line.startswith("•"))


def blocks_for_filing(html: str) -> list[Block]:
    """Items 1A and 7 as numbered blocks (numbering runs across both items)."""
    sections = {s.item: s for s in split_sections(html_to_lines(html))}
    blocks: list[Block] = []
    for item in ITEMS:
        if item not in sections:
            continue
        lines = [ln.strip() for ln in sections[item].text.split("\n")
                 if ln.strip() and not PAGE_NOISE.match(ln.strip())]
        current = Block(len(blocks) + 1, item, "")
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if _is_heading(line, nxt) and current.lines:
                blocks.append(current)
                current = Block(len(blocks) + 1, item, line)
            elif _is_heading(line, nxt) and not current.heading:
                current.heading = line
            else:
                current.lines.append(line)
        if current.lines or current.heading:
            blocks.append(current)
    return blocks


def sample_pair(later: list[Block], earlier: list[Block], share: float, seed: int) -> tuple[set[int], set[int]]:
    """Block numbers to label in each year. Later-year blocks are sampled per item (so MD&A
    isn't crowded out by the longer Risk Factors); earlier-year blocks with a matching heading
    in the same item are added, so the same disclosures are labelled in both years."""
    if share >= 1:
        return {b.number for b in later}, {b.number for b in earlier}
    rng = random.Random(seed)
    chosen_later: set[int] = set()
    for item in ITEMS:
        pool = [b for b in later if b.item == item]
        k = max(1, round(share * len(pool))) if pool else 0
        chosen_later |= {b.number for b in rng.sample(pool, k)}
    chosen_earlier: set[int] = set()
    for b in later:
        if b.number not in chosen_later or not b.heading:
            continue
        # The single best-matching earlier heading, if it's close enough.
        scored = [(difflib.SequenceMatcher(None, normalize(b.heading), normalize(e.heading)).ratio(), e.number)
                  for e in earlier if e.item == b.item and e.heading and e.number not in chosen_earlier]
        if scored and max(scored)[0] >= HEADING_MATCH:
            chosen_earlier.add(max(scored)[1])
    return chosen_later, chosen_earlier


# --- Label packs -----------------------------------------------------------------------------


def filing_html(ticker: str, fiscal_year: int) -> tuple[Filing, str]:
    """A filing ingested in Phase 1, by company and fiscal year."""
    raw = config.data_dir() / "raw" / ticker.upper()
    for meta in sorted(raw.glob("10-K_*.json")):
        filing = Filing.model_validate_json(meta.read_text())
        if int(filing.report_date[:4]) == fiscal_year and filing.form == "10-K":
            return filing, meta.with_suffix(".html").read_text(encoding="utf-8")
    raise FileNotFoundError(f"No ingested 10-K for {ticker} FY{fiscal_year}; run `edgar ingest-multi` first")


def write_pack(ticker: str, fiscal_year: int, blocks: list[Block], chosen: set[int], filing: Filing,
               out_dir: Path, show_all: bool = False) -> Path:
    """show_all (the earlier year of a pair): print every block's text, so a claim whose heading
    was reworded can still be found and linked, while only LABEL blocks are labelled in full."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_{fiscal_year}.md"
    words = sum(len(blocks[n - 1].text.split()) for n in chosen)
    lines = [f"# {filing.company} ({ticker}) 10-K, fiscal {fiscal_year}", "",
             f"Filing: {filing.url}", f"Report date {filing.report_date}, accession {filing.accession_number}", "",
             (f"Label only the blocks marked **LABEL** ({len(chosen)} of {len(blocks)} blocks, about "
             f"{words:,} words). The rules are in evals/v3/LABELING.md."), ""]
    if show_all:
        lines += [("This is the earlier year of a pair: (skip) blocks are shown too. Label claims in them only "
                  "when a later-year claim links to them."), ""]
    for b in blocks:
        tag = "**LABEL**" if b.number in chosen else "(skip)"
        lines.append(f"## B{b.number:03d} · Item {b.item} · {tag}")
        if b.heading:
            lines.append(f"### {b.heading}")
        if b.number in chosen or show_all:
            lines += ["", *[f"{ln}\n" for ln in b.lines]]
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def write_template(ticker: str, fiscal_year: int, label_dir: Path = LABEL_DIR) -> Path:
    """An empty CSV with the right columns. Never overwrites labels already there."""
    label_dir.mkdir(parents=True, exist_ok=True)
    path = label_dir / f"{ticker}_{fiscal_year}.csv"
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)
    return path


def make_pair_packs(ticker: str, later_year: int, share: float, seed: int, pack_dir: Path,
                    label_dir: Path = LABEL_DIR) -> dict:
    """Label packs and CSV templates for one company's two consecutive years."""
    later_filing, later_html = filing_html(ticker, later_year)
    earlier_filing, earlier_html = filing_html(ticker, later_year - 1)
    later, earlier = blocks_for_filing(later_html), blocks_for_filing(earlier_html)
    chosen_later, chosen_earlier = sample_pair(later, earlier, share, seed)
    out = {}
    for fy, blocks, chosen, filing in ((later_year, later, chosen_later, later_filing),
                                       (later_year - 1, earlier, chosen_earlier, earlier_filing)):
        pack = write_pack(ticker, fy, blocks, chosen, filing, pack_dir, show_all=fy != later_year)
        template = write_template(ticker, fy, label_dir)
        words = sum(len(blocks[n - 1].text.split()) for n in chosen)
        all_words = sum(len(b.text.split()) for b in blocks)
        out[fy] = {"pack": str(pack), "labels": str(template), "accession": filing.accession_number,
                   "blocks": len(blocks), "chosen": sorted(chosen), "words": words, "all_words": all_words}
    return out


# --- Checking labels -------------------------------------------------------------------------


class LabelRow(BaseModel):
    claim_id: str
    ticker: str
    fiscal_year: int
    section: str
    block: int
    claim_type: str
    entities: str = ""
    statement: str = ""
    quote: str
    prior_claim_id: str = ""
    change: str = ""
    notes: str = ""


def read_labels(path: Path) -> list[LabelRow]:
    with path.open(newline="") as f:
        return [LabelRow.model_validate({k: (v or "").strip() for k, v in row.items() if k})
                for row in csv.DictReader(f) if any((v or "").strip() for v in row.values())]


def check_labels(ticker: str, fiscal_year: int, chosen: set[int], blocks: list[Block],
                 rows: list[LabelRow], prior_rows: list[LabelRow] | None,
                 link_targets: set[str] = frozenset()) -> list[str]:
    """Problems in one filing's labels; an empty list means they're usable.

    prior_rows: given for the later year of a pair (links and change labels are checked).
    link_targets: for the earlier year, ids the later year links to; these may sit outside the
    sampled blocks."""
    problems: list[str] = []
    by_number = {b.number: b for b in blocks}
    seen: set[str] = set()
    prior_ids = {r.claim_id: r for r in prior_rows or []}
    for r in rows:
        where = f"{r.claim_id or '(no id)'}"
        if not r.claim_id or r.claim_id in seen:
            problems.append(f"{where}: missing or duplicate claim_id")
        seen.add(r.claim_id)
        if r.ticker.upper() != ticker or r.fiscal_year != fiscal_year:
            problems.append(f"{where}: ticker/fiscal_year should be {ticker}/{fiscal_year}")
        if r.claim_type not in CLAIM_TYPES:
            problems.append(f"{where}: unknown claim_type {r.claim_type!r}")
        block = by_number.get(r.block)
        if block is None or (r.block not in chosen and r.claim_id not in link_targets):
            problems.append(f"{where}: block B{r.block:03d} is not one of the blocks to label")
        elif block.item != r.section:
            problems.append(f"{where}: block B{r.block:03d} is in Item {block.item}, not {r.section}")
        elif normalize(r.quote) not in normalize(block.text):
            problems.append(f"{where}: quote not found verbatim in block B{r.block:03d}")
        if len(r.quote) > 800:
            problems.append(f"{where}: quote is {len(r.quote)} chars; keep it to the sentences that make the claim")
        if prior_rows is not None:  # the later year of a pair: links and change labels expected
            if r.change not in CHANGE_VALUES:
                problems.append(f"{where}: change must be one of {CHANGE_VALUES}")
            if r.change == "new" and r.prior_claim_id:
                problems.append(f"{where}: a new claim has no prior_claim_id")
            if r.change in ("changed", "unchanged"):
                prior = prior_ids.get(r.prior_claim_id)
                if prior is None:
                    problems.append(f"{where}: prior_claim_id {r.prior_claim_id!r} not in the prior year's labels")
                elif prior.claim_type != r.claim_type:
                    problems.append(f"{where}: linked claims must have the same claim_type")
                elif (normalize(prior.quote) == normalize(r.quote)) != (r.change == "unchanged") and not r.notes:
                    problems.append(f"{where}: labelled {r.change!r} but the quotes are "
                                    f"{'identical' if r.change == 'changed' else 'different'} (check, or explain in notes)")
    if prior_rows is not None:
        links = [r.prior_claim_id for r in rows if r.prior_claim_id]
        for dup in {x for x in links if links.count(x) > 1}:
            problems.append(f"{dup}: linked from more than one later-year claim")
    return problems


# --- Manifest: which filings and blocks are labelled -----------------------------------------

MANIFEST = LABEL_DIR / "manifest.json"
