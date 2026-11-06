"""Claim extraction from 10-K Risk Factors (Item 1A) and MD&A (Item 7) (v3 Phase 2).

    filing HTML -> blocks (labeling.blocks_for_filing) -> windows of whole blocks
                -> deepseek-flash, JSON output -> typed claims -> verbatim quote check
                -> dedupe -> claims with company, year, section, block, quote, chunk, link

The claim types and rules match the hand-labelling guide (evals/v3/LABELING.md), so the
extractor is asked for exactly what the labels measure.

Quote check: every quote must appear verbatim (after whitespace and curly-quote normalisation)
in the block text it came from. Claims that fail are dropped and counted, never kept with a
paraphrased quote, so every stored claim is grounded in the filing.

Paid results are cached by (filing sha256, prompt sha256, model, part): re-running the same
extractor version on the same filing costs nothing. Results are written once per extractor
version and never overwritten.
"""

import asyncio
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from edgar_rag import budget, config, llm
from edgar_rag.labeling import CLAIM_TYPES, Block, blocks_for_filing, filing_html, normalize
from edgar_rag.llm_cache import LlmCache, sha256_json, sha256_text

MODEL = "deepseek-flash"
WINDOW_CHARS = 20_000  # about 5k tokens of filing text per call
MAX_TOKENS = 12_000  # reasoning plus a JSON list of claims; a window rarely needs half of it
CONCURRENCY = 8
# deepseek-flash "thinks" before answering by default, and its hidden reasoning is billed as
# output. On Micron FY2026 that was 6-12k output tokens per window, and 2 of 10 windows ran out of
# tokens before writing any JSON. Off: the model answers directly. Part of the version hash, so
# both settings can be scored on dev and kept side by side.
THINKING = False
CLAIMS_DIR = "v3/claims"  # under DATA_DIR

ClaimType = Literal[tuple(CLAIM_TYPES)]  # type: ignore[valid-type]
EntityKind = Literal["company", "country", "regulation", "regulator", "product", "segment", "other"]


class Entity(BaseModel):
    kind: EntityKind
    name: str = Field(max_length=200)


class ExtractedClaim(BaseModel):
    """What the model returns for one claim."""

    claim_type: ClaimType
    entities: list[Entity] = []
    statement: str = Field(max_length=400)
    quote: str = Field(max_length=1500)
    block: int  # the B number the quote is from


class Claim(BaseModel):
    """A stored claim: the extraction plus where it came from."""

    claim_id: str  # TICKER-FY-NNN, in reading order
    ticker: str
    fiscal_year: int
    section: str  # 1A or 7
    block: int
    claim_type: str
    entities: list[Entity]
    statement: str
    quote: str
    chunk_id: str | None  # the indexed chunk containing the quote, when one does
    accession: str
    filing_url: str
    extractor: str  # prompt sha256 prefix: which extractor version made it


class Extraction(BaseModel):
    """One filing's result, with counts for the eval and the health report."""

    ticker: str
    fiscal_year: int
    extractor: str
    claims: list[Claim]
    windows: int
    returned: int  # claims the model returned
    dropped_quote: int  # failed the verbatim quote check
    dropped_invalid: int  # failed schema validation (unknown type, missing fields)
    dropped_duplicate: int
    truncated_windows: int
    cost_usd: float  # list price of the calls made now (0 when everything came from the cache)
    cached_windows: int


# --- Prompt ------------------------------------------------------------------------------------

SYSTEM = """You extract claims from a company's 10-K for a research database. Read the text and return every claim that meets the rules below, as JSON.

A claim is ONE specific assertion that:
1. fits exactly one of these types:
{types}
2. has at least one concrete anchor: a named entity (company, country, regulation, regulator, product, segment), a number, percentage or date, or a specific event. Generic risk language without an anchor is NOT a claim (e.g. "our markets are highly competitive", "we may be subject to export restrictions").

What is NOT an anchor: generic categories ("export controls", "tariffs", "regulations", "AI", "data centers", "suppliers", "customers", "channel partners"), the company itself or its products in general, and broad regions ("regions where we operate"). Most risk-factor text has no anchor and must be skipped; returning nothing for a block is normal.

Rules:
- Label disclosures of what is: facts, current dependencies, events that happened, named exposures. Skip pure "could"/"may" statements unless they name a specific entity, regulation or event.
- Skip items in a list of factors that could affect the business unless the item itself names a specific country, company, regulation or event.
- Type limits: ai_demand only when AI is stated as a driver of the company's demand, revenue or investment (cryptocurrency mining is not AI). cybersecurity_incident only for a specific incident that happened. legal_regulatory needs a named law, regulation, regulator, agency or case (a product defect or inventory charge is not legal_regulatory). capacity_investment is the company's own fabs, datacenters, manufacturing expansion or capex plans, not purchase commitments. segment_change is about reportable segments only, not product lines or architectures.
- One paragraph can hold several claims; return each separately. If the same assertion repeats in this text, return it once (first occurrence).
- A heading that only announces a topic is not a claim; the specific sentence under it is.
- A bullet list naming several countries or companies for one point is one claim, with all of them as entities.
- When two types fit, pick what the sentence is mainly about. Trade actions against a country are export_controls_trade, not geographic_exposure. A named customer's share of revenue is customer_concentration.
- quote: copy the shortest span (usually one sentence, at most three) that makes the claim, EXACTLY as written, character for character. Never paraphrase, shorten with "...", or join non-adjacent sentences.
- block: the number N of the [BN] block the quote comes from.
- statement: one line in your own words.
- entities: named things as {{"kind": ..., "name": ...}}; kind is one of company, country, regulation, regulator, product, segment, other.

Return JSON: {{"claims": [{{"claim_type": ..., "entities": [...], "statement": ..., "quote": ..., "block": N}}]}}. Return {{"claims": []}} if there are none."""

USER = """Company: {company} ({ticker}), 10-K for fiscal {fiscal_year}, Item {item} ({item_name}).

{text}"""

ITEM_NAMES = {"1A": "Risk Factors", "7": "Management's Discussion and Analysis"}


def system_prompt() -> str:
    types = "\n".join(f"   - {name}: {desc}" for name, desc in CLAIM_TYPES.items())
    return SYSTEM.format(types=types)


def extractor_version() -> str:
    """Hash of everything that decides the output except the filing: prompt, model, window size,
    token limit. Any change to these is a new extractor version, with its own cache entries."""
    return sha256_json({"system": system_prompt(), "user": USER, "model": MODEL, "window": WINDOW_CHARS,
                        "max_tokens": MAX_TOKENS, "schema": ExtractedClaim.model_json_schema(),
                        "thinking": THINKING})


# --- Windows -----------------------------------------------------------------------------------


def windows(blocks: list[Block]) -> list[list[Block]]:
    """Consecutive whole blocks of one item, up to WINDOW_CHARS each, so a claim is never cut in
    half by a window edge. A block longer than the limit gets a window to itself."""
    out: list[list[Block]] = []
    for item in ("1A", "7"):
        current: list[Block] = []
        size = 0
        for b in (b for b in blocks if b.item == item):
            n = len(b.text)
            if current and size + n > WINDOW_CHARS:
                out.append(current)
                current, size = [], 0
            current.append(b)
            size += n
        if current:
            out.append(current)
    return out


def window_text(window: list[Block]) -> str:
    return "\n\n".join(f"[B{b.number}]\n{b.text}" for b in window)


# --- Extraction --------------------------------------------------------------------------------


def _parse(content: str) -> tuple[list[ExtractedClaim], int]:
    """Valid claims and the number of invalid ones in a JSON reply."""
    try:
        raw = json.loads(content).get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return [], 1
    good, bad = [], 0
    for item in raw if isinstance(raw, list) else []:
        try:
            good.append(ExtractedClaim.model_validate(item))
        except ValidationError:
            bad += 1
    return good, bad


async def _extract_window(window: list[Block], meta: dict, filing_hash: str, version: str,
                          cache: LlmCache) -> tuple[dict, float, bool]:
    """The model's reply for one window: from the cache if this version has seen it."""
    part = f"{window[0].item}:{window[0].number}-{window[-1].number}"
    cached = cache.get(filing_hash, version, MODEL, part)
    if cached is not None:
        return cached, 0.0, True
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": USER.format(item=window[0].item, item_name=ITEM_NAMES[window[0].item],
                                                        text=window_text(window), **meta)}]
    with budget.purpose("extract"):
        response = await llm.acomplete(messages, max_tokens=MAX_TOKENS, model=MODEL, span_name="extract",
                                       extra={"response_format": {"type": "json_object"},
                                              "extra_body": {"thinking": {"type": "enabled" if THINKING else "disabled"}}})
    cost, *_ = budget.usage_cost(MODEL, response)
    result = {"content": response.choices[0].message.content or "",
              "truncated": response.choices[0].finish_reason == "length"}
    if not result["truncated"]:  # a cut-off reply is a failure: retry it next run, don't keep it
        cache.put(filing_hash, version, MODEL, part, result, cost_usd=cost)
    return result, cost, False


def _chunk_for(quote: str, chunks: list[dict], item: str) -> str | None:
    q = normalize(quote)
    for c in chunks:
        if c["item"] == item and q in normalize(c["text"]):
            return c["chunk_id"]
    return None


async def extract_filing(ticker: str, fiscal_year: int, cache: LlmCache | None = None) -> Extraction:
    ticker = ticker.upper()
    filing, html = filing_html(ticker, fiscal_year)
    blocks = blocks_for_filing(html)
    by_number = {b.number: b for b in blocks}
    version = extractor_version()
    cache = cache or LlmCache()
    meta = {"company": filing.company, "ticker": ticker, "fiscal_year": fiscal_year}
    wins = windows(blocks)

    sem = asyncio.Semaphore(CONCURRENCY)

    async def run(w):
        async with sem:
            return await _extract_window(w, meta, sha256_text(html), version, cache)

    results = await asyncio.gather(*(run(w) for w in wins))

    chunks = _load_chunks(ticker, filing.accession_number)
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()
    returned = dropped_quote = dropped_invalid = dropped_dup = truncated = 0
    for w, (result, _, _) in zip(wins, results):
        truncated += bool(result["truncated"])
        found, invalid = _parse(result["content"])
        returned += len(found) + invalid
        dropped_invalid += invalid
        in_window = {b.number for b in w}
        for x in found:
            # Trust the quote, not the model's block number: find the block that contains it.
            block = next((by_number[n] for n in [x.block, *sorted(in_window)]
                          if n in in_window and normalize(x.quote) in normalize(by_number[n].text)), None)
            if block is None or len(normalize(x.quote)) < 20:
                dropped_quote += 1
                continue
            key = (x.claim_type, normalize(x.quote))
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            claims.append(Claim(
                claim_id="", ticker=ticker, fiscal_year=fiscal_year, section=block.item, block=block.number,
                claim_type=x.claim_type, entities=x.entities, statement=x.statement, quote=x.quote,
                chunk_id=_chunk_for(x.quote, chunks, block.item), accession=filing.accession_number,
                filing_url=filing.url, extractor=version[:12]))
    claims.sort(key=lambda c: c.block)
    for i, c in enumerate(claims, 1):
        c.claim_id = f"{ticker}-{fiscal_year}-{i:03d}"
    return Extraction(
        ticker=ticker, fiscal_year=fiscal_year, extractor=version[:12], claims=claims, windows=len(wins),
        returned=returned, dropped_quote=dropped_quote, dropped_invalid=dropped_invalid,
        dropped_duplicate=dropped_dup, truncated_windows=truncated,
        cost_usd=sum(cost for _, cost, _ in results), cached_windows=sum(hit for _, _, hit in results))


def _load_chunks(ticker: str, accession: str) -> list[dict]:
    from edgar_rag.pipeline import MULTI_DIR

    for path in (config.data_dir() / MULTI_DIR / ticker).glob(f"*{accession}*.chunks.jsonl"):
        return [json.loads(line) for line in path.read_text().splitlines()]
    return []


def save(extraction: Extraction) -> Path:
    """Write one filing's claims under this extractor version. An existing file for the same
    filing and version is identical (same cached replies), so it's left as it is."""
    out = config.data_dir() / CLAIMS_DIR / extraction.extractor
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{extraction.ticker}_{extraction.fiscal_year}.json"
    if not path.exists():
        path.write_text(extraction.model_dump_json(indent=1))
    return path


def load(ticker: str, fiscal_year: int, extractor: str) -> Extraction:
    path = config.data_dir() / CLAIMS_DIR / extractor / f"{ticker}_{fiscal_year}.json"
    return Extraction.model_validate_json(path.read_text())
