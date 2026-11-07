"""Year-to-year claim linking (v3 Phase 3).

For each company and each pair of consecutive fiscal years (Y-1, Y), every year-Y claim is
compared with the year Y-1 claims of the same type and section (1A with 1A, 7 with 7: MD&A
often repeats last year's figure for comparison, which would otherwise steal the 1A link):

    identical normalised quotes            -> link, status unchanged, method "exact"
    cosine >= AUTO_COSINE                  -> link candidate, method "score"
    BORDER_COSINE <= cosine < AUTO_COSINE  -> borderline: deepseek-flash is shown the claim and its
                                              borderline candidates and says which (if any) is the
                                              same claim, method "llm"
    cosine < BORDER_COSINE                 -> not a candidate

Cosine is between local bge-small embeddings of the two quotes; a string similarity (difflib
ratio of the normalised quotes) is stored next to it. Candidates are then matched one-to-one,
exact links first and then by cosine, so one earlier claim never links to two later ones.

    later claim matched, same wording      -> unchanged
    later claim matched, wording differs   -> changed

"Same wording" is checked against the filings, not the two quote spans: each quote must appear
verbatim in the other year's text of that section. The extractor may quote one sentence one year
and two the next; if both sentences are in both filings, nothing changed.
    later claim not matched                -> new
    earlier claim not matched              -> dropped (a row in year Y pointing back to it)

The thresholds were set on the dev pairs (NVDA 2025-26: true links scored 0.78-1.0, wrong
same-type pairs up to 0.85, so the band between goes to the model). The section rule, the
wording rule and the prompt's "updated numbers or definitions" guidance came from the first dev
run's errors (PROJECT_PLAN_v3.md, Phase 3 progress). Everything that decides the
output is hashed into the linker version, which is stored on every link row.
"""

import asyncio
import difflib
import json
from dataclasses import dataclass

import numpy as np

from edgar_rag import budget, embeddings, llm
from edgar_rag.labeling import normalize
from edgar_rag.llm_cache import LlmCache, sha256_json

AUTO_COSINE = 0.90
BORDER_COSINE = 0.72
MAX_BORDER_CANDIDATES = 4
MODEL = "deepseek-flash"
MAX_TOKENS = 300
CONCURRENCY = 8
STATUSES = ("new", "changed", "unchanged", "dropped")

SYSTEM = """You compare claims from two consecutive annual 10-K filings of the same company. Decide which of the earlier-year claims, if any, is the SAME claim as the later-year claim: this year's version of the same disclosure.

It is the same claim when it covers the same subject and the company has updated it: new numbers or percentages, a new fiscal year, a different count of customers, a slightly changed definition of the metric, added or removed detail, or rewording. For example, "two direct customers were 22% and 14% of revenue" is the same claim as last year's "Customers A, B and C were 12%, 11% and 11% of revenue".

It is a different claim when the subject is different: a different specific risk, customer, supplier, regulation, regulator, country, event or metric. A new development on a subject (an investigation reaching a finding, a new rule under the same law) is the same claim only if the earlier claim was about that same matter.

Return JSON: {"same": [candidate numbers that are the same claim]}; an empty list if none is."""

USER = """Claim type: {claim_type}

Later year (FY{later_year}):
{later}

Earlier-year candidates (FY{earlier_year}):
{candidates}"""


def linker_version() -> str:
    return sha256_json({"auto": AUTO_COSINE, "border": BORDER_COSINE, "k": MAX_BORDER_CANDIDATES,
                        "model": MODEL, "max_tokens": MAX_TOKENS, "system": SYSTEM, "user": USER,
                        "embedder": embeddings.MODEL_NAME, "unchanged": "each quote verbatim in the other year's section",
                        "match": "one-to-one, exact first then cosine", "candidates": "same type and section"})[:12]


@dataclass
class C:
    """The fields of a claim the linker needs (works for DB rows and Claim models)."""

    uid: str
    claim_type: str
    quote: str
    section: str = "1A"


@dataclass
class Link:
    claim_uid: str | None
    prior_uid: str | None
    status: str
    method: str
    cosine: float | None = None
    string_sim: float | None = None


_vectors: dict[str, np.ndarray] = {}


def embed(cs: list[C]) -> np.ndarray:
    """Quote embeddings, memoised by quote text (the estimate and the run share them)."""
    todo = sorted({c.quote for c in cs} - _vectors.keys())
    if todo:
        _vectors.update(zip(todo, embeddings.embed_documents(todo)))
    return np.array([_vectors[c.quote] for c in cs]) if cs else np.zeros((0, 384))


def string_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio()


@dataclass
class Candidate:
    later: int
    earlier: int
    cosine: float
    exact: bool


def candidates(earlier: list[C], later: list[C], vec_e: np.ndarray, vec_l: np.ndarray) -> list[Candidate]:
    out = []
    for j, lc in enumerate(later):
        for i, ec in enumerate(earlier):
            if ec.claim_type != lc.claim_type or ec.section != lc.section:
                continue
            exact = normalize(ec.quote) == normalize(lc.quote)
            cos = float(vec_e[i] @ vec_l[j])
            if exact or cos >= BORDER_COSINE:
                out.append(Candidate(j, i, 1.0 if exact else cos, exact))
    return out


async def _judge(lc: C, cands: list[C], later_year: int, cache: LlmCache) -> tuple[list[int], float, bool]:
    """Indexes (into cands) the model calls the same claim, the cost of asking, and whether an
    API call was made (False on a cache hit)."""
    key = sha256_json({"later": lc.quote, "type": lc.claim_type, "cands": [c.quote for c in cands]})
    version = linker_version()
    hit = cache.get(key, version, MODEL, "link")
    if hit is not None:
        return hit["same"], 0.0, False
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(
                    claim_type=lc.claim_type, later_year=later_year, later=lc.quote, earlier_year=later_year - 1,
                    candidates="\n\n".join(f"[{k + 1}] {c.quote}" for k, c in enumerate(cands)))}]
    with budget.purpose("link"):
        response = await llm.acomplete(messages, max_tokens=MAX_TOKENS, model=MODEL, span_name="link",
                                       extra={"response_format": {"type": "json_object"},
                                              "extra_body": {"thinking": {"type": "disabled"}}})
    cost, *_ = budget.usage_cost(MODEL, response)
    try:
        raw = json.loads(response.choices[0].message.content or "{}").get("same", [])
        same = sorted({int(k) - 1 for k in raw if 1 <= int(k) <= len(cands)})
    except (ValueError, TypeError, AttributeError):
        return [], cost, True  # an unreadable reply is "no match" and isn't cached, so a re-run asks again
    cache.put(key, version, MODEL, "link", {"same": same}, cost_usd=cost)
    return same, cost, True


def borderline_groups(cands: list[Candidate]) -> dict[int, list[Candidate]]:
    """Later claims with no exact or auto candidate, and their best borderline candidates."""
    strong = {c.later for c in cands if c.exact or c.cosine >= AUTO_COSINE}
    groups: dict[int, list[Candidate]] = {}
    for c in cands:
        if c.later not in strong:
            groups.setdefault(c.later, []).append(c)
    return {j: sorted(g, key=lambda c: -c.cosine)[:MAX_BORDER_CANDIDATES] for j, g in groups.items()}


def same_wording(ec: C, lc: C, earlier_text: dict[str, str] | None, later_text: dict[str, str] | None) -> bool:
    """Identical normalised quotes, or each quote verbatim in the other year's section text
    (texts are normalised, by section). Without texts, only identical quotes count."""
    a, b = normalize(ec.quote), normalize(lc.quote)
    if a == b:
        return True
    if earlier_text is None or later_text is None:
        return False
    return b in earlier_text.get(lc.section, "") and a in later_text.get(ec.section, "")


async def link_years(earlier: list[C], later: list[C], later_year: int, cache: LlmCache | None = None,
                     judge=None, earlier_text: dict[str, str] | None = None,
                     later_text: dict[str, str] | None = None) -> tuple[list[Link], float, int]:
    """Links for one company's (Y-1, Y). Returns (links, cost, API calls made; cache hits are free).
    earlier_text / later_text: each filing's normalised text by section, for the wording check.
    `judge` replaces the model call in tests: judge(later C, candidate Cs) -> indexes that are
    the same claim."""
    if not later and not earlier:
        return [], 0.0, 0
    cands = candidates(earlier, later, embed(earlier), embed(later))
    accepted = [(c, "exact" if c.exact else "score") for c in cands if c.exact or c.cosine >= AUTO_COSINE]

    groups = borderline_groups(cands)
    cost, calls = 0.0, 0
    if groups:
        cache = cache or LlmCache()
        sem = asyncio.Semaphore(CONCURRENCY)

        async def ask(j, group):
            async with sem:
                cs = [earlier[c.earlier] for c in group]
                if judge is not None:
                    return j, group, judge(later[j], cs), 0.0, False
                same, c, called = await _judge(later[j], cs, later_year, cache)
                return j, group, same, c, called

        for j, group, same, c, called in await asyncio.gather(*(ask(j, g) for j, g in groups.items())):
            cost += c
            calls += called
            accepted += [(group[k], "llm") for k in same]

    # One-to-one: exact links first, then highest cosine.
    accepted.sort(key=lambda x: (not x[0].exact, -x[0].cosine))
    used_l: set[int] = set()
    used_e: set[int] = set()
    links: list[Link] = []
    for c, method in accepted:
        if c.later in used_l or c.earlier in used_e:
            continue
        used_l.add(c.later)
        used_e.add(c.earlier)
        lc, ec = later[c.later], earlier[c.earlier]
        status = "unchanged" if same_wording(ec, lc, earlier_text, later_text) else "changed"
        links.append(Link(lc.uid, ec.uid, status, method,
                          round(c.cosine, 4), round(string_sim(ec.quote, lc.quote), 4)))
    links += [Link(lc.uid, None, "new", "none") for j, lc in enumerate(later) if j not in used_l]
    links += [Link(None, ec.uid, "dropped", "none") for i, ec in enumerate(earlier) if i not in used_e]
    return links, cost, calls


def estimate(earlier: list[C], later: list[C]) -> int:
    """Model calls a link_years run would make before the cache (for the cost estimate)."""
    if not later or not earlier:
        return 0
    return len(borderline_groups(candidates(earlier, later, embed(earlier), embed(later))))


def estimate_call_usd(prompt_tokens: int = 900, out_tokens: int = 40) -> float:
    """List price of one borderline call: system prompt plus up to five quotes in, a short list out."""
    from edgar_rag import config

    _, miss, out = config.price(MODEL)
    return (prompt_tokens * miss + out_tokens * out) / 1e6
