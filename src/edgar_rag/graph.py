"""Entity resolution and graph edges over the claims table (v3 Phase 3).

Nodes: company (the filer, and named customers and suppliers), segment, risk topic, claim.
Edges, each pointing at the claim whose quote supports it and its fiscal year:

    states               filer -> claim               every claim with a verified quote
    faces_risk           filer -> risk_topic:<type>   every Item 1A claim
    has_segment          filer -> segment             segment entities of any claim
    names_customer       filer -> company             company entities of customer_concentration claims
    depends_on_supplier  filer -> company             company entities of supplier_dependence claims

Grounding: an entity edge is made only when the entity's name appears in the supporting quote
(case-insensitive, whole words): as written, without corporate suffixes, or as the canonical
name it resolves to ("Samsung Electronics Co., Ltd." is grounded by "Samsung"). The extractor sometimes adds entities the quote doesn't
name (the filer itself, a customer named elsewhere); those claims stay, but give no edge.

Resolution: names are reduced to a key (lowercase, no punctuation, no parenthetical, no
corporate suffixes such as Inc., Ltd., Holding N.V.), then evals/v3/entity_aliases.csv maps
reviewed aliases ("tsmc") onto canonical keys ("taiwan semiconductor manufacturing").
`merge_candidates` lists likely duplicates the alias file doesn't cover yet, for review.
Segments are scoped to their company: AMD's "Data Center" and NVIDIA's are different nodes.
"""

import csv
import difflib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from edgar_rag.labeling import normalize
from edgar_rag.llm_cache import sha256_json, sha256_text

ALIASES = Path(__file__).resolve().parents[2] / "evals/v3/entity_aliases.csv"  # not cwd-relative
RULES = "v3: grounded entity edges; 1A claims face risk topics; segments scoped by ticker; no /XX state suffix"

SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "nv", "plc",
            "holding", "holdings", "llc", "sa", "ag", "se"}
SEGMENT_SUFFIXES = {"segment", "segments", "business", "reporting", "unit"}
# Placeholders and generic phrases that aren't entities of their own.
GENERIC = re.compile(r"^(customer [a-z]|customers?|suppliers?|distributors?|other|all other|other segment|"
                     r"corporate and other|single operating|single operating segment|single reporting unit|"
                     r"two industry segments|one segment|.*\bend market|.*\bcustomers?|.*\bsuppliers?)$")
RELATION_FOR_TYPE = {"customer_concentration": "names_customer", "supplier_dependence": "depends_on_supplier"}


def key(name: str) -> str:
    s = re.sub(r"/[a-z]{2}$", "", name.lower().strip())  # SEC names carry a state: "QUALCOMM INC/DE"
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    words = s.split()
    while len(words) > 1 and words[-1] in SUFFIXES:
        words.pop()
    return " ".join(words)


def segment_key(name: str) -> str:
    words = key(name).split()
    while len(words) > 1 and words[-1] in SEGMENT_SUFFIXES:
        words.pop()
    return " ".join(words)


def surfaces(name: str) -> set[str]:
    """Strings to look for in a quote: the name, without its parenthetical, and the parenthetical
    acronym ("Taiwan Semiconductor Manufacturing Company (TSMC)" -> both)."""
    out = {normalize(name).lower(), normalize(re.sub(r"\s*\([^)]*\)", "", name)).lower()}
    out |= {normalize(m).lower() for m in re.findall(r"\(([^)]+)\)", name)}
    return {s.strip() for s in out if len(s.strip()) >= 2}


def read_aliases(path: Path = ALIASES) -> dict[str, str]:
    """alias key -> canonical key."""
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {key(r["alias"]): key(r["canonical"]) for r in csv.DictReader(f) if r.get("alias")}


def graph_version(extractor: str, path: Path = ALIASES) -> str:
    aliases = sha256_text(path.read_text()) if path.exists() else ""
    return sha256_json({"extractor": extractor, "aliases": aliases, "rules": RULES})[:12]


def _grounded(quote: str, names: set[str]) -> bool:
    q = normalize(quote).lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", q) for n in names)


def plan_edges(claims: list[sqlite3.Row], filers: dict[str, str], aliases: dict[str, str]):
    """(entities, aliases, edges, skipped) for a set of claims, without writing anything.

    entities: entity_id -> (kind, display name); alias rows: (surface, entity_id);
    edges: (source, relation, target, fiscal_year, claim_uid); skipped: Counter of reasons."""
    names: dict[str, Counter] = defaultdict(Counter)
    kinds: dict[str, str] = {}
    alias_rows: set[tuple[str, str]] = set()
    edges: list[tuple] = []
    skipped: Counter = Counter()

    def node(entity_id: str, kind: str, display: str, surface: set[str]):
        kinds[entity_id] = kind
        names[entity_id][display] += 1
        alias_rows.update((s, entity_id) for s in surface)

    filer_keys = {}
    for ticker, company in filers.items():
        fid = f"company:{aliases.get(key(company), key(company))}"
        filer_keys[ticker] = fid
        node(fid, "company", company, surfaces(company) | {ticker.lower()})

    for c in claims:
        if not c["quote_verified"]:
            skipped["quote not verified"] += 1
            continue
        filer = filer_keys[c["ticker"]]
        fy, uid = c["fiscal_year"], c["uid"]
        node(f"claim:{uid}", "claim", c["claim_id"], set())
        edges.append((filer, "states", f"claim:{uid}", fy, uid))
        if c["section"] == "1A":
            topic = f"risk_topic:{c['claim_type']}"
            node(topic, "risk_topic", c["claim_type"].replace("_", " "), set())
            edges.append((filer, "faces_risk", topic, fy, uid))
        for e in json.loads(c["entities_json"]):
            relation = ("has_segment" if e["kind"] == "segment"
                        else RELATION_FOR_TYPE.get(c["claim_type"]) if e["kind"] == "company" else None)
            if relation is None:
                continue
            k = segment_key(e["name"]) if relation == "has_segment" else key(e["name"])
            k = aliases.get(k, k)
            if not k or GENERIC.match(k):
                skipped["generic name"] += 1
                continue
            target = f"segment:{c['ticker'].lower()}:{k}" if relation == "has_segment" else f"company:{k}"
            if target == filer:
                skipped["the filer itself"] += 1
                continue
            # The name as written, without suffixes, or the canonical name it resolves to.
            surface = surfaces(e["name"]) | {key(e["name"]), k}
            if not _grounded(c["quote"], surface):
                skipped["name not in quote"] += 1
                continue
            node(target, "segment" if relation == "has_segment" else "company", e["name"], surface)
            edges.append((filer, relation, target, fy, uid))
    entities = {i: (kinds[i], names[i].most_common(1)[0][0]) for i in kinds}
    return entities, alias_rows, edges, skipped


def build(conn: sqlite3.Connection, extractor: str, filers: dict[str, str], path: Path = ALIASES) -> dict:
    """Write entities, aliases and edges for one extractor version under its graph version."""
    claims = conn.execute("SELECT * FROM claims WHERE extractor=?", (extractor,)).fetchall()
    graph = graph_version(extractor, path)
    entities, alias_rows, edges, skipped = plan_edges(claims, filers, read_aliases(path))
    with conn:
        conn.executemany("INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?)",
                         [(graph, i, k, n) for i, (k, n) in sorted(entities.items())])
        conn.executemany("INSERT OR IGNORE INTO entity_aliases VALUES (?, ?, ?)",
                         [(graph, a, i) for a, i in sorted(alias_rows)])
        conn.executemany("INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?, ?, ?)",
                         [(graph, *e) for e in edges])
    return {"graph": graph, "edges": Counter(e[1] for e in edges), "entities": Counter(k for k, _ in entities.values()),
            "skipped": skipped}


def ungrounded_edges(conn: sqlite3.Connection, graph: str) -> int:
    """Entity edges where none of the target's surface names appears in the supporting quote.
    Must be 0 (states and faces_risk edges point at the claim itself, so they're exempt)."""
    rows = conn.execute(
        "SELECT e.target, c.quote FROM edges e JOIN claims c ON c.uid = e.claim_uid "
        "WHERE e.graph=? AND e.relation IN ('has_segment', 'names_customer', 'depends_on_supplier')",
        (graph,)).fetchall()
    surface: dict[str, set[str]] = defaultdict(set)
    for alias, entity in conn.execute("SELECT alias, entity_id FROM entity_aliases WHERE graph=?", (graph,)):
        surface[entity].add(alias)
    return sum(not _grounded(r["quote"], surface[r["target"]]) for r in rows)


def merge_candidates(conn: sqlite3.Connection, extractor: str, path: Path = ALIASES,
                     min_ratio: float = 0.75) -> list[tuple[str, str, str]]:
    """Pairs of company keys that look like the same entity and aren't merged yet:
    an acronym of the other, one contained in the other, or a close string match."""
    aliases = read_aliases(path)
    seen: Counter = Counter()
    for (entities_json,) in conn.execute("SELECT entities_json FROM claims WHERE extractor=?", (extractor,)):
        for e in json.loads(entities_json):
            if e["kind"] == "company":
                k = key(e["name"])
                seen[aliases.get(k, k)] += 1
                for acr in re.findall(r"\(([^)]+)\)", e["name"]):
                    seen[aliases.get(key(acr), key(acr))] += 0  # parenthetical acronyms count as names
    keys = sorted(k for k in seen if k and not GENERIC.match(k))
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            initials = lambda s: "".join(w[0] for w in s.split())
            if a.replace(" ", "") == initials(b) or b.replace(" ", "") == initials(a):
                why = "acronym"
            elif re.search(rf"\b{re.escape(a)}\b", b) or re.search(rf"\b{re.escape(b)}\b", a):
                why = "contained"
            elif difflib.SequenceMatcher(None, a, b).ratio() >= min_ratio:
                why = "similar"
            else:
                continue
            out.append((a, b, why))
    return out
