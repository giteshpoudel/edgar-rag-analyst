"""The v3 claims database: plain SQLite, append-only (v3 Phase 3).

    filings          one row per 10-K (ticker, fiscal year, accession, link)
    extraction_runs  which extractor version was loaded, when; the newest row is the active one
    claims           every extracted claim with its exact quote, keyed by extractor version
    claim_links      year-to-year links (new / changed / unchanged / dropped), keyed by linker version
    entities         resolved graph nodes (company, segment, customer, supplier, risk topic)
    entity_aliases   surface names that resolve to an entity
    edges            company -> entity relations, each pointing at the claim whose quote supports it

Append-only: triggers reject UPDATE and DELETE on every table, so a stored claim or link can
never change. A new extractor or linker version adds rows under its own version id, and the
old rows stay readable next to it. Re-loading the same version is a no-op (INSERT OR IGNORE).
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from edgar_rag import config

DB_NAME = "v3/claims.sqlite"  # under DATA_DIR

TABLES = ("filings", "extraction_runs", "claims", "claim_links", "entities", "entity_aliases", "edges")

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    ticker TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    accession TEXT NOT NULL,
    company TEXT NOT NULL,
    filing_url TEXT NOT NULL,
    PRIMARY KEY (ticker, fiscal_year)
);
CREATE TABLE IF NOT EXISTS extraction_runs (
    run INTEGER PRIMARY KEY AUTOINCREMENT,
    extractor TEXT NOT NULL,
    loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    filings INTEGER NOT NULL,
    claims INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    uid TEXT PRIMARY KEY,               -- extractor:claim_id
    extractor TEXT NOT NULL,
    claim_id TEXT NOT NULL,             -- TICKER-FY-NNN
    ticker TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    section TEXT NOT NULL,
    block INTEGER NOT NULL,
    claim_type TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    statement TEXT NOT NULL,
    quote TEXT NOT NULL,
    quote_verified INTEGER NOT NULL,    -- 1 if the quote was found verbatim in the filing block at load
    chunk_id TEXT,
    accession TEXT NOT NULL,
    filing_url TEXT NOT NULL,
    FOREIGN KEY (ticker, fiscal_year) REFERENCES filings (ticker, fiscal_year)
);
CREATE INDEX IF NOT EXISTS claims_by_filing ON claims (extractor, ticker, fiscal_year, claim_type);
CREATE TABLE IF NOT EXISTS claim_links (
    linker TEXT NOT NULL,
    ticker TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,       -- the later year
    claim_uid TEXT,                     -- the later-year claim; NULL for dropped
    prior_uid TEXT,                     -- the earlier-year claim; NULL for new
    status TEXT NOT NULL CHECK (status IN ('new', 'changed', 'unchanged', 'dropped')),
    method TEXT NOT NULL,               -- exact, score, llm, none
    cosine REAL,
    string_sim REAL,
    score REAL,
    UNIQUE (linker, claim_uid, prior_uid)
);
CREATE INDEX IF NOT EXISTS links_by_claim ON claim_links (linker, claim_uid);
CREATE INDEX IF NOT EXISTS links_by_prior ON claim_links (linker, prior_uid);
CREATE TABLE IF NOT EXISTS entities (
    graph TEXT NOT NULL,                -- graph version (see edges)
    entity_id TEXT NOT NULL,            -- kind:key, e.g. company:taiwan semiconductor manufacturing
    kind TEXT NOT NULL,                 -- company, segment, risk_topic, claim
    name TEXT NOT NULL,                 -- display name
    PRIMARY KEY (graph, entity_id)
);
CREATE TABLE IF NOT EXISTS entity_aliases (
    graph TEXT NOT NULL,
    alias TEXT NOT NULL,                -- lowercase surface name as written in filings, e.g. tsmc
    entity_id TEXT NOT NULL,
    PRIMARY KEY (graph, alias, entity_id)
);
CREATE TABLE IF NOT EXISTS edges (
    graph TEXT NOT NULL,                -- graph version: extractor plus alias-file hash
    source TEXT NOT NULL,               -- entity_id of the company
    relation TEXT NOT NULL CHECK (relation IN
        ('has_segment', 'names_customer', 'depends_on_supplier', 'faces_risk', 'states')),
    target TEXT NOT NULL,               -- entity_id, or claim:uid for states
    fiscal_year INTEGER NOT NULL,
    claim_uid TEXT NOT NULL REFERENCES claims (uid),
    UNIQUE (graph, source, relation, target, claim_uid)
);
CREATE INDEX IF NOT EXISTS edges_by_target ON edges (graph, target);
CREATE INDEX IF NOT EXISTS edges_by_source ON edges (graph, source);
"""


def _triggers() -> str:
    out = []
    for table in TABLES:
        for op in ("UPDATE", "DELETE"):
            out.append(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
                       f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;")
    return "\n".join(out)


def db_path() -> Path:
    return config.data_dir() / DB_NAME


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA + _triggers())
    return conn


def active_extractor(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT extractor FROM extraction_runs ORDER BY run DESC LIMIT 1").fetchone()
    return row["extractor"] if row else None


def load_extractions(conn: sqlite3.Connection, extractions, verify, company_names: dict[str, str]) -> tuple[int, int]:
    """Insert filings and claims, and mark this extractor version active. verify(claim) -> bool
    re-checks the quote against the filing text. Returns (claims inserted, claims unverified)."""
    inserted = unverified = 0
    extractors = {x.extractor for x in extractions}
    if len(extractors) != 1:
        raise ValueError(f"load one extractor version at a time, got {sorted(extractors)}")
    extractor = extractors.pop()
    with conn:
        for x in extractions:
            for c in x.claims[:1]:
                conn.execute("INSERT OR IGNORE INTO filings VALUES (?, ?, ?, ?, ?)",
                             (x.ticker, x.fiscal_year, c.accession, company_names.get(x.ticker, x.ticker),
                              c.filing_url))
            for c in x.claims:
                ok = bool(verify(c))
                unverified += not ok
                cur = conn.execute(
                    "INSERT OR IGNORE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"{c.extractor}:{c.claim_id}", c.extractor, c.claim_id, c.ticker, c.fiscal_year, c.section,
                     c.block, c.claim_type, _entities_json(c), c.statement, c.quote, int(ok), c.chunk_id,
                     c.accession, c.filing_url))
                inserted += cur.rowcount
        if inserted or active_extractor(conn) != extractor:
            conn.execute("INSERT INTO extraction_runs (extractor, filings, claims) VALUES (?, ?, ?)",
                         (extractor, len(extractions), sum(len(x.claims) for x in extractions)))
    return inserted, unverified


def _entities_json(c) -> str:
    import json

    return json.dumps([e.model_dump() for e in c.entities])


def claims_for(conn: sqlite3.Connection, extractor: str, ticker: str | None = None,
               fiscal_year: int | None = None) -> list[sqlite3.Row]:
    sql, args = "SELECT * FROM claims WHERE extractor=?", [extractor]
    if ticker:
        sql += " AND ticker=?"
        args.append(ticker)
    if fiscal_year:
        sql += " AND fiscal_year=?"
        args.append(fiscal_year)
    return conn.execute(sql + " ORDER BY ticker, fiscal_year, claim_id", args).fetchall()


def unverified_edges(conn: sqlite3.Connection, graph: str) -> int:
    """Edges whose supporting claim is missing or whose quote wasn't verified verbatim. Must be 0."""
    return conn.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN claims c ON c.uid = e.claim_uid "
        "WHERE e.graph=? AND (c.uid IS NULL OR c.quote_verified != 1 OR c.quote = '')", (graph,)).fetchone()[0]


def connect_readonly(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path or db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    with closing(conn.cursor()) as cur:
        return {t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def has_links(conn: sqlite3.Connection, linker: str, ticker: str, fiscal_year: int) -> bool:
    return conn.execute("SELECT 1 FROM claim_links WHERE linker=? AND ticker=? AND fiscal_year=? LIMIT 1",
                        (linker, ticker, fiscal_year)).fetchone() is not None


def insert_links(conn: sqlite3.Connection, linker: str, ticker: str, fiscal_year: int, links) -> int:
    """Store one company-year's links. A year already linked by this linker version is left as
    it is (NULLs don't collide in UNIQUE, so re-inserting would duplicate new/dropped rows)."""
    if has_links(conn, linker, ticker, fiscal_year):
        return 0
    with conn:
        conn.executemany(
            "INSERT INTO claim_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(linker, ticker, fiscal_year, k.claim_uid, k.prior_uid, k.status, k.method, k.cosine, k.string_sim,
              k.cosine) for k in links])
    return len(links)


def links_for(conn: sqlite3.Connection, linker: str, ticker: str | None = None) -> list[sqlite3.Row]:
    sql, args = "SELECT * FROM claim_links WHERE linker=?", [linker]
    if ticker:
        sql += " AND ticker=?"
        args.append(ticker)
    return conn.execute(sql, args).fetchall()
