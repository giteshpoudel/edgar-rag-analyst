"""Cache for paid LLM results that should never be paid for twice (v3 claim extraction).

A result is keyed by everything that determines it:

    filing sha256  the exact filing text
    prompt sha256  the exact prompt template, schema and instructions
    model          e.g. deepseek-flash
    window         which slice of the filing (e.g. "1A:3")

Change any of them and it is a different key, so a new prompt is paid for once, and re-running
the same prompt costs nothing. Entries are never overwritten: the first stored result for a key
stays, which keeps a cached run reproducible.

Stored in SQLite at <DATA_DIR>/v3/llm_cache.sqlite (gitignored, like all of data/).
"""

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from edgar_rag import config

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    filing_sha256 TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    model TEXT NOT NULL,
    part TEXT NOT NULL,              -- which slice of the filing
    result_json TEXT NOT NULL,
    cost_usd REAL,
    created_utc TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (filing_sha256, prompt_sha256, model, part)
);
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    """Hash of a JSON-able value with sorted keys, so the same content always hashes the same."""
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


class LlmCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.data_dir() / "v3" / "llm_cache.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, filing_sha256: str, prompt_sha256: str, model: str, window: str):
        """The stored result (decoded JSON), or None on a miss."""
        with _lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT result_json FROM results WHERE filing_sha256=? AND prompt_sha256=? "
                "AND model=? AND part=?",
                (filing_sha256, prompt_sha256, model, window),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, filing_sha256: str, prompt_sha256: str, model: str, window: str, result,
            cost_usd: float | None = None) -> bool:
        """Store a result. Returns False (and keeps the old one) if the key is already stored."""
        with _lock, closing(self._connect()) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO results (filing_sha256, prompt_sha256, model, part, "
                "result_json, cost_usd) VALUES (?, ?, ?, ?, ?, ?)",
                (filing_sha256, prompt_sha256, model, window, json.dumps(result), cost_usd),
            )
        return cur.rowcount == 1

    def stats(self) -> tuple[int, float]:
        """(entries, total cost of the calls that produced them): what a full re-run saves."""
        with _lock, closing(self._connect()) as conn:
            n, cost = conn.execute("SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM results").fetchone()
        return n, cost
