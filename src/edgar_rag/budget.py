"""Cost ledger and hard budget cap (v3 Phase 0).

Every LLM call goes through `reserve()`:

    1. Before the call, reserve its worst case: the prompt at the cache-miss rate plus
       max_tokens at the output rate. If settled spend + open reservations + this worst case
       would cross the cap, raise BudgetExceeded and make no call.
    2. After the call, settle the row to the real cost from the response's token counts.

Why reserve rather than check the spent total? An eval runs 16 questions at once. If each only
checked "spent so far", all 16 could see $9.99 of $10 spent and all go ahead. A reservation
counts against the cap the moment it's made, so the last few cents go to one call, not 16.

The ledger is one SQLite file (default data/ledger.sqlite, gitignored), shared by every process
on this machine. Costs use list (peak) prices, so the cap is conservative: off-peak calls really
cost half. Each row keeps its UTC time, so off-peak savings can be worked out afterwards.

Settings:
    EDGAR_BUDGET_USD   the cap in USD (default 10)
    EDGAR_LEDGER       on (default) or off; the public demo server turns it off
    EDGAR_LEDGER_PATH  where the ledger lives (default <DATA_DIR>/ledger.sqlite)
    EDGAR_RUN_ID       label for this run (default: a fresh id per process)
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from edgar_rag import config

DEFAULT_CAP_USD = 10.0
CHARS_PER_TOKEN_WORST = 3.0  # real text averages ~4 chars per token; 3 over-estimates on purpose
RESERVATION_TTL_SECONDS = 15 * 60  # a crashed process's reservation stops counting after this

_PROCESS_RUN_ID = uuid.uuid4().hex[:12]
_purpose: ContextVar[str | None] = ContextVar("budget_purpose", default=None)
_lock = threading.Lock()  # one writer at a time within a process; SQLite locks across processes


class BudgetExceeded(RuntimeError):
    """The next call could cross EDGAR_BUDGET_USD, so it was not made."""


class UnpricedModel(RuntimeError):
    """A model with no price in config.PRICES: its cost can't be counted against the cap."""


# --- Settings ------------------------------------------------------------------------------


def enabled() -> bool:
    return os.getenv("EDGAR_LEDGER", "on").strip().lower() not in ("off", "0", "false")


def cap_usd() -> float:
    value = os.getenv("EDGAR_BUDGET_USD", "").strip()
    return float(value) if value else DEFAULT_CAP_USD


def ledger_path() -> Path:
    value = os.getenv("EDGAR_LEDGER_PATH", "").strip()
    return Path(value) if value else config.data_dir() / "ledger.sqlite"


def run_id() -> str:
    return os.getenv("EDGAR_RUN_ID", "").strip() or _PROCESS_RUN_ID


@contextmanager
def purpose(name: str):
    """Label the LLM calls made inside this block (e.g. "extract", "link") in the ledger.
    Without one, a call is labelled by its span name ("llm", "judge")."""
    token = _purpose.set(name)
    try:
        yield
    finally:
        _purpose.reset(token)


# --- Storage -------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    created_utc REAL NOT NULL,
    run_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,              -- reserved | settled | failed
    reserved_usd REAL NOT NULL,        -- worst case, counted against the cap until settled
    cost_usd REAL,                     -- actual list-price cost once settled
    prompt_tokens INTEGER,
    cached_prompt_tokens INTEGER,
    completion_tokens INTEGER,
    note TEXT
);
CREATE INDEX IF NOT EXISTS calls_status ON calls(status);
"""


def _connect() -> sqlite3.Connection:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: we issue BEGIN IMMEDIATE ourselves, so check-and-insert is atomic
    # across processes (a second process waits on the lock instead of reading a stale total).
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _committed_usd(conn: sqlite3.Connection, now: float) -> float:
    """Settled spend plus reservations still open (and not expired)."""
    settled = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE status = 'settled'").fetchone()[0]
    open_ = conn.execute(
        "SELECT COALESCE(SUM(reserved_usd), 0) FROM calls WHERE status = 'reserved' AND created_utc > ?",
        (now - RESERVATION_TTL_SECONDS,),
    ).fetchone()[0]
    return settled + open_


# --- DeepSeek peak hours ------------------------------------------------------------------

# From https://api-docs.deepseek.com/quick_start/pricing (checked 2026-11-06): peak is 01:00-04:00
# and 06:00-10:00 UTC, Monday to Friday; every other hour is off-peak at half price. (Chinese
# public holidays are off-peak too; treating them as peak only errs on the safe side.)
PEAK_HOURS_UTC = ((1, 4), (6, 10))


def is_offpeak(ts: float | None = None) -> bool:
    t = time.gmtime(time.time() if ts is None else ts)
    if t.tm_wday >= 5:  # Saturday, Sunday
        return True
    return not any(start <= t.tm_hour < end for start, end in PEAK_HOURS_UTC)


# --- Estimates -----------------------------------------------------------------------------


def worst_case_usd(model: str, messages: list[dict], tools: list[dict] | None, max_tokens: int) -> float:
    """An upper bound on one call's cost, before it is made."""
    prices = config.price(model)
    if prices is None:
        raise UnpricedModel(f"No price for model {model!r}: add it to config.PRICES before using it")
    _, miss, out = prices
    chars = len(json.dumps(messages, default=str)) + len(json.dumps(tools or [], default=str))
    prompt_tokens = chars / CHARS_PER_TOKEN_WORST
    return (prompt_tokens * miss + max_tokens * out) / 1e6


def usage_cost(model: str, response) -> tuple[float, int, int, int]:
    """(list-price cost, prompt tokens, cached prompt tokens, completion tokens) of a response."""
    from edgar_rag.tracing import cost_usd, usage_counts

    prompt, cached, completion = usage_counts(response)
    cost = cost_usd(model, prompt, completion, cached)
    return (sum(cost) if cost else 0.0), prompt, cached, completion


# --- Reserve and settle --------------------------------------------------------------------


@dataclass
class Reservation:
    row_id: int | None  # None when the ledger is off
    model: str
    settled: bool = False

    def settle(self, response) -> float:
        """Replace the reservation with the response's real cost. Returns that cost."""
        cost, prompt, cached, completion = usage_cost(self.model, response)
        if self.row_id is not None:
            with _lock, closing(_connect()) as conn:
                conn.execute(
                    "UPDATE calls SET status='settled', cost_usd=?, prompt_tokens=?, "
                    "cached_prompt_tokens=?, completion_tokens=? WHERE id=?",
                    (cost, prompt, cached, completion, self.row_id),
                )
        self.settled = True
        return cost

    def fail(self, error: BaseException) -> None:
        """The call raised: release the reservation. Failed API calls (timeouts, 429s, 5xx)
        are recorded at $0, since providers don't bill requests that return no completion."""
        if self.row_id is not None:
            with _lock, closing(_connect()) as conn:
                conn.execute("UPDATE calls SET status='failed', cost_usd=0, note=? WHERE id=?",
                             (type(error).__name__, self.row_id))
        self.settled = True


def open_reservation(model: str, messages: list[dict], tools: list[dict] | None, max_tokens: int,
                     purpose_name: str) -> Reservation:
    """Reserve the worst case for one call, or raise BudgetExceeded without reserving."""
    if not enabled():
        return Reservation(row_id=None, model=model)
    worst = worst_case_usd(model, messages, tools, max_tokens)
    cap = cap_usd()
    now = time.time()
    with _lock:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            committed = _committed_usd(conn, now)
            if committed + worst > cap:
                conn.execute("ROLLBACK")
                raise BudgetExceeded(
                    f"Budget cap ${cap:.2f} reached: ${committed:.4f} spent or reserved, and this "
                    f"{model} call could cost up to ${worst:.4f}. Raise EDGAR_BUDGET_USD only "
                    "with the owner's go-ahead."
                )
            cur = conn.execute(
                "INSERT INTO calls (created_utc, run_id, purpose, model, status, reserved_usd) "
                "VALUES (?, ?, ?, ?, 'reserved', ?)",
                (now, run_id(), _purpose.get() or purpose_name, model, worst),
            )
            conn.execute("COMMIT")
            return Reservation(row_id=cur.lastrowid, model=model)
        finally:
            conn.close()


@contextmanager
def reserve(model: str, messages: list[dict], tools: list[dict] | None, max_tokens: int,
            purpose_name: str = "llm"):
    """Wrap one LLM call: reserve before, settle (or release on error) after.

        with budget.reserve(model, messages, tools, max_tokens) as r:
            response = client.chat.completions.create(...)
            r.settle(response)
    """
    reservation = open_reservation(model, messages, tools, max_tokens, purpose_name)
    try:
        yield reservation
    except BaseException as e:
        if not reservation.settled:
            reservation.fail(e)
        raise
    if not reservation.settled:  # the caller forgot to settle: keep the worst case counted
        reservation.fail(RuntimeError("not settled"))


# --- Reporting -----------------------------------------------------------------------------


@dataclass
class Summary:
    cap_usd: float
    spent_usd: float
    reserved_usd: float
    calls: int
    failed: int
    by_purpose: list[tuple[str, int, float]]
    by_model: list[tuple[str, int, float]]
    by_run: list[tuple[str, str, int, float]]  # run id, first call (UTC), calls, cost

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd - self.reserved_usd


def summary(last_runs: int = 10) -> Summary:
    now = time.time()
    with _lock, closing(_connect()) as conn:
        spent = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE status='settled'").fetchone()[0]
        reserved = _committed_usd(conn, now) - spent
        calls = conn.execute("SELECT COUNT(*) FROM calls WHERE status='settled'").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM calls WHERE status='failed'").fetchone()[0]
        group = ("SELECT {col}, COUNT(*), SUM(cost_usd) FROM calls WHERE status='settled' "
                 "GROUP BY {col} ORDER BY SUM(cost_usd) DESC")
        by_purpose = conn.execute(group.format(col="purpose")).fetchall()
        by_model = conn.execute(group.format(col="model")).fetchall()
        by_run = conn.execute(
            "SELECT run_id, datetime(MIN(created_utc), 'unixepoch'), COUNT(*), SUM(cost_usd) "
            "FROM calls WHERE status='settled' GROUP BY run_id ORDER BY MIN(created_utc) DESC LIMIT ?",
            (last_runs,),
        ).fetchall()
    return Summary(cap_usd(), spent, reserved, calls, failed, by_purpose, by_model, by_run)


def add_manual(cost_usd: float, purpose_name: str, note: str, model: str = "manual") -> None:
    """Record spend the ledger couldn't see, such as CI runs on GitHub (from the job summary)."""
    with _lock, closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO calls (created_utc, run_id, purpose, model, status, reserved_usd, cost_usd, note) "
            "VALUES (?, 'manual', ?, ?, 'settled', 0, ?, ?)",
            (time.time(), purpose_name, model, cost_usd, note),
        )
