"""v3 tools over the claims database (v3 Phase 4). Active only with EDGAR_TOOLSET=v3.

    claim_history    one company's claims on a topic, year by year, with each year's change status
    compare_claims   several companies' claims on a topic in one year; "none found" is explicit
    graph_neighbors  entities connected to an entity (suppliers, customers, segments, risk topics)
    find_entity      resolve a name ("TSMC") to graph entities through the alias table

Same rules as the v2 tools: typed Pydantic arguments, results capped (MAX_ITEMS), and every
claim printed with its exact quote and cited through Sources.cite (kind "claim", ref = the
claim's uid, url = the filing). The quote is in the tool text, so the number-grounding guard
can check any figure the answer takes from it.

Versions: the claims of the active extractor (newest extraction_runs row), the links of the
current linker code (linking.linker_version()) and the graph built from the current alias file
(graph.graph_version()). EDGAR_LINKER and EDGAR_GRAPH override the last two.
"""

import json
import os
import sqlite3
from contextlib import closing
from typing import Literal

from pydantic import BaseModel, Field

from edgar_rag import claimstore, graph, linking
from edgar_rag.labeling import CLAIM_TYPES

MAX_ITEMS = 20
QUOTE_CHARS = 500  # longer quotes are shortened in the tool text (the citation keeps the full claim)
PER_COMPANY = 4

Topic = Literal[tuple(CLAIM_TYPES)]  # type: ignore[valid-type]
Relation = Literal["has_segment", "names_customer", "depends_on_supplier", "faces_risk"]


class NoDatabase(RuntimeError):
    pass


# --- Arguments -------------------------------------------------------------------------------


class ClaimHistoryArgs(BaseModel):
    ticker: str = Field(description="Company ticker, e.g. NVDA")
    topic: Topic = Field(description="Claim type: " + "; ".join(f"{k} = {v}" for k, v in CLAIM_TYPES.items()))
    keyword: str | None = Field(None, description="Optional word or name the claim must mention, e.g. China, TSMC")
    from_year: int | None = Field(None, description="First fiscal year (default: earliest in the data)")
    to_year: int | None = Field(None, description="Last fiscal year (default: latest in the data)")


class CompareClaimsArgs(BaseModel):
    tickers: list[str] | None = Field(None, description="Tickers to compare (default: every company)")
    topic: Topic = Field(description="Claim type (see claim_history)")
    fiscal_year: int | None = Field(None, description="Fiscal year (default: each company's latest)")
    keyword: str | None = Field(None, description="Optional word or name the claim must mention")


class GraphNeighborsArgs(BaseModel):
    entity: str = Field(description="A company, supplier, customer or segment name, ticker, or an id from find_entity")
    relation: Relation | None = Field(None, description="Only this relation (default: all)")
    fiscal_year: int | None = Field(None, description="Only edges supported by this fiscal year's filing")


class FindEntityArgs(BaseModel):
    name: str = Field(description="Name to look up, e.g. TSMC, Samsung, Data Center")


# --- Database access -------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    path = claimstore.db_path()
    if not path.exists():
        raise NoDatabase("ERROR: the v3 claims database isn't built on this machine "
                         "(edgar claims-load, edgar link, edgar graph).")
    return claimstore.connect_readonly(path)


def _versions(conn: sqlite3.Connection) -> tuple[str, str, str]:
    extractor = claimstore.active_extractor(conn)
    if extractor is None:
        raise NoDatabase("ERROR: no claims are loaded in the v3 database.")
    linker = os.environ.get("EDGAR_LINKER") or linking.linker_version()
    graph_v = os.environ.get("EDGAR_GRAPH") or graph.graph_version(extractor)
    return extractor, linker, graph_v


def _years(conn: sqlite3.Connection, ticker: str) -> list[int]:
    return [r[0] for r in conn.execute("SELECT fiscal_year FROM filings WHERE ticker=? ORDER BY 1", (ticker,))]


def _matches(row: sqlite3.Row, keyword: str | None) -> bool:
    if not keyword:
        return True
    k = keyword.lower()
    return k in row["quote"].lower() or k in row["statement"].lower() or k in row["entities_json"].lower()


def _quote(text: str) -> str:
    return f'"{text}"' if len(text) <= QUOTE_CHARS else f'"{text[:QUOTE_CHARS].rstrip()}…" (shortened)'


def _cite(row: sqlite3.Row, sources) -> str:
    label = f"{row['ticker']} 10-K FY{row['fiscal_year']}, Item {row['section']}: {row['quote'][:90]}"
    return sources.cite("claim", label, row["uid"], row["filing_url"])


def _run(fn):
    """Turn a missing database into the tool's ERROR text instead of an exception."""

    def wrapped(args, sources) -> str:
        try:
            return fn(args, sources)
        except NoDatabase as e:
            return str(e)

    wrapped.__name__ = fn.__name__
    return wrapped


# --- Tools -----------------------------------------------------------------------------------


@_run
def claim_history(args: ClaimHistoryArgs, sources) -> str:
    ticker = args.ticker.upper()
    with closing(_connect()) as conn:
        extractor, linker, _ = _versions(conn)
        years = _years(conn, ticker)
        if not years:
            return f"ERROR: {ticker} has no filings in the v3 claims database."
        lo, hi = args.from_year or years[0], args.to_year or years[-1]
        span = [y for y in years if lo <= y <= hi]
        if not span:
            return f"ERROR: no {ticker} filings between FY{lo} and FY{hi}; the data covers FY{years[0]}-FY{years[-1]}."
        rows = conn.execute(
            "SELECT * FROM claims WHERE extractor=? AND ticker=? AND claim_type=? AND fiscal_year BETWEEN ? AND ? "
            "ORDER BY fiscal_year, claim_id", (extractor, ticker, args.topic, span[0], span[-1])).fetchall()
        by_uid = {r["uid"]: r for r in rows}
        links = conn.execute(
            "SELECT * FROM claim_links WHERE linker=? AND ticker=? AND fiscal_year BETWEEN ? AND ?",
            (linker, ticker, span[0], span[-1])).fetchall()
        status = {k["claim_uid"]: k for k in links if k["claim_uid"]}
        dropped = {}
        for k in links:
            if k["status"] == "dropped":
                prior = conn.execute("SELECT * FROM claims WHERE uid=?", (k["prior_uid"],)).fetchone()
                if prior is not None and prior["claim_type"] == args.topic:
                    dropped.setdefault(k["fiscal_year"], []).append(prior)

        items: list[tuple[int, str]] = []
        for y in span:
            for r in (r for r in rows if r["fiscal_year"] == y and _matches(r, args.keyword)):
                k = status.get(r["uid"])
                if k is None:
                    line = f"{_cite(r, sources)} first year in the data: {_quote(r['quote'])}"
                elif k["status"] == "new":
                    line = f"{_cite(r, sources)} new this year: {_quote(r['quote'])}"
                else:
                    prior = by_uid.get(k["prior_uid"]) or conn.execute(
                        "SELECT * FROM claims WHERE uid=?", (k["prior_uid"],)).fetchone()
                    line = f"{_cite(r, sources)} {k['status']} from FY{y - 1} {_cite(prior, sources)}: {_quote(r['quote'])}"
                items.append((y, line))
            for p in (p for p in dropped.get(y, []) if _matches(p, args.keyword)):
                items.append((y, (f"dropped: this FY{y - 1} claim isn't in the FY{y} filing {_cite(p, sources)}: "
                                 f"{_quote(p['quote'])}")))

    what = f"{ticker} {args.topic} claims, FY{span[0]}-FY{span[-1]}" + (f', mentioning "{args.keyword}"' if args.keyword else "")
    if not items:
        return f"{what}: none found. The data covers FY{years[0]}-FY{years[-1]}."
    shown = items[:MAX_ITEMS]
    out = [f"{what} ({len(shown)} of {len(items)} shown; statuses compare each year with the year before):"]
    for y in span:
        lines = [line for yy, line in shown if yy == y]
        hidden = sum(yy == y for yy, _ in items[MAX_ITEMS:])
        empty = " none" if not hidden else f" {hidden} not shown"
        out.append(f"FY{y}:" + ("".join(f"\n- {line}" for line in lines) if lines else empty)
                   + (f"\n- ...and {hidden} more not shown" if lines and hidden else ""))
    if len(items) > MAX_ITEMS:
        rest = sorted({y for y, _ in items[MAX_ITEMS:]})
        out.append(f"{len(items) - MAX_ITEMS} more not shown (FY{rest[0]}-FY{rest[-1]}); narrow with keyword or years.")
    return "\n".join(out)


@_run
def compare_claims(args: CompareClaimsArgs, sources) -> str:
    with closing(_connect()) as conn:
        extractor, _, _ = _versions(conn)
        all_tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM filings ORDER BY 1")]
        tickers = [t.upper() for t in args.tickers] if args.tickers else all_tickers
        out, shown = [], 0
        for t in tickers:
            years = _years(conn, t)
            if not years:
                out.append(f"{t}: not in the v3 claims database.")
                continue
            y = args.fiscal_year or years[-1]
            if y not in years:
                out.append(f"{t}: no FY{y} filing in the data (covers FY{years[0]}-FY{years[-1]}).")
                continue
            rows = [r for r in conn.execute(
                "SELECT * FROM claims WHERE extractor=? AND ticker=? AND fiscal_year=? AND claim_type=? ORDER BY claim_id",
                (extractor, t, y, args.topic)) if _matches(r, args.keyword)]
            if not rows:
                out.append(f"{t} FY{y}: none found.")
                continue
            room = max(0, min(PER_COMPANY, MAX_ITEMS - shown))
            lines = [f"- {_cite(r, sources)} {_quote(r['quote'])}" for r in rows[:room]]
            shown += len(lines)
            more = f" ({len(rows) - len(lines)} more not shown)" if len(rows) > len(lines) else ""
            out.append(f"{t} FY{y}: {len(rows)} claim(s){more}" + "".join(f"\n{line}" for line in lines))
    head = f"{args.topic} claims" + (f' mentioning "{args.keyword}"' if args.keyword else "") + \
        (f", FY{args.fiscal_year}" if args.fiscal_year else ", each company's latest fiscal year")
    return head + ":\n" + "\n".join(out)


def _resolve(conn: sqlite3.Connection, graph_v: str, name: str) -> list[sqlite3.Row]:
    """Entities matching a name: id, exact surface name, resolved key, then substring."""
    q = name.strip().lower()
    exact = conn.execute("SELECT * FROM entities WHERE graph=? AND entity_id=? AND kind!='claim'", (graph_v, q)).fetchall()
    if exact:
        return exact
    aliases = graph.read_aliases()
    k = graph.key(name)
    keys = {f"company:{aliases.get(k, k)}", f"company:{k}"}
    filer = conn.execute("SELECT company FROM filings WHERE ticker=? LIMIT 1", (name.strip().upper(),)).fetchone()
    if filer:
        fk = graph.key(filer[0])
        keys.add(f"company:{aliases.get(fk, fk)}")
    rows = conn.execute(
        f"SELECT DISTINCT e.* FROM entities e LEFT JOIN entity_aliases a ON a.graph=e.graph AND a.entity_id=e.entity_id "
        f"WHERE e.graph=? AND e.kind!='claim' AND (a.alias=? OR e.entity_id IN ({','.join('?' * len(keys))}))",
        (graph_v, q, *keys)).fetchall()
    if rows:
        return rows
    like = f"%{q}%"
    return conn.execute(
        "SELECT DISTINCT e.* FROM entities e LEFT JOIN entity_aliases a ON a.graph=e.graph AND a.entity_id=e.entity_id "
        "WHERE e.graph=? AND e.kind!='claim' AND (a.alias LIKE ? OR lower(e.name) LIKE ? OR e.entity_id LIKE ?) "
        "ORDER BY e.kind, e.entity_id LIMIT ?", (graph_v, like, like, like, MAX_ITEMS)).fetchall()


@_run
def find_entity(args: FindEntityArgs, sources) -> str:
    with closing(_connect()) as conn:
        _, _, graph_v = _versions(conn)
        rows = _resolve(conn, graph_v, args.name)[:MAX_ITEMS]
        if not rows:
            return f"No graph entity matches '{args.name}'. Try a shorter name, or search_filings."
        out = [f"Entities matching '{args.name}':"]
        for r in rows:
            n = conn.execute("SELECT COUNT(*) FROM edges WHERE graph=? AND relation!='states' AND (source=? OR target=?)",
                             (graph_v, r["entity_id"], r["entity_id"])).fetchone()[0]
            names = [a[0] for a in conn.execute(
                "SELECT alias FROM entity_aliases WHERE graph=? AND entity_id=? ORDER BY length(alias) LIMIT 5",
                (graph_v, r["entity_id"]))]
            out.append(f"- {r['entity_id']} ({r['kind']}): {r['name']}; {n} edges"
                       + (f"; also written as {', '.join(names)}" if names else ""))
    return "\n".join(out)


@_run
def graph_neighbors(args: GraphNeighborsArgs, sources) -> str:
    with closing(_connect()) as conn:
        _, _, graph_v = _versions(conn)
        found = _resolve(conn, graph_v, args.entity)
        if not found:
            return f"ERROR: no graph entity matches '{args.entity}'. Use find_entity."
        if len(found) > 1 and not any(f["entity_id"] == args.entity.lower() for f in found):
            options = "; ".join(f"{f['entity_id']} ({f['name']})" for f in found[:10])
            return f"ERROR: '{args.entity}' matches several entities: {options}. Call again with one id."
        node = found[0]
        sql = ("SELECT e.*, c.* FROM edges e JOIN claims c ON c.uid = e.claim_uid "
               "WHERE e.graph=? AND e.relation!='states' AND (e.source=? OR e.target=?)")
        params: list = [graph_v, node["entity_id"], node["entity_id"]]
        if args.relation:
            sql += " AND e.relation=?"
            params.append(args.relation)
        if args.fiscal_year:
            sql += " AND e.fiscal_year=?"
            params.append(args.fiscal_year)
        edges = conn.execute(sql + " ORDER BY e.fiscal_year DESC, c.claim_id", params).fetchall()
        names = {r["entity_id"]: r["name"] for r in conn.execute("SELECT entity_id, name FROM entities WHERE graph=?", (graph_v,))}

    groups: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
    for e in edges:
        groups.setdefault((e["source"], e["relation"], e["target"]), []).append(e)
    if not groups:
        what = f" for {args.relation}" if args.relation else ""
        return f"{node['name']} ({node['entity_id']}): no edges{what}" + (f" in FY{args.fiscal_year}." if args.fiscal_year else ".")
    out = [f"{node['name']} ({node['entity_id']}), {len(groups)} connection(s), most recent supporting claim cited:"]
    for (src, rel, tgt), es in list(groups.items())[:MAX_ITEMS]:
        latest = es[0]  # edges are newest first
        years = sorted({e["fiscal_year"] for e in es})
        out.append(f"- {names.get(src, src)} --{rel}--> {names.get(tgt, tgt)}; FY{', FY'.join(map(str, years))} "
                   f"({len(es)} claim(s)) {_cite(latest, sources)}: {_quote(latest['quote'])}")
    if len(groups) > MAX_ITEMS:
        out.append(f"{len(groups) - MAX_ITEMS} more connections not shown; filter by relation or fiscal_year.")
    return "\n".join(out)


def tools() -> list:
    from edgar_rag.tools import Tool

    return [
        Tool(name="claim_history",
             description="One company's claims on a topic across fiscal years, from its 10-K risk factors and MD&A, "
             "each with its exact quote and whether it was new, changed, unchanged or dropped versus the year "
             "before. Use for 'when did X first appear', 'how has X changed', 'did they stop mentioning X'.",
             args_model=ClaimHistoryArgs, fn=claim_history),
        Tool(name="compare_claims",
             description="Several companies' claims on one topic in one fiscal year, each quoted and cited, with "
             "'none found' stated per company. Use for 'which companies mention X' and side-by-side comparisons.",
             args_model=CompareClaimsArgs, fn=compare_claims),
        Tool(name="graph_neighbors",
             description="What an entity is connected to: a company's named suppliers, customers, segments and "
             "risk topics, or which companies name a given supplier or customer. Every connection cites the "
             "claim that supports it.",
             args_model=GraphNeighborsArgs, fn=graph_neighbors),
        Tool(name="find_entity",
             description="Resolve a name (e.g. TSMC, Samsung, Data Center) to graph entities and their ids. "
             "Use before graph_neighbors when a name is ambiguous.",
             args_model=FindEntityArgs, fn=find_entity),
    ]


def describe() -> str:
    """For debugging: the versions the tools read."""
    with closing(_connect()) as conn:
        return json.dumps(dict(zip(("extractor", "linker", "graph"), _versions(conn))))
