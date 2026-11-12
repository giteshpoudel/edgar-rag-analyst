"""v3 claim and graph tools (v3 Phase 4). Offline: a small fixture database built with the real
store, linker (fake embedder and judge) and graph code."""

import asyncio
import json

import numpy as np
import pytest

from edgar_rag import agent, claim_tools, claimstore, graph, linking, tools
from edgar_rag.claims import Claim, Entity, Extraction
from edgar_rag.tools import Sources

TSMC = "Taiwan Semiconductor Manufacturing Company (TSMC)"


def claim(t, fy, n, quote, ctype="supplier_dependence", section="1A", entities=()):
    return Claim(claim_id=f"{t}-{fy}-{n:03d}", ticker=t, fiscal_year=fy, section=section, block=n, claim_type=ctype,
                 entities=[Entity(kind=k, name=v) for k, v in entities], statement="s", quote=quote, chunk_id=None,
                 accession=f"{t}-{fy}", filing_url=f"https://sec.example/{t}/{fy}", extractor="vfix")


def extraction(t, fy, cs):
    return Extraction(ticker=t, fiscal_year=fy, extractor="vfix", claims=cs, windows=1, returned=len(cs),
                      dropped_quote=0, dropped_invalid=0, dropped_duplicate=0, truncated_windows=0, cost_usd=0,
                      cached_windows=0)


FILINGS = {
    ("AAA", 2024): [claim("AAA", 2024, 1, "We rely on TSMC for all of our wafers.", entities=[("company", TSMC)]),
                    claim("AAA", 2024, 2, "Export rules restrict our sales to China of our X100 chips.",
                          "export_controls_trade", entities=[("country", "China")]),
                    claim("AAA", 2024, 3, "Our Russia operations closed after sanctions.", "export_controls_trade")],
    ("AAA", 2025): [claim("AAA", 2025, 1, "We rely on TSMC for all of our wafers.", entities=[("company", TSMC)]),
                    claim("AAA", 2025, 2, "Export rules restrict our sales to China of our X100 and X200 chips.",
                          "export_controls_trade", entities=[("country", "China")]),
                    claim("AAA", 2025, 3, "New tariffs on imports from Mexico raise our costs.", "export_controls_trade")]
                   + [claim("AAA", 2025, 10 + i, f"Rule {i} limits exports of product line {i} to China.",
                            "export_controls_trade") for i in range(25)],
    ("BBB", 2025): [claim("BBB", 2025, 1, "Substantially all our chips are made by TSMC.", entities=[("company", "TSMC")]),
                    claim("BBB", 2025, 2, "Our Data Center segment grew the most.", "segment_change",
                          entities=[("segment", "Data Center")])],
    ("CCC", 2025): [claim("CCC", 2025, 1, "Our Data Center business is our largest.", "segment_change",
                          entities=[("segment", "Data Center")])],
}
NAMES = {"AAA": "Alpha Corp", "BBB": "Beta Inc", "CCC": "Gamma Ltd"}


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "claims.sqlite"
    conn = claimstore.connect(path)
    claimstore.load_extractions(conn, [extraction(t, fy, cs) for (t, fy), cs in FILINGS.items()],
                                verify=lambda c: True, company_names=NAMES)
    # A shared vocabulary for both years, so the two China sentences land close together.
    vocab: dict[str, int] = {}

    def embed(cs):
        rows = []
        for c in cs:
            v = np.zeros(512)
            for w in c.quote.lower().replace(".", "").split():
                v[vocab.setdefault(w, len(vocab))] += 1
            rows.append(v / np.linalg.norm(v))
        return np.array(rows) if rows else np.zeros((0, 512))

    monkeypatch.setattr(linking, "embed", embed)
    rows = claimstore.claims_for(conn, "vfix", "AAA")
    to_c = lambda fy: [linking.C(r["uid"], r["claim_type"], r["quote"], r["section"])
                       for r in rows if r["fiscal_year"] == fy]
    links, _, _ = asyncio.run(linking.link_years(to_c(2024), to_c(2025), 2025, judge=lambda lc, cs: [0]))
    claimstore.insert_links(conn, linking.linker_version(), "AAA", 2025, links)
    graph.build(conn, "vfix", NAMES)
    conn.close()
    monkeypatch.setattr(claimstore, "db_path", lambda: path)
    return path


def run(fn, args_model, **kw):
    sources = Sources()
    return fn(args_model(**kw), sources), sources


# --- claim_history ---------------------------------------------------------------------------


def test_history_statuses_and_citations(db):
    text, sources = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="aaa",
                        topic="supplier_dependence")
    assert "FY2024:\n- [1] first year in the data" in text
    assert "unchanged from FY2024 [1]" in text
    assert all(s.kind == "claim" and s.url.startswith("https://sec.example/AAA/") for s in sources.items)
    assert {s.ref for s in sources.items} == {"vfix:AAA-2024-001", "vfix:AAA-2025-001"}


def test_history_changed_new_dropped_and_keyword(db):
    text, _ = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="AAA",
                  topic="export_controls_trade", keyword="x100")
    assert "changed from FY2024" in text and "X200" in text
    assert "Russia" not in text and "Mexico" not in text  # keyword filter
    text, _ = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="AAA",
                  topic="export_controls_trade", keyword="russia")
    assert "dropped: this FY2024 claim isn't in the FY2025 filing" in text
    text, _ = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="AAA",
                  topic="export_controls_trade", keyword="mexico")
    assert "new this year" in text


def test_history_is_capped_and_says_what_it_left_out(db):
    text, _sources = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="AAA",
                        topic="export_controls_trade")
    assert f"({claim_tools.MAX_ITEMS} of " in text and "more not shown" in text
    assert text.count("\n- ") <= claim_tools.MAX_ITEMS + 1  # plus the "...and N more" line


def test_history_errors_and_empty(db):
    text, _ = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="ZZZ", topic="competition")
    assert text.startswith("ERROR") and "ZZZ" in text
    text, _ = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="AAA", topic="competition",
                  from_year=2010, to_year=2012)
    assert text.startswith("ERROR") and "FY2024-FY2025" in text
    text, sources = run(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, ticker="AAA", topic="competition")
    assert "none found" in text and not sources.items


# --- compare_claims --------------------------------------------------------------------------


def test_compare_says_none_found_per_company(db):
    text, sources = run(claim_tools.compare_claims, claim_tools.CompareClaimsArgs, topic="supplier_dependence",
                        keyword="tsmc", tickers=["AAA", "BBB", "CCC", "ZZZ"])
    assert "AAA FY2025: 1 claim(s)" in text and "BBB FY2025: 1 claim(s)" in text
    assert "CCC FY2025: none found." in text
    assert "ZZZ: not in the v3 claims database." in text
    assert len(sources.items) == 2


def test_compare_caps_each_company(db):
    text, _ = run(claim_tools.compare_claims, claim_tools.CompareClaimsArgs, topic="export_controls_trade",
                  tickers=["AAA"], fiscal_year=2025)
    assert text.count("\n- ") == claim_tools.PER_COMPANY and "more not shown" in text
    text, _ = run(claim_tools.compare_claims, claim_tools.CompareClaimsArgs, topic="export_controls_trade",
                  tickers=["BBB"], fiscal_year=2019)
    assert "no FY2019 filing" in text


# --- graph tools -----------------------------------------------------------------------------


def test_find_entity_resolves_aliases(db):
    text, _ = run(claim_tools.find_entity, claim_tools.FindEntityArgs, name="tsmc")
    assert "company:taiwan semiconductor manufacturing" in text
    text, _ = run(claim_tools.find_entity, claim_tools.FindEntityArgs, name="Nonexistent Widgets")
    assert text.startswith("No graph entity")


def test_neighbors_of_a_supplier_cite_each_edge(db):
    text, sources = run(claim_tools.graph_neighbors, claim_tools.GraphNeighborsArgs, entity="TSMC")
    assert "Alpha Corp --depends_on_supplier--> " in text and "FY2024, FY2025 (2 claim(s))" in text
    assert "Beta Inc --depends_on_supplier--> " in text
    assert len(sources.items) == 2 and all(s.kind == "claim" for s in sources.items)


def test_neighbors_of_a_filer_filters_and_ambiguity(db):
    text, _ = run(claim_tools.graph_neighbors, claim_tools.GraphNeighborsArgs, entity="AAA",
                  relation="depends_on_supplier")
    assert "--depends_on_supplier-->" in text and "faces_risk" not in text
    text, _ = run(claim_tools.graph_neighbors, claim_tools.GraphNeighborsArgs, entity="AAA", relation="faces_risk",
                  fiscal_year=2024)
    assert "export controls trade" in text and "FY2025" not in text
    text, _ = run(claim_tools.graph_neighbors, claim_tools.GraphNeighborsArgs, entity="Data Center")
    assert text.startswith("ERROR") and "segment:bbb:data center" in text and "segment:ccc:data center" in text
    text, _ = run(claim_tools.graph_neighbors, claim_tools.GraphNeighborsArgs, entity="segment:ccc:data center")
    assert "Gamma Ltd --has_segment-->" in text


# --- Missing database and toolset switch ------------------------------------------------------


def test_missing_database_is_an_error_message(tmp_path, monkeypatch):
    monkeypatch.setattr(claimstore, "db_path", lambda: tmp_path / "none.sqlite")
    for fn, model, kw in [(claim_tools.claim_history, claim_tools.ClaimHistoryArgs, {"ticker": "A", "topic": "competition"}),
                          (claim_tools.compare_claims, claim_tools.CompareClaimsArgs, {"topic": "competition"}),
                          (claim_tools.graph_neighbors, claim_tools.GraphNeighborsArgs, {"entity": "x"}),
                          (claim_tools.find_entity, claim_tools.FindEntityArgs, {"name": "x"})]:
        text, _ = run(fn, model, **kw)
        assert text.startswith("ERROR") and "isn't built" in text


def test_toolset_switch(monkeypatch):
    monkeypatch.delenv("EDGAR_TOOLSET", raising=False)
    assert agent.active_tools() is tools.TOOLS
    monkeypatch.setenv("EDGAR_TOOLSET", "v3")
    names = set(agent.active_tools())
    assert names == set(tools.TOOLS) | {"claim_history", "compare_claims", "graph_neighbors", "find_entity"}
    schema = json.dumps(claim_tools.ClaimHistoryArgs.model_json_schema())  # what the LLM sees
    assert all(f'"{t}"' in schema for t in claim_tools.CLAIM_TYPES)
    monkeypatch.setenv("EDGAR_TOOLSET", "v4")
    with pytest.raises(RuntimeError, match="EDGAR_TOOLSET"):
        agent.active_tools()


def test_v3_tool_runs_through_the_agent(db, monkeypatch):
    monkeypatch.setenv("EDGAR_TOOLSET", "v3")
    sources = Sources()
    _, text, error = agent.run_tool("graph_neighbors", '{"entity": "TSMC"}', sources)
    assert not error and "[1]" in text and sources.items[0].kind == "claim"
    _, text, error = agent.run_tool("claim_history", '{"ticker": "AAA", "topic": "not_a_type"}', sources)
    assert error and "invalid arguments" in text
