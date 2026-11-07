"""Claims database, year-to-year linking, graph and link eval (v3 Phase 3). Offline: a fake
embedder, a fake judge and an in-memory set of claims."""

import asyncio
import json
import sqlite3

import numpy as np
import pytest

from edgar_rag import claimstore, graph, link_eval, linking
from edgar_rag.claims import Claim, Entity, Extraction
from edgar_rag.labeling import LabelRow

# --- Fixtures --------------------------------------------------------------------------------


def claim(n: int, fy: int, quote: str, claim_type="export_controls_trade", section="1A", block=1,
          entities=()) -> Claim:
    return Claim(claim_id=f"ACME-{fy}-{n:03d}", ticker="ACME", fiscal_year=fy, section=section, block=block,
                 claim_type=claim_type, entities=[Entity(kind=k, name=v) for k, v in entities],
                 statement="s", quote=quote, chunk_id=None, accession=f"000-{fy}", filing_url=f"https://x/{fy}",
                 extractor="vtest")


def extraction(fy: int, cs: list[Claim]) -> Extraction:
    return Extraction(ticker="ACME", fiscal_year=fy, extractor="vtest", claims=cs, windows=1, returned=len(cs),
                      dropped_quote=0, dropped_invalid=0, dropped_duplicate=0, truncated_windows=0, cost_usd=0,
                      cached_windows=0)


@pytest.fixture
def db(tmp_path):
    conn = claimstore.connect(tmp_path / "claims.sqlite")
    xs = [extraction(2024, [claim(1, 2024, "Sales to Customer A were 12% of revenue.", "customer_concentration"),
                            claim(2, 2024, "We rely on TSMC to manufacture our chips.", "supplier_dependence",
                                  entities=[("company", "Taiwan Semiconductor Manufacturing Company (TSMC)")])]),
          extraction(2025, [claim(1, 2025, "Sales to Customer A were 22% of revenue.", "customer_concentration"),
                            claim(2, 2025, "Our Graphics segment depends on Samsung for memory.",
                                  "supplier_dependence", section="7",
                                  entities=[("company", "Samsung Electronics Co., Ltd."), ("segment", "Graphics"),
                                            ("company", "Micron"), ("company", "ACME Corp")])])]
    claimstore.load_extractions(conn, xs, verify=lambda c: True, company_names={"ACME": "ACME Corp"})
    return conn


@pytest.fixture
def fake_embed(monkeypatch):
    """Word-count vectors: identical quotes score 1, unrelated ones near 0."""
    vocab: dict[str, int] = {}

    def embed(cs):
        rows = []
        for c in cs:
            v = np.zeros(256)
            for w in c.quote.lower().replace(".", "").split():
                v[vocab.setdefault(w, len(vocab)) % 256] += 1
            rows.append(v / (np.linalg.norm(v) or 1))
        return np.array(rows) if rows else np.zeros((0, 256))

    monkeypatch.setattr(linking, "embed", embed)


# --- Append-only store -----------------------------------------------------------------------


@pytest.mark.parametrize("table", claimstore.TABLES)
def test_every_table_rejects_update_and_delete(db, table):
    if table in ("claim_links", "entities", "entity_aliases", "edges"):
        graph.build(db, "vtest", {"ACME": "ACME Corp"}, path=claimstore.Path("/nonexistent.csv"))
        claimstore.insert_links(db, "ltest", "ACME", 2025, [linking.Link("vtest:ACME-2025-001", None, "new", "none")])
    assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
    for sql in (f"UPDATE {table} SET rowid = rowid", f"DELETE FROM {table}"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(sql)


def test_reloading_the_same_version_adds_nothing(db):
    before = claimstore.table_counts(db)
    again = [extraction(2024, [claim(1, 2024, "Sales to Customer A were 12% of revenue.", "customer_concentration")])]
    inserted, _ = claimstore.load_extractions(db, again, verify=lambda c: True, company_names={})
    assert inserted == 0
    assert claimstore.table_counts(db) == before


def test_links_for_a_year_are_written_once(db):
    links = [linking.Link("vtest:ACME-2025-001", None, "new", "none"), linking.Link(None, "x", "dropped", "none")]
    assert claimstore.insert_links(db, "ltest", "ACME", 2025, links) == 2
    assert claimstore.insert_links(db, "ltest", "ACME", 2025, links) == 0  # NULLs would slip past UNIQUE


# --- Linking ---------------------------------------------------------------------------------


def C(uid, quote, t="export_controls_trade", section="1A"):
    return linking.C(uid, t, quote, section)


def run(earlier, later, judge=lambda lc, cs: [], **kw):
    links, _, _ = asyncio.run(linking.link_years(earlier, later, 2025, judge=judge, **kw))
    return {(k.claim_uid, k.prior_uid): k for k in links}


def test_identical_quote_is_unchanged_and_unmatched_are_new_or_dropped(fake_embed):
    q = "The U.S. restricts exports of our H100 products to China."
    links = run([C("e1", q), C("e2", "Our Russia business stopped after sanctions in 2022.")],
                [C("l1", q), C("l2", "We opened a new fab in Arizona with state support.")])
    assert links[("l1", "e1")].status == "unchanged" and links[("l1", "e1")].method == "exact"
    assert links[("l2", None)].status == "new"
    assert links[(None, "e2")].status == "dropped"


def test_one_to_one_exact_beats_a_near_duplicate(fake_embed):
    q = "The U.S. restricts exports of our H100 products to China."
    links = run([C("e1", q)], [C("l1", q + " Again."), C("l2", q)])
    assert ("l2", "e1") in links and links[("l1", None)].status == "new"


def test_type_and_section_must_match(fake_embed):
    q = "The U.S. restricts exports of our H100 products to China."
    links = run([C("e1", q, section="7"), C("e2", q, t="legal_regulatory")], [C("l1", q)])
    assert links[("l1", None)].status == "new"


def test_borderline_pairs_go_to_the_judge(fake_embed, monkeypatch):
    monkeypatch.setattr(linking, "AUTO_COSINE", 0.99)
    monkeypatch.setattr(linking, "BORDER_COSINE", 0.3)
    asked = []

    def judge(lc, cs):
        asked.append((lc.uid, [c.uid for c in cs]))
        return [0]

    e = C("e1", "Sales to one customer were 12% of revenue in fiscal 2024.")
    lc = C("l1", "Sales to one customer were 22% of revenue in fiscal 2025.")
    links = run([e], [lc], judge=judge)
    assert asked == [("l1", ["e1"])]
    assert links[("l1", "e1")].status == "changed" and links[("l1", "e1")].method == "llm"
    assert run([e], [lc], judge=lambda lc, cs: [])[("l1", None)].status == "new"


def test_same_wording_checks_the_filings_not_the_quote_spans(fake_embed, monkeypatch):
    monkeypatch.setattr(linking, "BORDER_COSINE", 0.3)
    one = "The EU AI Act became effective on August 1, 2024."
    two = one + " It may limit how we train models in the EU."
    texts = {"1A": two, "7": ""}
    links = run([C("e1", two)], [C("l1", one)], judge=lambda lc, cs: [0], earlier_text=texts, later_text=texts)
    assert links[("l1", "e1")].status == "unchanged"
    # The later filing dropped the second sentence: the wording changed.
    links = run([C("e1", two)], [C("l1", one)], judge=lambda lc, cs: [0], earlier_text=texts,
                later_text={"1A": one, "7": ""})
    assert links[("l1", "e1")].status == "changed"


def test_linker_version_changes_with_thresholds(monkeypatch):
    v = linking.linker_version()
    monkeypatch.setattr(linking, "AUTO_COSINE", 0.5)
    assert linking.linker_version() != v


# --- Graph -----------------------------------------------------------------------------------


def test_entity_keys_and_surfaces():
    assert graph.key("Taiwan Semiconductor Manufacturing Company Limited (TSMC)") == "taiwan semiconductor manufacturing"
    assert graph.key("QUALCOMM INC/DE") == "qualcomm"
    assert graph.key("ASML Holding N.V.") == "asml holding n v"
    assert graph.segment_key("Client segment") == "client"
    assert "tsmc" in graph.surfaces("Taiwan Semiconductor Manufacturing Company (TSMC)")
    assert graph.GENERIC.match("customer a") and graph.GENERIC.match("semiconductor solutions customer")


def test_edges_are_grounded_resolved_and_checked(db, tmp_path):
    aliases = tmp_path / "aliases.csv"
    aliases.write_text("alias,canonical,note\nsamsung electronics,samsung,legal name\n")
    result = graph.build(db, "vtest", {"ACME": "ACME Corp"}, path=aliases)
    g = result["graph"]
    edges = {(r["relation"], r["target"]) for r in db.execute("SELECT * FROM edges WHERE graph=?", (g,))}
    assert ("depends_on_supplier", "company:taiwan semiconductor manufacturing") in edges  # "TSMC" in quote
    assert ("depends_on_supplier", "company:samsung") in edges  # alias resolved
    assert ("has_segment", "segment:acme:graphics") in edges
    assert ("depends_on_supplier", "company:micron") not in edges  # not named in the quote
    assert result["skipped"]["name not in quote"] == 1
    assert result["skipped"]["the filer itself"] == 1
    assert result["edges"]["faces_risk"] == 3  # Item 1A claims only; the Item 7 one isn't a risk
    assert result["edges"]["states"] == 4
    assert claimstore.unverified_edges(db, g) == 0
    assert graph.ungrounded_edges(db, g) == 0


def test_unverified_quotes_give_no_edges_and_the_check_catches_bad_rows(tmp_path):
    conn = claimstore.connect(tmp_path / "c.sqlite")
    claimstore.load_extractions(conn, [extraction(2024, [claim(1, 2024, "We rely on TSMC.", "supplier_dependence",
                                                               entities=[("company", "TSMC")])])],
                                verify=lambda c: False, company_names={"ACME": "ACME Corp"})
    result = graph.build(conn, "vtest", {"ACME": "ACME Corp"}, path=tmp_path / "none.csv")
    assert sum(result["edges"].values()) == 0
    conn.execute("INSERT INTO edges VALUES ('g', 'company:acme', 'states', 'claim:x', 2024, 'vtest:ACME-2024-001')")
    assert claimstore.unverified_edges(conn, "g") == 1


# --- Link eval -------------------------------------------------------------------------------


def row(cid, fy, block, quote, change="", prior=""):
    return LabelRow(claim_id=cid, ticker="ACME", fiscal_year=fy, section="1A", block=block,
                    claim_type="export_controls_trade", quote=quote, change=change, prior_claim_id=prior)


def test_score_pair_views_and_wrong_prior():
    later = [row("L1", 2025, 1, "a", "changed", "E1"), row("L2", 2025, 1, "b", "new"),
             row("L3", 2025, 2, "c", "unchanged", "E2"), row("L4", 2025, 3, "d", "new")]
    earlier = [row("E1", 2024, 1, "a"), row("E2", 2024, 2, "c"), row("E3", 2024, 1, "x")]
    m_later = {"L1": "l1", "L2": "l2", "L3": "l3"}  # L4 wasn't extracted
    m_earlier = {"E1": "e1", "E2": "e2", "E3": "e3"}
    links = [{"claim_uid": "l1", "prior_uid": "e1", "status": "changed"},
             {"claim_uid": "l2", "prior_uid": None, "status": "new"},
             {"claim_uid": "l3", "prior_uid": "e3", "status": "unchanged"},  # linked to the wrong claim
             {"claim_uid": None, "prior_uid": "e2", "status": "dropped"}]
    items = {i["claim"]: i for i in link_eval.score_pair(later, earlier, {1, 2}, m_later, m_earlier, links)}
    assert items["L1"]["correct"] and items["L2"]["correct"]
    assert items["L3"]["pred"] == "wrong prior"
    assert items["L4"]["pred"] == "missed" and not items["L4"]["extracted"]
    assert items["E3"]["gold"] == "dropped" and items["E3"]["pred"] == "linked"
    assert link_eval.summarise(list(items.values()))["correct"] == 2
    text = link_eval.markdown({"split": "t", "linker": "x", "end_to_end": {**link_eval.summarise([]), "ci95": (None, None)},
                               "given_extracted": {**link_eval.summarise([]), "ci95": (None, None)},
                               "confusion_end_to_end": {}, "confusion_given_extracted": {}, "pairs": []})
    assert "Link eval" in text
    json.dumps(items)  # the report is written as JSON
