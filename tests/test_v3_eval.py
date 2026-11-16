"""v3 Phase 5a: the multi-year search switch, the year check, the drafting helpers and the
v3 golden set's shape. Offline."""

from collections import Counter

import pytest

from edgar_rag import agent, drafting, retriever, tools
from edgar_rag.evaluation import GOLDEN_SETS, REFUSAL_CATEGORIES, load_golden, year_named

# --- year check --------------------------------------------------------------------------------


@pytest.mark.parametrize("answer, ok", [
    ("It first appeared in the FY2023 10-K.", True),
    ("Starting with fiscal 2023, the filing says...", True),
    ("(2023)", True),
    ("Revenue was $20,230 million.", False),  # digits inside a bigger number don't count
    ("12023 units", False),
    ("It first appeared in fiscal 2024.", False),
])
def test_year_named(answer, ok):
    assert year_named(2023, answer) is ok


# --- multi-year search switch ------------------------------------------------------------------


def test_search_table_switch(monkeypatch):
    monkeypatch.delenv("EDGAR_SEARCH_TABLE", raising=False)
    monkeypatch.delenv("EDGAR_TOOLSET", raising=False)
    assert agent.active_tools() is tools.TOOLS  # v2 default untouched
    monkeypatch.setenv("EDGAR_SEARCH_TABLE", "chunks_multi")
    search = agent.active_tools()["search_filings"]
    assert "fiscal_year" in search.args_model.model_fields
    assert "fiscal_year" not in tools.TOOLS["search_filings"].args_model.model_fields  # original not mutated
    monkeypatch.setenv("EDGAR_SEARCH_TABLE", "chunks_mutli")
    with pytest.raises(RuntimeError, match="EDGAR_SEARCH_TABLE"):
        agent.active_tools()


def test_where_adds_fiscal_year():
    assert retriever._where("nvda", "1a", 2024) == "fiscal_year = 2024 AND ticker = 'NVDA' AND item = '1A'"
    assert retriever._where(None, None) is None
    assert retriever._where(None, None, "2025") == "fiscal_year = 2025"  # coerced to int, no injection


# --- drafting helpers --------------------------------------------------------------------------

TEXT = {
    2023: {"1A": "We rely on TSMC for wafers. Our COVID-19 risks remain significant today.", "7": ""},
    2024: {"1A": "We rely on TSMC for wafers. The EU AI Act could impose new obligations on us.", "7": ""},
}


@pytest.fixture
def fake_filings(monkeypatch):
    monkeypatch.setattr(drafting, "section_text", lambda t, fy: TEXT[fy])
    monkeypatch.setattr(drafting, "years", lambda t: sorted(TEXT))


def test_sentence_diff(fake_filings):
    added, removed = drafting.sentence_diff("AAA", 2024, "1A")
    assert added == ["The EU AI Act could impose new obligations on us."]
    assert removed == ["Our COVID-19 risks remain significant today."]


def test_mentions_and_quote_years(fake_filings):
    assert drafting.mentions("AAA", "EU AI Act") == {2023: 0, 2024: 1}
    assert drafting.mentions("AAA", "covid-19", section="7") == {2023: 0, 2024: 0}
    assert drafting.mentions("AAA", "TSM") == {2023: 0, 2024: 0}  # whole words only
    assert drafting.quote_years("AAA", "We rely on TSMC for wafers.") == [2023, 2024]


# --- the v3 golden set -------------------------------------------------------------------------


def test_golden_v3_shape():
    items = load_golden(GOLDEN_SETS["v3"])
    assert len({i.id for i in items}) == len(items) >= 80
    assert all(i.source == "drafted" and i.verified and i.split in ("dev", "test") for i in items)
    by_cat = Counter(i.category for i in items)
    assert set(by_cat) == {"first_appeared", "dropped", "wording_change", "multi_year", "shared_entity",
                           "mixed_claim", "unanswerable"}
    for cat in by_cat:  # every category is in both splits
        assert {i.split for i in items if i.category == cat} == {"dev", "test"}
    for i in items:
        if i.category in REFUSAL_CATEGORIES:
            assert i.should_refuse and not i.expected_years
        else:
            assert i.evidence and i.key_points and not i.should_refuse
    assert all(i.expected_years for i in items if i.category in ("first_appeared", "dropped", "multi_year",
                                                               "wording_change", "mixed_claim"))
