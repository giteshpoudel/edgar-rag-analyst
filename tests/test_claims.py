"""Claim extractor and extraction eval (v3 Phase 2). Offline: a fake LLM and a synthetic filing."""

import asyncio
import json

import pytest
from openai.types.chat import ChatCompletion

from edgar_rag import claims, extraction_eval, llm
from edgar_rag.labeling import Block, LabelRow
from edgar_rag.llm_cache import LlmCache

PARA = "filler " * 60
B1 = Block(1, "1A", "Export controls.", [f"The U.S. government restricts exports of our H100 products to China. {PARA}"])
B2 = Block(2, "1A", "Customers.", [f"One customer accounted for 22% of our revenue in fiscal 2026. {PARA}"])
B3 = Block(3, "7", "Overview.", [f"We report two segments: Compute and Graphics. {PARA}"])


def reply(items) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps({"claims": items})}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 300, "total_tokens": 1300},
    })


class FakeLLM:
    def __init__(self):
        self.calls = []
        outer = self

        class Completions:
            async def create(self, **kw):
                outer.calls.append(kw)
                text = kw["messages"][1]["content"]
                items = []
                if "[B1]" in text:
                    items = [
                        {"claim_type": "export_controls_trade", "block": 1,
                         "quote": "The U.S. government restricts exports of our H100 products to China.",
                         "statement": "US export limits on H100 to China",
                         "entities": [{"kind": "country", "name": "China"}]},
                        {"claim_type": "export_controls_trade", "block": 1,  # duplicate
                         "quote": "The U.S. government restricts exports of our H100 products to China.",
                         "statement": "dup", "entities": []},
                        {"claim_type": "customer_concentration", "block": 2,  # paraphrased: not verbatim
                         "quote": "A single customer made up 22% of revenue in 2026.", "statement": "x"},
                        {"claim_type": "weather", "block": 2, "quote": "x", "statement": "x"},  # invalid type
                        {"claim_type": "customer_concentration", "block": 1,  # wrong block number, right quote
                         "quote": "One customer accounted for 22% of our revenue in fiscal 2026.",
                         "statement": "22% customer"},
                    ]
                return reply(items)

        self.chat = type("Chat", (), {"completions": Completions()})()


class FakeFiling:
    company, accession_number, url = "NVIDIA CORP", "0001-26-000001", "https://www.sec.gov/x"


@pytest.fixture
def setup(monkeypatch, tmp_path):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "_aclient", lambda: fake)
    monkeypatch.setattr(claims, "filing_html", lambda t, fy: (FakeFiling(), "<html>filing</html>"))
    monkeypatch.setattr(claims, "blocks_for_filing", lambda html: [B1, B2, B3])
    monkeypatch.setattr(claims, "_load_chunks", lambda t, a: [{"item": "1A", "chunk_id": "c-1",
                                                               "text": B1.text}])
    return fake, LlmCache(tmp_path / "cache.sqlite")


def test_windows_keep_whole_blocks_within_one_item(monkeypatch):
    monkeypatch.setattr(claims, "WINDOW_CHARS", len(B1.text) + 10)
    wins = claims.windows([B1, B2, B3])
    assert [[b.number for b in w] for w in wins] == [[1], [2], [3]]
    monkeypatch.setattr(claims, "WINDOW_CHARS", 100_000)
    assert [[b.number for b in w] for w in claims.windows([B1, B2, B3])] == [[1, 2], [3]]


def test_only_verbatim_deduplicated_valid_claims_are_kept(setup):
    fake, cache = setup
    x = asyncio.run(claims.extract_filing("NVDA", 2026, cache))
    assert [(c.claim_type, c.block) for c in x.claims] == [("export_controls_trade", 1), ("customer_concentration", 2)]
    assert x.dropped_quote == 1 and x.dropped_duplicate == 1 and x.dropped_invalid == 1
    assert x.claims[0].chunk_id == "c-1" and x.claims[1].chunk_id is None
    assert [c.claim_id for c in x.claims] == ["NVDA-2026-001", "NVDA-2026-002"]
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_second_run_reads_the_cache_and_makes_no_calls(setup):
    fake, cache = setup
    first = asyncio.run(claims.extract_filing("NVDA", 2026, cache))
    n = len(fake.calls)
    second = asyncio.run(claims.extract_filing("NVDA", 2026, cache))
    assert len(fake.calls) == n and second.cached_windows == second.windows and second.cost_usd == 0
    assert second.claims == first.claims


def test_prompt_change_is_a_new_extractor_version(monkeypatch):
    v1 = claims.extractor_version()
    monkeypatch.setattr(claims, "SYSTEM", claims.SYSTEM + "\nBe careful.")
    assert claims.extractor_version() != v1


def test_bad_json_counts_as_invalid_not_a_crash():
    assert claims._parse("not json") == ([], 1)
    assert claims._parse('{"claims": "nope"}') == ([], 0)


# --- extraction eval -------------------------------------------------------------------------


def gold(quote, t="export_controls_trade", block=1):
    return LabelRow(claim_id="g", ticker="NVDA", fiscal_year=2026, section="1A", block=block,
                    claim_type=t, quote=quote)


def pred(quote, t="export_controls_trade", block=1):
    return claims.Claim(claim_id="p", ticker="NVDA", fiscal_year=2026, section="1A", block=block, claim_type=t,
                        entities=[], statement="", quote=quote, chunk_id=None, accession="a", filing_url="u",
                        extractor="v")


TEXT = "First sentence about China exports. Second sentence about H100. Third sentence about tariffs."


def test_longer_quote_containing_the_gold_sentence_matches():
    m = extraction_eval.match_block([gold("Second sentence about H100.")],
                                    [pred("First sentence about China exports. Second sentence about H100.")], TEXT)
    assert m == [(0, 0)]


def test_different_type_or_sentence_does_not_match():
    assert extraction_eval.match_block([gold("Second sentence about H100.")],
                                       [pred("Second sentence about H100.", t="competition")], TEXT) == []
    assert extraction_eval.match_block([gold("Second sentence about H100.")],
                                       [pred("Third sentence about tariffs.")], TEXT) == []


def test_matches_are_one_to_one():
    m = extraction_eval.match_block([gold("Second sentence about H100.")],
                                    [pred("Second sentence about H100."), pred("Second sentence about H100.")], TEXT)
    assert len(m) == 1


def test_precision_recall_and_type_confusion():
    block = Block(1, "1A", "", [TEXT])
    g = [gold("First sentence about China exports."), gold("Third sentence about tariffs.")]
    p = [pred("First sentence about China exports."), pred("Second sentence about H100."),
         pred("Third sentence about tariffs.", t="geographic_exposure")]
    units = extraction_eval.score_filing("NVDA", 2026, {1}, g, p, [block])
    m = extraction_eval.metrics(units)
    assert m["matched"] == 1 and m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(1 / 2)
    assert m["matched_ignoring_type"] == 2  # the tariffs claim had the wrong type
