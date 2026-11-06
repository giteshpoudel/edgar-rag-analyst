"""Cost ledger, hard cap and LLM cache (v3 Phase 0). Offline: the LLM client is a fake."""

import asyncio
import sqlite3
import time

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from edgar_rag import budget, evaluation, llm, tracing
from edgar_rag.llm_cache import LlmCache, sha256_json, sha256_text

MESSAGES = [{"role": "user", "content": "What was Apple's revenue in fiscal 2025?"}]


def response(prompt=1000, cached=800, completion=200) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                  "prompt_cache_hit_tokens": cached, "prompt_cache_miss_tokens": prompt - cached},
    })


class FakeClient:
    """Counts calls; optionally waits (async) or raises."""

    def __init__(self, delay=0.0, error=None):
        self.calls, self.delay, self.error = 0, delay, error
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls += 1
                if outer.error:
                    raise outer.error
                return response()

        class AsyncCompletions:
            async def create(self, **kwargs):
                outer.calls += 1
                await asyncio.sleep(outer.delay)
                return response()

        self.chat = type("Chat", (), {"completions": Completions()})()
        self.async_client = type("AsyncClient", (), {"chat": type("Chat", (), {"completions": AsyncCompletions()})()})()


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("EDGAR_LEDGER", "on")
    monkeypatch.setenv("EDGAR_LEDGER_PATH", str(tmp_path / "ledger.sqlite"))
    monkeypatch.setenv("EDGAR_BUDGET_USD", "10")
    return tmp_path / "ledger.sqlite"


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm, "_client", lambda: client)
    return client


def worst(max_tokens=4000) -> float:
    return budget.worst_case_usd("deepseek-flash", MESSAGES, None, max_tokens)


ONE_CALL = sum(tracing.cost_usd("deepseek-flash", 1000, 200, 800))


def test_worst_case_is_an_upper_bound_and_settles_to_actual(ledger, fake):
    llm.complete(MESSAGES, model="deepseek-flash")
    s = budget.summary()
    assert s.spent_usd == pytest.approx(ONE_CALL) and s.reserved_usd == pytest.approx(0)
    assert s.calls == 1 and worst() > ONE_CALL


def test_call_that_would_cross_the_cap_is_never_made(ledger, fake, monkeypatch):
    monkeypatch.setenv("EDGAR_BUDGET_USD", str(worst() * 0.5))
    with pytest.raises(budget.BudgetExceeded):
        llm.complete(MESSAGES, model="deepseek-flash")
    assert fake.calls == 0
    assert budget.summary().spent_usd == 0


def test_spend_accumulates_until_the_cap(ledger, fake, monkeypatch):
    # Each call needs spent + its worst case to fit. After two calls 2 x actual + worst no longer does.
    monkeypatch.setenv("EDGAR_BUDGET_USD", str(worst() + 1.5 * ONE_CALL))
    llm.complete(MESSAGES, model="deepseek-flash")
    llm.complete(MESSAGES, model="deepseek-flash")
    with pytest.raises(budget.BudgetExceeded):
        llm.complete(MESSAGES, model="deepseek-flash")
    assert fake.calls == 2


def test_concurrent_calls_cannot_all_slip_under_the_cap(ledger, monkeypatch):
    # 16 calls start at once with room for 3 worst cases. Each reserves before calling, so only
    # 3 are made, even though each would see $0 spent if it only checked settled spend.
    client = FakeClient(delay=0.05)
    monkeypatch.setattr(llm, "_aclient", lambda: client.async_client)
    monkeypatch.setenv("EDGAR_BUDGET_USD", str(worst() * 3.5))

    async def run():
        return await asyncio.gather(*(llm.acomplete(MESSAGES, model="deepseek-flash") for _ in range(16)),
                                    return_exceptions=True)

    results = asyncio.run(run())
    refused = [r for r in results if isinstance(r, budget.BudgetExceeded)]
    assert client.calls == 3 and len(refused) == 13
    assert budget.summary().spent_usd == pytest.approx(3 * ONE_CALL)


def test_failed_call_releases_its_reservation(ledger, monkeypatch):
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    error = openai.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    client = FakeClient(error=error)
    monkeypatch.setattr(llm, "_client", lambda: client)
    with pytest.raises(openai.BadRequestError):
        llm.complete(MESSAGES, model="deepseek-flash")
    s = budget.summary()
    assert s.failed == 1 and s.reserved_usd == pytest.approx(0) and s.spent_usd == 0


def test_unpriced_model_is_refused_before_calling(ledger, fake):
    with pytest.raises(budget.UnpricedModel):
        llm.complete(MESSAGES, model="some-unpriced-model")
    assert fake.calls == 0


def test_ledger_off_writes_nothing(ledger, fake, monkeypatch):
    monkeypatch.setenv("EDGAR_LEDGER", "off")
    llm.complete(MESSAGES, model="deepseek-flash")
    assert fake.calls == 1 and not ledger.exists()


def test_purpose_and_span_name_label_calls(ledger, fake):
    llm.complete(MESSAGES, model="deepseek-flash")
    llm.complete(MESSAGES, model="deepseek-v4-pro", span_name="judge")
    with budget.purpose("extract"):
        llm.complete(MESSAGES, model="deepseek-flash")
    assert {p for p, _, _ in budget.summary().by_purpose} == {"llm", "judge", "extract"}


def test_expired_reservation_stops_counting(ledger, monkeypatch):
    monkeypatch.setenv("EDGAR_BUDGET_USD", str(worst() * 1.5))
    budget.open_reservation("deepseek-flash", MESSAGES, None, 4000, "llm")  # a crashed process
    with pytest.raises(budget.BudgetExceeded):
        budget.open_reservation("deepseek-flash", MESSAGES, None, 4000, "llm")
    with sqlite3.connect(ledger) as conn:
        conn.execute("UPDATE calls SET created_utc = ?", (time.time() - budget.RESERVATION_TTL_SECONDS - 1,))
    budget.open_reservation("deepseek-flash", MESSAGES, None, 4000, "llm")


def test_manual_spend_counts_against_the_cap(ledger, fake, monkeypatch):
    monkeypatch.setenv("EDGAR_BUDGET_USD", "1")
    budget.add_manual(0.999, "ci", "CI gate")
    with pytest.raises(budget.BudgetExceeded):
        llm.complete(MESSAGES, model="deepseek-flash")


def test_eval_run_stops_on_budget_instead_of_scoring_failures(monkeypatch):
    async def out_of_budget(question, mode):
        raise budget.BudgetExceeded("cap")

    monkeypatch.setattr(evaluation, "_run_mode", out_of_budget)
    item = evaluation.GoldenItem(id="q1", category="numeric", question="What was Apple's revenue?")
    with pytest.raises(budget.BudgetExceeded):
        asyncio.run(evaluation._evaluate(item, "agent"))


# --- LLM cache -------------------------------------------------------------------------------


def test_cache_hit_returns_stored_result_and_never_overwrites(tmp_path):
    cache = LlmCache(tmp_path / "cache.sqlite")
    key = (sha256_text("filing text"), sha256_json({"prompt": "v1"}), "deepseek-flash", "1A:0")
    assert cache.get(*key) is None
    assert cache.put(*key, {"claims": [1]}, cost_usd=0.01) is True
    assert cache.put(*key, {"claims": [2]}) is False  # first result stays
    assert cache.get(*key) == {"claims": [1]}
    assert cache.stats() == (1, pytest.approx(0.01))


def test_any_key_change_is_a_miss(tmp_path):
    cache = LlmCache(tmp_path / "cache.sqlite")
    base = [sha256_text("filing"), sha256_json({"prompt": "v1"}), "deepseek-flash", "1A:0"]
    cache.put(*base, {"ok": True})
    for i, other in enumerate([sha256_text("filing2"), sha256_json({"prompt": "v2"}), "deepseek-v4-pro", "1A:1"]):
        key = list(base)
        key[i] = other
        assert cache.get(*key) is None


def test_json_hash_ignores_key_order():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
