"""Async agent, parallel tool calls, retries and the SEC throttle (W5). Offline."""

import asyncio
import time
from itertools import pairwise

import httpx
import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from edgar_rag import agent, llm, retry, sec_client, tools


def http_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# --- retry ---------------------------------------------------------------------------------


def test_retries_transient_errors_then_succeeds():
    attempts, waits = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(503)
        return "ok"

    assert retry.call(flaky, sleep=waits.append) == "ok"
    assert len(attempts) == 3 and len(waits) == 2


def test_does_not_retry_a_bad_request():
    attempts = []

    def bad():
        attempts.append(1)
        raise http_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        retry.call(bad, sleep=lambda s: None)
    assert len(attempts) == 1


def test_gives_up_after_the_last_attempt():
    attempts = []

    def down():
        attempts.append(1)
        raise http_error(503)

    with pytest.raises(httpx.HTTPStatusError):
        retry.call(down, attempts=3, sleep=lambda s: None)
    assert len(attempts) == 3


def test_backoff_honours_retry_after_and_grows_otherwise():
    assert retry.backoff(0, http_error(429, {"retry-after": "7"})) == 7
    assert retry.backoff(0, http_error(429, {"retry-after": "999"})) == retry.CAP_SECONDS
    assert all(0 <= retry.backoff(3) <= 8 for _ in range(100))  # base 1 s * 2^3


def test_async_retry(monkeypatch):
    monkeypatch.setattr(retry, "backoff", lambda attempt, exc=None: 0)
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 2:
            raise http_error(429)
        return "ok"

    assert asyncio.run(retry.acall(flaky)) == "ok"
    assert len(attempts) == 2


# --- staged citations ----------------------------------------------------------------------


def test_staged_citations_are_numbered_in_request_order_not_finish_order():
    sources = tools.Sources()
    first, second = tools.StagedSources(), tools.StagedSources()
    a = first.cite("filing", "A", "ref-a", "u") + first.cite("filing", "B", "ref-b", "u")
    b = second.cite("filing", "B", "ref-b", "u") + second.cite("filing", "C", "ref-c", "u")
    assert first.commit(a, sources) == "[1][2]"
    assert second.commit(b, sources) == "[2][3]"  # ref-b keeps its number
    assert [s.ref for s in sources.items] == ["ref-a", "ref-b", "ref-c"]


def test_staged_placeholder_ten_is_not_confused_with_one():
    staged = tools.StagedSources()
    text = "".join(staged.cite("filing", str(i), f"r{i}", "u") for i in range(11))
    assert staged.commit(text, tools.Sources()) == "".join(f"[{i}]" for i in range(1, 12))


# --- parallel tool calls in the agent loop ---------------------------------------------------


def completion(message: dict, finish: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


class FakeAsyncClient:
    """Replays scripted responses: first two tool calls, then a final answer."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.chat = type("Chat", (), {})()
        outer = self

        class Completions:
            async def create(self, **kwargs):
                return outer.responses.pop(0)

        self.chat.completions = Completions()


class SlowArgs(BaseModel):
    label: str
    seconds: float


def slow_cite(args: SlowArgs, sources: tools.Sources) -> str:
    time.sleep(args.seconds)  # blocking, like a real search; runs in a worker thread
    return sources.cite("filing", args.label, f"ref-{args.label}", "u") + f" result {args.label}"


@pytest.fixture
def scripted_agent(monkeypatch):
    monkeypatch.setitem(tools.TOOLS, "slow_cite", tools.Tool(
        name="slow_cite", description="test", args_model=SlowArgs, fn=slow_cite))
    monkeypatch.setattr(agent, "TOOLS", tools.TOOLS)
    monkeypatch.setattr(agent, "available_companies", lambda: "none")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "slow_cite", "arguments": '{"label": "first", "seconds": 0.4}'}},
        {"id": "c2", "type": "function", "function": {"name": "slow_cite", "arguments": '{"label": "second", "seconds": 0.1}'}},
    ]
    fake = FakeAsyncClient([
        completion({"content": "", "tool_calls": calls}, "tool_calls"),
        completion({"content": "Done [1][2]."}),
    ])
    monkeypatch.setattr(llm, "_aclient", lambda: fake)


def test_tool_calls_from_one_turn_run_in_parallel(scripted_agent):
    started = time.monotonic()
    answer = agent.ask("q")
    elapsed = time.monotonic() - started
    assert elapsed < 0.45  # 0.4 s + 0.1 s would be 0.5 s one after the other
    # The second call finished first, but numbering follows the order the model asked for.
    assert [c.result for c in answer.tool_calls] == ["[1] result first", "[2] result second"]
    assert [s.ref for s in answer.sources] == ["ref-first", "ref-second"]
    assert answer.cited == [1, 2] and not answer.invalid_citations


# --- SEC throttle --------------------------------------------------------------------------


def test_sec_throttle_is_shared_across_clients_and_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("SEC_USER_AGENT", "Test Person test@example.org")
    times, statuses = [], [503, 200, 200]

    def handler(request):
        times.append(time.monotonic())
        return httpx.Response(statuses.pop(0), json={}, headers={"retry-after": "0"})

    transport = httpx.MockTransport(handler)
    one = sec_client.SecClient(cache_dir=tmp_path, transport=transport)
    two = sec_client.SecClient(cache_dir=tmp_path, transport=transport)
    one._get("https://data.sec.gov/a")  # 503, then retried
    two._get("https://data.sec.gov/b")  # a different client still waits its turn
    gaps = [b - a for a, b in pairwise(times)]
    assert len(times) == 3
    assert all(g >= sec_client.MIN_SECONDS_BETWEEN_REQUESTS * 0.95 for g in gaps)


def test_cached_reads_need_no_user_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    (tmp_path / "company_tickers.json").write_text('{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}')
    assert sec_client.SecClient(cache_dir=tmp_path).lookup_cik("AAPL") == (320193, "Apple Inc.")
