"""Tracing and cost accounting (W4). Offline: the LLM client is replaced by a fake."""

import pytest
from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from edgar_rag import llm, tracing
from edgar_rag.evaluation import percentile


def fake_response(prompt=1000, cached=800, completion=200, reasoning=50, content="OK") -> ChatCompletion:
    """A response shaped like DeepSeek's, including its prompt_cache_hit_tokens extra field."""
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion,
                  "completion_tokens_details": {"reasoning_tokens": reasoning},
                  "prompt_cache_hit_tokens": cached, "prompt_cache_miss_tokens": prompt - cached},
    })


class FakeClient:
    def __init__(self, response):
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {"create": lambda _self, **kw: response})()


@pytest.fixture
def spans(monkeypatch):
    """Capture finished spans in memory, and answer every LLM call with a fake response."""
    exporter = InMemorySpanExporter()
    tracing._provider().add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(llm, "_client", lambda: FakeClient(fake_response()))
    return exporter


def test_cost_uses_cache_hit_rate_for_cached_input():
    # deepseek-flash: $0.006 cache hit, $0.30 miss, $1.20 output per 1M tokens
    prompt_cost, output_cost = tracing.cost_usd("deepseek-flash", 1000, 200, cached=800)
    assert prompt_cost == pytest.approx((800 * 0.006 + 200 * 0.30) / 1e6)
    assert output_cost == pytest.approx(200 * 1.20 / 1e6)


def test_unknown_model_has_no_cost_rather_than_zero():
    assert tracing.cost_usd("some-other-model", 1000, 200) is None


def test_llm_call_records_tokens_cost_and_messages(spans):
    llm.complete([{"role": "user", "content": "hi"}], model="deepseek-flash")
    (span,) = [s for s in spans.get_finished_spans() if s.name == "llm"]
    a = span.attributes
    assert a["openinference.span.kind"] == "LLM"
    assert a["llm.token_count.prompt"] == 1000
    assert a["llm.token_count.prompt_details.cache_read"] == 800
    assert a["llm.token_count.completion_details.reasoning"] == 50
    assert a["llm.input_messages.0.message.content"] == "hi"
    assert a["llm.output_messages.0.message.content"] == "OK"
    assert a["llm.cost.total"] == pytest.approx(sum(tracing.cost_usd("deepseek-flash", 1000, 200, 800)))


def test_collector_totals_one_trace_and_keeps_judge_cost_apart(spans):
    with tracing.span("eval.question", "CHAIN"):
        trace_id = tracing.current_trace_id()
        with tracing.span("agent", "AGENT"):
            llm.complete([{"role": "user", "content": "q"}], model="deepseek-flash")
            llm.complete([{"role": "user", "content": "q"}], model="deepseek-flash")
        llm.complete([{"role": "user", "content": "grade"}], model="deepseek-v4-pro", span_name="judge")
    usage = tracing.COLLECTOR.pop(trace_id)
    one_call = sum(tracing.cost_usd("deepseek-flash", 1000, 200, 800))
    assert usage.llm_calls == 2  # the judge call isn't counted as the system's
    assert usage.cost_usd == pytest.approx(2 * one_call)
    assert usage.judge_cost_usd == pytest.approx(sum(tracing.cost_usd("deepseek-v4-pro", 1000, 200, 800)))
    assert usage.prompt_tokens == 2000 and usage.cached_prompt_tokens == 1600
    assert "LLM" in usage.seconds and "JUDGE" in usage.seconds


def test_spans_nest_into_one_trace(spans):
    with tracing.span("eval.question", "CHAIN"), tracing.span("agent", "AGENT"):
        llm.complete([{"role": "user", "content": "q"}], model="deepseek-flash")
    by_name = {s.name: s for s in spans.get_finished_spans()}
    root, agent, call = by_name["eval.question"], by_name["agent"], by_name["llm"]
    assert len({root.context.trace_id, agent.context.trace_id, call.context.trace_id}) == 1
    assert call.parent.span_id == agent.context.span_id
    assert agent.parent.span_id == root.context.span_id


def test_unpriced_model_marks_trace_cost_unknown(spans):
    with tracing.span("q", "CHAIN"):
        trace_id = tracing.current_trace_id()
        llm.complete([{"role": "user", "content": "q"}], model="some-other-model")
    assert tracing.COLLECTOR.pop(trace_id).cost_known is False


def test_long_attributes_are_cut():
    assert len(tracing._value("x" * 50_000)) < tracing.MAX_TEXT + 100


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == pytest.approx(9.55)
    assert percentile([], 50) is None
