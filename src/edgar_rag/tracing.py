"""OpenTelemetry tracing (W4): each question becomes one trace, a tree of timed spans.

    eval.question                 CHAIN      the eval wrapper (id, category, passed)
      agent                       AGENT      question in, answer out
        llm                       LLM        messages, reply, tokens, cost
        tool.search_filings       TOOL       arguments, result
          retrieve                RETRIEVER  query, mode, passages returned with scores
            embed_query           EMBEDDING  (time waiting for the shared model lock is recorded)
        llm                       LLM
        grounding.check           CHAIN      numbers flagged as unsupported
      judge                       LLM        grading call (eval cost, kept apart from system cost)

Attribute names follow OpenInference (github.com/Arize-ai/openinference), the convention Arize
Phoenix reads, so it can show prompts, token counts and retrieved passages properly.

Spans are always recorded. An in-process collector totals each trace's cost and time, so eval
reports use the same numbers Phoenix shows. With TRACING=phoenix the spans are also exported.
"""

import atexit
import json
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.trace import Span, Status, StatusCode

from edgar_rag import config

KIND = "openinference.span.kind"
PROJECT = "edgar-rag-analyst"
MAX_TEXT = 20_000  # characters per attribute: long tool results are cut so traces stay small
JUDGE_SPAN = "judge"  # LLM spans with this name are eval cost, not the system's cost


# --- Cost ----------------------------------------------------------------------------------


def cost_usd(model: str, prompt: int, completion: int, cached: int = 0) -> tuple[float, float] | None:
    """(input cost, output cost) in USD, or None if the model has no price in the table.
    Cached input tokens (DeepSeek's prompt_cache_hit_tokens) are billed at the cache-hit rate."""
    prices = config.price(model)
    if prices is None:
        return None
    hit, miss, out = prices
    cached = min(cached, prompt)
    return (cached * hit + (prompt - cached) * miss) / 1e6, completion * out / 1e6


# --- Per-trace totals ----------------------------------------------------------------------


@dataclass
class TraceUsage:
    """Totals for one trace (one question)."""

    cost_usd: float = 0.0  # the system's LLM calls (judge excluded)
    cost_known: bool = True  # False if any call used a model missing from the price table
    judge_cost_usd: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    # Seconds by span kind. LLM spans never nest inside each other, and neither do RETRIEVER
    # spans, so these sums don't double count (TOOL spans contain RETRIEVER spans, so the TOOL
    # total includes retrieval time).
    seconds: dict[str, float] = field(default_factory=dict)


class UsageCollector(SpanProcessor):
    """Adds each finished span's tokens, cost and duration to its trace's totals."""

    def __init__(self, keep: int = 5000) -> None:
        self._traces: OrderedDict[int, TraceUsage] = OrderedDict()
        self._lock = threading.Lock()  # eval questions finish spans from several threads
        self._keep = keep  # cap memory if traces are never popped

    def on_end(self, span: ReadableSpan) -> None:
        attrs = span.attributes or {}
        kind = str(attrs.get(KIND, "UNKNOWN"))
        seconds = (span.end_time - span.start_time) / 1e9
        is_judge = kind == "LLM" and span.name == JUDGE_SPAN
        with self._lock:
            usage = self._traces.setdefault(span.context.trace_id, TraceUsage())
            key = "JUDGE" if is_judge else kind
            usage.seconds[key] = usage.seconds.get(key, 0.0) + seconds
            if kind == "LLM":
                cost = attrs.get("llm.cost.total")
                if is_judge:
                    usage.judge_cost_usd += float(cost or 0.0)
                else:
                    usage.llm_calls += 1
                    usage.prompt_tokens += int(attrs.get("llm.token_count.prompt", 0))
                    usage.cached_prompt_tokens += int(attrs.get("llm.token_count.prompt_details.cache_read", 0))
                    usage.completion_tokens += int(attrs.get("llm.token_count.completion", 0))
                    usage.reasoning_tokens += int(attrs.get("llm.token_count.completion_details.reasoning", 0))
                    if cost is None:
                        usage.cost_known = False
                    else:
                        usage.cost_usd += float(cost)
            while len(self._traces) > self._keep:
                self._traces.popitem(last=False)

    def pop(self, trace_id: int) -> TraceUsage:
        with self._lock:
            return self._traces.pop(trace_id, TraceUsage())


COLLECTOR = UsageCollector()


# --- Setup ---------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _provider() -> TracerProvider:
    # Our own provider rather than the global one, so importing this never changes tracing for
    # other libraries in the process.
    provider = TracerProvider(
        resource=Resource.create({"service.name": PROJECT, "openinference.project.name": PROJECT})
    )
    provider.add_span_processor(COLLECTOR)
    if config.tracing() == "phoenix":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Batches spans and sends them in the background, so tracing adds no wait to answers.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=config.phoenix_endpoint())))
    atexit.register(provider.shutdown)  # flush the last batch before the process exits
    return provider


def tracer() -> trace.Tracer:
    return _provider().get_tracer("edgar_rag")


def _value(v):
    """OpenTelemetry attributes must be str, bool, int or float (or lists of them)."""
    if isinstance(v, (bool, int, float)):
        return v
    text = v if isinstance(v, str) else json.dumps(v, default=str)
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + f"... [{len(text) - MAX_TEXT:,} chars cut]"


def set_attributes(span: Span, attributes: dict) -> None:
    for key, v in attributes.items():
        if v is not None:
            span.set_attribute(key, _value(v))


@contextmanager
def span(name: str, kind: str, **attributes):
    """A span of an OpenInference kind (CHAIN, AGENT, LLM, TOOL, RETRIEVER, EMBEDDING,
    RERANKER). Exceptions are recorded on the span and re-raised."""
    with tracer().start_as_current_span(name) as s:
        s.set_attribute(KIND, kind)
        set_attributes(s, attributes)
        yield s


def start_span(name: str, kind: str, **attributes) -> Span:
    """A span that isn't made current and must be ended by the caller: for work that runs in
    another thread (enter it there with trace.use_span) and finishes after it returns."""
    s = tracer().start_span(name)
    s.set_attribute(KIND, kind)
    set_attributes(s, attributes)
    return s


def current_trace_id() -> int:
    return trace.get_current_span().get_span_context().trace_id


def trace_id_hex(trace_id: int) -> str:
    """The id Phoenix shows and searches by."""
    return f"{trace_id:032x}"


def mark_error(span: Span, message: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))


# --- LLM spans -----------------------------------------------------------------------------


def _message_attributes(prefix: str, i: int, message: dict) -> dict:
    p = f"{prefix}.{i}.message"
    out = {f"{p}.role": message.get("role"), f"{p}.content": message.get("content") or None}
    for j, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function", {})
        out[f"{p}.tool_calls.{j}.tool_call.function.name"] = fn.get("name")
        out[f"{p}.tool_calls.{j}.tool_call.function.arguments"] = fn.get("arguments")
    return out


def record_llm(span: Span, model: str, messages: list[dict], response, params: dict) -> None:
    """Put an LLM call's inputs, reply, token counts and cost on its span."""
    attrs: dict = {"llm.model_name": model, "llm.provider": "deepseek",
                   "llm.invocation_parameters": params}
    for i, m in enumerate(messages):
        attrs.update(_message_attributes("llm.input_messages", i, m))
    choice = response.choices[0]
    reply = choice.message.model_dump(include={"role", "content", "tool_calls"}, exclude_none=True)
    attrs.update(_message_attributes("llm.output_messages", 0, reply))
    attrs["output.value"] = choice.message.content or None
    attrs["edgar.finish_reason"] = choice.finish_reason

    usage = response.usage
    if usage is not None:
        extra = usage.model_extra or {}
        cached = int(extra.get("prompt_cache_hit_tokens") or getattr(
            usage.prompt_tokens_details, "cached_tokens", 0) or 0)
        reasoning = int(getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0)
        attrs.update({
            "llm.token_count.prompt": usage.prompt_tokens,
            "llm.token_count.completion": usage.completion_tokens,
            "llm.token_count.total": usage.total_tokens,
            "llm.token_count.prompt_details.cache_read": cached,
            "llm.token_count.completion_details.reasoning": reasoning,
        })
        cost = cost_usd(model, usage.prompt_tokens, usage.completion_tokens, cached)
        if cost is not None:
            attrs.update({"llm.cost.prompt": cost[0], "llm.cost.completion": cost[1],
                          "llm.cost.total": cost[0] + cost[1]})
    set_attributes(span, attrs)
