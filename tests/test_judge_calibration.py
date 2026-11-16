"""v3 Phase 6: second-provider judge routing, rescoring stored answers, and judge calibration
statistics. Offline."""

import asyncio
from types import SimpleNamespace

import pytest

from edgar_rag import calibration, config, evaluation, llm, scoring, tracing
from edgar_rag.evaluation import GoldenItem, Record

# --- Cohen's kappa -----------------------------------------------------------------------------


def test_kappa_known_values():
    a = [True, True, False, False]
    assert calibration.kappa(a, a) == 1.0
    assert calibration.kappa(a, [not x for x in a]) == -1.0
    # 3 of 4 agree, both graders 50% true: chance 0.5, kappa (0.75 - 0.5) / 0.5
    assert calibration.kappa(a, [True, False, False, False]) == pytest.approx(0.5)
    assert calibration.kappa([True] * 3, [True] * 3) is None  # one class only: undefined, not 1
    assert calibration.agreement(a, [True, False, False, False]) == 0.75


def test_verdict_rule_matches_eval_pass_rule():
    assert calibration.verdict("unanswerable", [], refused=True)
    assert not calibration.verdict("unanswerable", [], refused=False)
    assert calibration.verdict("mixed_claim", [True, False, True], refused=False)  # 2/3 coverage
    assert not calibration.verdict("mixed_claim", [True, False, False], refused=False)
    assert not calibration.verdict("first_appeared", [True, True], refused=True)


def test_reference_set_shape():
    rows = calibration.load_reference()
    assert len(rows) == 80 and sum(r["kind"] == "perturbed" for r in rows) == 20
    for r in rows:
        assert len(r["key_points"]) == len(r["item"].key_points) and r["answer"]
        if r["kind"] == "perturbed":  # every perturbed answer carries an error the reference fails
            assert r["perturbation"]


# --- Provider routing --------------------------------------------------------------------------


def test_gemini_models_route_to_google(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    assert config.provider("gemini-3.1-flash-lite") == ("g-test", config.GEMINI_BASE_URL)
    assert config.provider("deepseek-v4-pro")[1] == config.llm_base_url()
    assert config.price("gemini-3.1-flash-lite") is not None  # never an unpriced (free-looking) call
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.provider("gemini-3.1-flash-lite")


def test_client_choice(monkeypatch):
    deepseek, gemini = object(), object()
    monkeypatch.setattr(llm, "_client", lambda: deepseek)
    monkeypatch.setattr(llm, "_gemini_client", lambda: gemini)
    assert llm._client_for("deepseek-flash") is deepseek
    assert llm._client_for("gemini-3.1-flash-lite") is gemini


def test_usage_counts_include_unreported_thinking_tokens():
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=170,
                            prompt_tokens_details=None, model_extra={})
    assert tracing.usage_counts(SimpleNamespace(usage=usage)) == (100, 0, 70)
    usage.total_tokens = 120  # DeepSeek: total = prompt + completion, nothing added
    assert tracing.usage_counts(SimpleNamespace(usage=usage)) == (100, 0, 20)


# --- Rescoring stored answers ------------------------------------------------------------------


def _record(answer: str) -> Record:
    return Record(id="q1", category="first_appeared", mode="agent", question="q", answer=answer, tools_used=[],
                  retrieved_items=[], invalid_citations=[], prompt_tokens=0, completion_tokens=0, truncated=False,
                  top_score=None, seconds=1.0, numbers_found=[], years_found=[], retrieval_hit=None,
                  reciprocal_rank=None, key_points_covered=[False], refused=False, passed=False)


def test_rescore_uses_current_gold_and_chosen_judge(monkeypatch):
    seen = {}

    async def fake_judge(question, answer, key_points, model=None):
        seen["model"] = model
        return scoring.Judgement(key_points=[True] * len(key_points), refused=False)

    monkeypatch.setattr(evaluation, "ajudge", fake_judge)
    item = GoldenItem(id="q1", category="first_appeared", question="q", key_points=["first in FY2023"],
                      expected_numbers=[{"value": 91, "kind": "percent", "tolerance": 0.5}], expected_years=[2023])
    out = asyncio.run(evaluation.arescore([_record("FY2023: about 91.2%.")], {"q1": item}, "gemini-x"))
    assert seen["model"] == "gemini-x"
    assert out[0].passed and out[0].years_found == [True] and out[0].numbers_found == [True]
    out = asyncio.run(evaluation.arescore([_record("FY2024: about 91.2%.")], {"q1": item}, "gemini-x"))
    assert not out[0].passed  # the year check is deterministic, whatever the judge says
