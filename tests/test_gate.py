"""CI gate logic (W6). Offline: synthetic eval records."""

import pytest

from edgar_rag import gate
from edgar_rag.evaluation import Record


def rec(i: int, passed: bool = True, numbers: list[bool] | None = None, error: str = "") -> Record:
    return Record(
        id=f"q{i:02d}", category="numeric", mode="agent", question="?", answer="a", tools_used=[],
        retrieved_items=[], invalid_citations=[], prompt_tokens=0, completion_tokens=0, seconds=1.0,
        numbers_found=numbers if numbers is not None else [True], retrieval_hit=None,
        reciprocal_rank=None, key_points_covered=[], refused=False, passed=passed, error=error,
    )


def run(failing: set[int], n: int = 40, wrong_numbers: set[int] = frozenset()) -> list[Record]:
    return [rec(i, passed=i not in failing, numbers=[i not in wrong_numbers]) for i in range(n)]


@pytest.fixture
def baseline() -> gate.Baseline:
    # Three runs of 40: 2, 1 and 3 failures (95.0%, 97.5%, 92.5%; mean 95.0%).
    runs = [run({1, 2}), run({1}), run({1, 2, 3})]
    return gate.make_baseline(runs, "v2", "dev", {})


def test_tolerance_is_the_spread_plus_one_question(baseline):
    assert baseline.pass_rate == pytest.approx(0.95)
    # widest drop below the mean: 95.0 - 92.5 = 2.5 points, plus one question (2.5 points)
    assert baseline.tolerance == pytest.approx(0.05)
    assert "q01" not in baseline.always_passed and "q05" in baseline.always_passed


def test_a_normal_unlucky_run_passes(baseline):
    result = gate.check(run({1, 2, 3, 4}), baseline)  # 90.0% = mean - tolerance
    assert result.passed


def test_a_real_regression_fails(baseline):
    result = gate.check(run({1, 2, 3, 4, 5}), baseline)  # 87.5%
    assert not result.passed
    assert [c.name for c in result.checks if not c.ok] == ["Pass rate"]
    assert result.newly_failing == ["q04", "q05"]


def test_any_drop_in_number_accuracy_fails(baseline):
    result = gate.check(run({1}, wrong_numbers={7}), baseline)
    assert [c.name for c in result.checks if not c.ok] == ["Number accuracy"]


def test_run_errors_fail_even_when_pass_rate_is_fine(baseline):
    records = run(set())
    records[0] = rec(0, error="ValueError: boom")
    assert [c.name for c in gate.check(records, baseline).checks if not c.ok] == ["Run errors"]


def test_markdown_names_the_failures(baseline):
    records = run({1, 2, 3, 4, 5})
    text = gate.markdown(gate.check(records, baseline), baseline, records)
    assert "❌ Failed" in text and "★ `q05`" in text and "`q01`" in text
