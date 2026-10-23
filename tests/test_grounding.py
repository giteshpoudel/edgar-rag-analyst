"""Number grounding checker (W3). Offline."""

from edgar_rag.grounding import check


def test_half_way_rounding_of_a_derived_gap_is_supported():
    # 71.07 - 49.52 = 21.55, which rounds to 21.6; in floats the difference is 0.0500000000000007.
    sources = "gross_profit / revenue (%):\n1. NVDA: 71.07% [1]\n2. AMD: 49.52% [2]"
    report = check("NVIDIA's margin is 21.6 percentage points above AMD's.", sources)
    assert not report.unsupported


def test_made_up_number_is_unsupported():
    sources = "[1] Apple Inc. revenue FY2025: 416,161,000,000.00 USD"
    report = check("Apple's revenue was $416.2 billion and its margin was 38.7%.", sources)
    assert [c.raw for c in report.unsupported] == ["38.7%"]


def test_dates_labels_and_years_are_not_claims():
    report = check("In Item 7 of its 10-K for fiscal 2025 (ended September 27, 2025), see [3].", "")
    assert report.claims == []
