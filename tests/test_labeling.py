"""Hand-labelling tools for the v3 extraction eval. Offline: synthetic filings."""

from edgar_rag import labeling
from edgar_rag.labeling import Block, LabelRow, check_labels, normalize, sample_pair

PARA = "x " * 200  # a paragraph longer than a heading


def blocks(*headings, item="1A", start=1):
    return [Block(start + i, item, h, [f"{h} body. {PARA}"]) for i, h in enumerate(headings)]


def test_headings_start_blocks_and_page_furniture_is_dropped(monkeypatch):
    lines = ["Risk Factors", "Customers matter to us.", "We rely on a few customers. " + PARA,
             "Table of Contents", "Export rules may change.", "The U.S. restricts exports. " + PARA]

    class S:
        def __init__(self, item, text):
            self.item, self.text = item, text

    monkeypatch.setattr(labeling, "html_to_lines", lambda html: html)
    monkeypatch.setattr(labeling, "split_sections", lambda ls: [S("1A", "\n".join(ls))])
    got = labeling.blocks_for_filing(lines)
    # The section title before the first heading becomes a short intro block.
    assert [b.heading for b in got] == ["", "Customers matter to us.", "Export rules may change."]
    assert all("Table of Contents" not in b.text for b in got)


def test_sample_is_reproducible_and_pairs_matching_headings():
    later = blocks("Customer concentration risk.", "Export controls on China.", "Supply chain.", "Competition.")
    earlier = blocks("Customer concentration risks.", "Export controls.", "Supply chain.", "Competition.")
    a = sample_pair(later, earlier, 0.5, seed=1)
    assert a == sample_pair(later, earlier, 0.5, seed=1)
    chosen_later, chosen_earlier = a
    assert len(chosen_later) == 2
    later_headings = {later[n - 1].heading for n in chosen_later}
    for n in chosen_earlier:  # every earlier block chosen has a near-identical later heading
        assert any(normalize(earlier[n - 1].heading)[:10] == normalize(h)[:10] for h in later_headings)


def test_one_later_heading_pulls_in_at_most_one_earlier_block():
    later = blocks("Supply chain.")
    earlier = blocks("Supply chain.", "Supply chain.", "Supply chain.")
    _, chosen_earlier = sample_pair(later, earlier, 0.99, seed=0)
    assert len(chosen_earlier) == 1


def row(**kw):
    base = {"claim_id": "NVDA-2026-001", "ticker": "NVDA", "fiscal_year": 2026, "section": "1A", "block": 1,
            "claim_type": "export_controls_trade", "quote": "Export controls on China. body."}
    return LabelRow(**{**base, **kw})


def test_valid_labels_pass():
    bl = blocks("Export controls on China.")
    assert check_labels("NVDA", 2026, {1}, bl, [row()], None) == []


def test_quote_must_be_verbatim_and_type_known():
    bl = blocks("Export controls on China.")
    problems = check_labels("NVDA", 2026, {1}, bl, [row(quote="Export rules on China.", claim_type="trade")], None)
    assert any("verbatim" in p for p in problems) and any("unknown claim_type" in p for p in problems)


def test_curly_quotes_and_spacing_still_count_as_verbatim():
    bl = [Block(1, "1A", "", ["We’re  reliant on TSMC."])]
    assert check_labels("NVDA", 2026, {1}, bl, [row(quote="We're reliant on TSMC.")], None) == []


def test_claims_outside_sampled_blocks_are_rejected_unless_a_link_target():
    bl = blocks("A.", "B.")
    r = row(block=2, quote="B. body.", claim_id="NVDA-2025-009", fiscal_year=2025)
    assert any("not one of the blocks" in p for p in check_labels("NVDA", 2025, {1}, bl, [r], None))
    assert check_labels("NVDA", 2025, {1}, bl, [r], None, link_targets={"NVDA-2025-009"}) == []


def test_link_and_change_labels_are_checked():
    bl = blocks("Export controls on China.")
    prior = [row(claim_id="NVDA-2025-001", fiscal_year=2025)]
    ok = row(prior_claim_id="NVDA-2025-001", change="unchanged")
    assert check_labels("NVDA", 2026, {1}, bl, [ok], prior) == []
    wrong_change = row(prior_claim_id="NVDA-2025-001", change="changed")  # quotes are identical
    assert any("identical" in p for p in check_labels("NVDA", 2026, {1}, bl, [wrong_change], prior))
    missing = row(prior_claim_id="NVDA-2025-404", change="changed")
    assert any("not in the prior year" in p for p in check_labels("NVDA", 2026, {1}, bl, [missing], prior))
    no_change = row()
    assert any("change must be" in p for p in check_labels("NVDA", 2026, {1}, bl, [no_change], prior))
