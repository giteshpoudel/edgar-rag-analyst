"""Number grounding: is every number in an answer backed by the run's tool results?

Each number the answer states is one of:
    supported    it appears in a tool result or retrieved passage (allowing rounding and units)
    derived      it can be recomputed from two source numbers (growth, ratio, difference, sum,
                 or a gap in percentage points)
    unsupported  neither: made up, misread or miscalculated

Things that look like numbers but aren't claims are skipped: dates, years, section and note
labels ("Item 1A", "Section 232"), citation markers ("[3]"), list numbering, and numbers the
question itself contains ("a 10% adverse move").
"""

import re
from itertools import permutations

from pydantic import BaseModel

SCALES = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6,
          "thousand": 1e3, "k": 1e3}
MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
LABEL_WORDS = ("item", "items", "note", "notes", "section", "sections", "article", "rule", "rules",
               "level", "tier", "part", "exhibit", "form", "schedule", "page", "phase", "step", "fy",
               "fiscal", "q", "quarter", "class", "series", "chapter", "topic", "asc", "asu", "no", "number")

NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-−])?\$?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s*(?:%|percentage points?\b|percent\b|pp\b|basis points?\b|bps\b|"
    r"trillion\b|billion\b|million\b|thousand\b|bn\b|[bmk]\b|x\b))?",
    re.IGNORECASE,
)


class NumberClaim(BaseModel):
    raw: str
    value: float
    kind: str  # percent | amount | ratio
    written: float  # the number as written, before scale words ("416.2" in "$416.2 billion")
    decimals: int  # digits after the decimal point as written (rounding precision)
    verdict: str = "unchecked"  # supported | derived | unsupported
    how: str = ""


class GroundingReport(BaseModel):
    claims: list[NumberClaim]

    @property
    def unsupported(self) -> list[NumberClaim]:
        return [c for c in self.claims if c.verdict == "unsupported"]

    @property
    def rate(self) -> float:
        """Share of the answer's numbers that are unsupported (0 when it has none)."""
        return len(self.unsupported) / len(self.claims) if self.claims else 0.0


def _is_label(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 14):start].lower()
    after = text[end:end + 12].lower()
    if re.search(rf"\b({'|'.join(LABEL_WORDS)})\.?\s*$", before):
        return True  # "Item 1A", "Note 20", "Section 232", "FY2025", "Q4"
    if re.search(rf"\b({MONTHS})\.?\s*$", before) or re.match(rf"\s*({MONTHS})\b", after):
        return True  # "September 27", "27 September"
    if before.endswith("[") or re.match(r"^\s*\]", after):
        return True  # citation marker [3]
    if re.match(r"^[-‑–](k|q|f)s?\b", after):
        return True  # form names: 10-K, 10-Q, 20-F
    return bool(re.match(r"^(st|nd|rd|th)\b", after))  # ordinals: 1st, 2nd


def extract_claims(text: str, skip_values: set[float] = frozenset()) -> list[NumberClaim]:
    """Numbers stated in text, normalized to a value and kind, minus things that aren't claims."""
    claims = []
    for m in NUMBER.finditer(text):
        num = m.group("num")
        suffix = (m.group("suffix") or "").strip().lower()
        start, end = m.span()
        if _is_label(text, start, end):
            continue
        value = float(num.replace(",", ""))
        if not suffix and re.fullmatch(r"(19|20)\d\d", num):
            continue  # a year
        if not suffix and "," not in num and "." not in num and value < 10 and re.match(r"^\s*[.)]\s", text[end:end + 3]):
            continue  # list numbering "1." / "2)"
        is_money = "$" in m.group(0)
        if not suffix and not is_money and "," not in num and "." not in num and value < 20:
            continue  # small counts ("two segments", "14 companies"): low risk, very common
        if m.group("sign") and suffix in ("%", "percent"):
            value = -value
        decimals = len(num.split(".")[1]) if "." in num else 0
        if suffix in ("%", "percent", "pp") or suffix.startswith("percentage point"):
            kind = "percent"
        elif suffix in ("bps",) or suffix.startswith("basis point"):
            kind, value, decimals = "percent", value / 100, decimals + 2
        elif suffix == "x":
            kind = "ratio"
        else:
            kind = "amount"
            value *= SCALES.get(suffix, 1)
        if abs(value) in skip_values:
            continue  # stated in the question itself
        claims.append(NumberClaim(raw=m.group(0).strip(), value=value, kind=kind,
                                  written=float(num.replace(",", "")), decimals=decimals))
    return claims


ISO_DATE = re.compile(r"\b\d{4}[-‑–]\d{2}[-‑–]\d{2}\b")  # also non-breaking (U+2011) and en dashes


class Sources(BaseModel):
    direct: list[float]  # values for direct matching, with the scaled copies tables imply
    percents: list[float]
    operands: list[float]  # meaningful figures as written (>= 100, or with decimals)
    rows: list[list[float]]  # operands grouped by source line (a table row, a sentence)
    singles: list[float]  # values from short lines: one XBRL lookup or calculation each


def _source_values(sources: str) -> Sources:
    """Numbers in the tool results, grouped so derivations can be limited to related figures."""
    text = ISO_DATE.sub(" ", sources)  # 2024-09-28 is a date, not three numbers
    direct, operands, percents, rows, singles = set(), set(), set(), [], set()
    for line in text.split("\n"):
        row = []
        for c in extract_claims(line):
            v = abs(c.value)
            if c.kind == "percent":
                percents.add(v)
                row.append(v)
                continue
            direct.add(v)
            for scale in (1e3, 1e6, 1e9):
                direct.add(v * scale)
            # Small integers (counts, day numbers, footnote marks) aren't meaningful operands.
            if v >= 100 or c.decimals:
                operands.add(v)
                row.append(v)
        if row:
            rows.append(row)
        if 1 <= len(row) <= 3:
            singles.update(row)
    return Sources(direct=sorted(direct), percents=sorted(percents), operands=sorted(operands),
                   rows=rows, singles=sorted(singles))


def _close(claim: NumberClaim, value: float) -> bool:
    """Within the claim's own rounding: "$416.2 billion" covers 416.15-416.25 billion, so the
    tolerance is half a unit of the last written digit, times any scale word. Never tighter
    than 0.05%, to absorb float noise."""
    target = abs(claim.value)
    if target == 0:
        return value == 0
    scale = target / claim.written if claim.written else 1
    tol = max(0.5 * 10 ** -claim.decimals * scale, 0.0005 * target)
    return abs(abs(value) - target) <= tol


def check(answer: str, sources: str, question: str = "") -> GroundingReport:
    """Classify every number in the answer against the numbers in the sources."""
    skip = {abs(c.value) for c in extract_claims(ISO_DATE.sub(" ", question))}
    claims = extract_claims(ISO_DATE.sub(" ", answer), skip)
    src = _source_values(sources + "\n" + question)
    # Tables often print percentages without a % sign ("Effective tax rate | 15.6 | 24.1"), so a
    # percentage may also match a meaningful plain number as written.
    percent_pool = sorted(set(src.percents) | set(src.operands))
    exact: dict[int, float] = {}
    for i, c in enumerate(claims):
        hit = next((v for v in (percent_pool if c.kind == "percent" else src.direct) if _close(c, v)), None)
        if hit is not None:
            c.verdict, c.how = "supported", "appears in a tool result"
            exact[i] = hit
    # Anchors: figures the answer states that the sources support, plus single-value tool
    # results. Derivations may combine two anchors, or two figures from the same source line;
    # pairing arbitrary numbers from long passages "derives" almost anything by coincidence.
    anchors = sorted(set(src.singles) | {abs(exact[i]) for i in exact})
    # Within a row, real comparisons are between nearby columns (this year vs last year).
    row_pairs = {(row[i], row[j]) for row in src.rows for i in range(len(row)) for j in range(len(row))
                 if i != j and abs(i - j) <= ROW_WINDOW}
    anchor_pairs = set(permutations(anchors, 2))
    for i, c in enumerate(claims):
        if c.verdict == "supported":
            continue
        c.verdict, c.how, value = _derive(c, anchor_pairs, sums=True)
        if c.verdict == "unsupported":
            # Two cells of one row are compared, rarely added: no sums from row pairs.
            c.verdict, c.how, value = _derive(c, row_pairs, sums=False)
        if value is not None:
            exact[i] = value

    # Second pass: gaps between figures the answer itself derived ("margin rose 0.46 points" from
    # two computed margins), using the exact derived values rather than the rounded text.
    derived_percents = sorted({exact[i] for i, c in enumerate(claims) if c.kind == "percent" and i in exact})
    for c in claims:
        if c.verdict == "unsupported" and c.kind == "percent":
            for a, b in permutations(derived_percents, 2):
                if _close(c, b - a):
                    c.verdict, c.how = "derived", f"gap between derived figures {b:.4g} and {a:.4g}"
                    break
    return GroundingReport(claims=claims)


ROW_WINDOW = 2  # cells at most this many columns apart can be compared


def _derive(c: NumberClaim, pairs: set[tuple[float, float]], sums: bool) -> tuple[str, str, float | None]:
    """Try to recompute the claim from a related pair of source numbers.
    Returns (verdict, explanation, exact recomputed value or None)."""
    for a, b in sorted(pairs):
        if c.kind == "percent":
            candidates = [(b - a, f"gap {b:g} - {a:g} (points)")]
            if sums:
                candidates.append((a + b, f"sum {a:g} + {b:g}"))
            if a:
                candidates += [((b - a) / a * 100, f"percent change {a:g} -> {b:g}")]
            if b:
                candidates += [(a / b * 100, f"share {a:g} / {b:g}")]
        else:
            candidates = [((b - a) * s, f"difference {b:g} - {a:g}") for s in (1, 1e3, 1e6, 1e9)]
            if sums:
                candidates += [((a + b) * s, f"sum {a:g} + {b:g}") for s in (1, 1e3, 1e6, 1e9)]
            if c.kind == "ratio" and b:
                candidates.append((a / b, f"ratio {a:g} / {b:g}"))
        for value, how in candidates:
            if _close(c, value):
                return "derived", how, value
    return "unsupported", "not in any tool result and not recomputable from related figures", None
