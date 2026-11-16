"""Build and verify the v3 golden set (evals/v3/golden_v3.jsonl) from raw filing text.

Every fact here was found by reading the 10-K sections (drafting.mentions / sentence_diff and the
sentence chains), never from the claims database. Running this script re-checks each item:

  first_appeared  the phrase is absent from every earlier fiscal year in the data, present in the stated one
  dropped         present in the stated last year, absent from every later year
  shared_entity   each listed "yes" company's latest filing names the entity; each "no" company's doesn't
  every item      each gold quote is found verbatim in the stated fiscal year's Items 1A/7

Usage:  python evals/v3/build_golden_v3.py      (writes the file only if every check passes)
"""

import json
import re
import sys
from pathlib import Path

from edgar_rag import companies, drafting
from edgar_rag.evaluation import GoldenItem

OUT = Path(__file__).with_name("golden_v3.jsonl")
NOTE = "verified by Claude against raw filing text (Items 1A/7) with drafting.py; not drawn from the claims DB"
NAME = {"AAPL": "Apple", "ADI": "Analog Devices", "AMAT": "Applied Materials", "AMD": "AMD", "AVGO": "Broadcom",
        "INTC": "Intel", "KLAC": "KLA", "LRCX": "Lam Research", "MCHP": "Microchip", "MRVL": "Marvell",
        "MSFT": "Microsoft", "MU": "Micron", "NVDA": "NVIDIA", "QCOM": "Qualcomm", "TXN": "Texas Instruments"}
LATEST = {c.ticker: c.fiscal_year for c in companies.indexed_companies()}
SCOPE = "in its 10-K risk factors (Item 1A) or MD&A (Item 7)"


def pct(*vals):
    """Filings round ("approximately 91%"), while an agent may compute 91.2% from dollar figures:
    whole-number percents get +/-0.5 points, one-decimal percents +/-0.05."""
    return [{"value": v, "kind": "percent", "tolerance": 0.5 if float(v).is_integer() else 0.05} for v in vals]


def usd(v, tol=None):
    return {"value": v, "kind": "usd", **({"tolerance": tol} if tol is not None else {})}


def ev(ticker, quote, year):
    return {"ticker": ticker, "quote": quote, "_year": year}


def phrase_quote(ticker, phrase, year):
    """A short exact quote around the first sentence naming the phrase in that year."""
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", re.IGNORECASE)
    for sec in drafting.SECTIONS:
        for s in drafting.sentences(drafting.section_text(ticker, year)[sec]):
            if m := pat.search(s):
                start = max(0, m.start() - 100)
                start = s.rfind(" ", 0, start) + 1 if start else 0
                end = min(len(s), m.end() + 110)
                end = s.find(" ", end) if s.find(" ", end) != -1 else len(s)
                return s[start:end].strip(" •")
    raise ValueError(f"{ticker} FY{year}: no sentence names {phrase!r}")


items: list[dict] = []
checks: list[tuple] = []  # (kind, ticker, phrase, year)


def add(id_, category, question, companies_, years, *, numbers=(), key_points=(), evidence=(), refuse=False,
        notes=NOTE):
    items.append({"id": id_, "category": category, "question": question, "expected_numbers": list(numbers),
                  "key_points": list(key_points), "should_refuse": refuse, "evidence": list(evidence),
                  "companies": list(companies_), "source": "drafted", "expected_years": list(years),
                  "verified": True, "notes": notes})


# --- first_appeared: phrase absent from every earlier year in the data -------------------------
FIRST = [
    ("NVDA", "H20", 2026, "NVIDIA's H20 chips"), ("NVDA", "Blackwell", 2025, "its Blackwell architecture"),
    ("NVDA", "EU AI Act", 2024, "the EU AI Act"), ("NVDA", "generative AI", 2024, "generative AI"),
    ("MSFT", "OpenAI", 2023, "OpenAI"), ("AMD", "ZT Systems", 2024, "ZT Systems"),
    ("AMD", "OpenAI", 2025, "OpenAI"), ("AMD", "AI act", 2023, "the EU AI Act"),
    ("AVGO", "VMware", 2022, "VMware"), ("TXN", "CHIPS and Science", 2022, "the CHIPS and Science Act (CHIPS Act)"),
    ("ADI", "Section 232", 2025, "Section 232 investigations"), ("KLAC", "rare earth", 2025, "rare earth minerals"),
    ("LRCX", "rare earth", 2025, "rare earth elements"), ("AAPL", "artificial intelligence", 2023, "artificial intelligence"),
    ("INTC", "Altera", 2024, "Altera"), ("MRVL", "Silicon Valley Bank", 2024, "Silicon Valley Bank"),
    ("MU", "Section 232", 2025, "Section 232 of the Trade Expansion Act"),
]
FIRST_Q = [
    "In which fiscal year's 10-K did {co} first mention {what} {scope}?",
    "Starting with which fiscal year does {co} discuss {what} {scope}?",
    "When did {what} first show up {scope2} of {co}'s annual reports?",
]
for i, (t, phrase, y, what) in enumerate(FIRST, 1):
    q = FIRST_Q[i % 3].format(co=NAME[t], what=what, scope=SCOPE,
                              scope2="in the risk factors (Item 1A) or MD&A (Item 7)")
    add(f"v3-first-{i:02d}", "first_appeared", q, [t], [y],
        key_points=[f"First mentioned in the fiscal {y} 10-K",
                    f"Not mentioned in the fiscal {y - 1} or earlier filings in the data"],
        evidence=[ev(t, phrase_quote(t, phrase, y), y)])
    checks.append(("first", t, phrase, y))

# --- dropped: last year that mentions the phrase ---------------------------------------------
DROPPED = [
    ("NVDA", "Arm", 2024, "the terminated Arm acquisition"), ("INTC", "LIBOR", 2022, "LIBOR"),
    ("AVGO", "Huawei", 2023, "Huawei"), ("MU", "COVID-19", 2022, "COVID-19"), ("NVDA", "COVID-19", 2023, "COVID-19"),
    ("AMAT", "COVID-19", 2022, "COVID-19"), ("TXN", "COVID-19", 2022, "COVID-19"), ("QCOM", "COVID-19", 2023, "COVID-19"),
    ("MRVL", "Inphi", 2025, "its Inphi acquisition"), ("LRCX", "Ukraine", 2024, "Ukraine"),
    ("MSFT", "Digital Markets Act", 2024, "the EU Digital Markets Act"),
    ("AVGO", "Inflation Reduction Act", 2023, "the Inflation Reduction Act"),
]
DROP_Q = [
    "What was the last fiscal year in which {co} mentioned {what} {scope}?",
    "{co} used to discuss {what} {scope}. Which fiscal year's filing was the last to mention it?",
    "Is {what} still mentioned {scope2} in {co}'s latest 10-K? If not, when did it last appear?",
]
for i, (t, phrase, y, what) in enumerate(DROPPED, 1):
    q = DROP_Q[i % 3].format(co=NAME[t], what=what, scope=SCOPE,
                             scope2="in the risk factors (Item 1A) or MD&A (Item 7)")
    add(f"v3-drop-{i:02d}", "dropped", q, [t], [y],
        key_points=[f"Last mentioned in the fiscal {y} 10-K",
                    f"Not mentioned in the fiscal {y + 1} or later filings"],
        evidence=[ev(t, phrase_quote(t, phrase, y), y)])
    checks.append(("drop", t, phrase, y))

# --- shared_entity: a candidate list mixes companies that do and don't name the entity ---------
SHARED = [
    ("TSMC", ["TSMC", "Taiwan Semiconductor"], ["AMD", "INTC", "AVGO"], ["NVDA", "MCHP"], "1A"),
    ("Huawei", ["Huawei"], ["QCOM", "LRCX", "KLAC"], ["AVGO", "NVDA"], "1A"),
    ("OpenAI", ["OpenAI"], ["MSFT", "NVDA", "AMD"], ["AVGO", "INTC"], None),
    ("Samsung", ["Samsung"], ["MU", "INTC", "QCOM", "AMD"], ["AAPL", "TXN"], "1A"),
    ("SMIC", ["SMIC", "Semiconductor Manufacturing International"], ["INTC", "MCHP"], ["AMAT", "KLAC", "LRCX"], "1A"),
    ("the EU AI Act", ["AI Act"], ["NVDA", "AMD", "MCHP", "MSFT", "ADI"], ["INTC", "TXN"], "1A"),
    ("Section 232 trade investigations", ["Section 232"], ["AAPL", "ADI", "KLAC", "MU"], ["NVDA", "TXN"], None),
    ("the CHIPS Act", ["CHIPS Act", "CHIPS and Science"], ["MU", "INTC", "TXN", "ADI", "AMAT"], ["NVDA", "QCOM"], None),
]
for i, (label, phrases, yes, no, sec) in enumerate(SHARED, 1):
    cands = sorted(yes + no, key=lambda t: NAME[t])
    where = "risk factors (Item 1A)" if sec == "1A" else "risk factors (Item 1A) or MD&A (Item 7)"
    q = (f"Which of these companies name {label} in the {where} of their most recent 10-K: "
         f"{', '.join(NAME[t] for t in cands)}?")
    add(f"v3-shared-{i:02d}", "shared_entity", q, cands, [],
        key_points=[f"Yes: {', '.join(NAME[t] for t in yes)}", f"No: {', '.join(NAME[t] for t in no)}"],
        evidence=[ev(t, phrase_quote(t, next(p for p in phrases if drafting.mentions(t, p).get(LATEST[t])), LATEST[t]),
                     LATEST[t]) for t in yes])
    checks += [("has", t, tuple(phrases), sec) for t in yes] + [("lacks", t, tuple(phrases), sec) for t in no]

# --- multi_year: one disclosed figure across all five filings ---------------------------------
MULTI = [
    ("AMAT", "share of net sales (net revenue) to customers outside the United States", range(2021, 2026),
     [91, 88, 85, 86, 89], [
         (2021, "approximately 91% of Applied's net sales were to customers in regions outside the United States"),
         (2025, "In fiscal 2025, approximately 89% of our net revenue was to customers in regions outside the United States")]),
    ("AVGO", "share of net revenue from sales to distributors", range(2021, 2026), [53, 56, 57, 48, 48], [
        (2021, "For fiscal year 2021, sales to distributors accounted for 53% of our net revenue"),
        (2025, "For fiscal year 2025, sales to distributors accounted for 48% of our net revenue")]),
    ("AVGO", "share of its contract manufacturers' wafers made by TSMC", range(2021, 2026), [89, 90, 90, 95, 95], [
        (2021, "TSMC, one of our CMs, manufactured approximately 89% of the wafers manufactured by our CMs during fiscal year 2021"),
        (2025, "During fiscal year 2025, approximately 95% of the wafers manufactured by our CMs were produced by TSMC")]),
    ("AVGO", "share of net revenue from shipments to China (including Hong Kong)", range(2021, 2026),
     [35, 35, 32, 20, 17], [
         (2021, "In each of fiscal years 2021 and 2020, approximately 35% of our net revenue came from shipments or deliveries to China"),
         (2025, "In fiscal years 2025 and 2024, 17% and 20%, respectively, of our net revenue came from shipments or deliveries to China")]),
    ("INTC", "share of net revenue from its three largest customers", range(2021, 2026), [43, 42, 40, 45, 43], [
        (2025, "Collectively, our three largest customers accounted for 43% of our net revenue in 2025, 45% of our net revenue in 2024 and 40% of our net revenue in 2023")]),
    ("LRCX", "share of total revenue from China", range(2022, 2027), [31, 26, 42, 34, 34], [
        (2022, "represented approximately 31%, 35%, and 31% of our total revenue for fiscal years 2022, 2021, and 2020"),
        (2026, "represented approximately 34%, 34%, and 42% of our total revenue for fiscal years 2026, 2025, and 2024")]),
    ("MRVL", "share of net revenue from its ten largest customers", range(2022, 2027), [56, 63, 72, 81, 82], [
        (2022, "represented 56% of our net revenue for the year ended January 29, 2022"),
        (2026, "represented 82% of our total net revenue for fiscal 2026")]),
    ("MRVL", "share of net revenue from sales shipped to customers with operations in Asia", range(2022, 2027),
     [78, 75, 70, 75, 77], [
         (2026, "Sales shipped to customers with operations in Asia represented approximately 77% and 75% of our net revenue in fiscal 2026 and 2025")]),
    ("NVDA", "unfavorable gross margin impact from inventory provisions (net of sales of items previously written down)",
     range(2022, 2027), [0.9, 7.5, 2.7, 2.3, 2.6], [
         (2023, "was an unfavorable impact of 7.5% in fiscal year 2023 and 0.9% in fiscal year 2022"),
         (2026, "was an unfavorable impact of 2.6% in fiscal year 2026 and 2.3% in fiscal year 2025")]),
    ("TXN", "free cash flow as a percentage of revenue", range(2021, 2026), [34.3, 29.6, 7.7, 9.6, 16.6], [
        (2021, "Free cash flow was $6.29 billion and represented 34.3% of revenue"),
        (2023, "Free cash flow was $1.35 billion and represented 7.7% of revenue"),
        (2025, "Free cash flow was $2.94 billion and represented 16.6% of revenue")]),
    ("MSFT", "effective tax rate", range(2022, 2027), [13, 19, 18, 18, 19], [
        (2023, "Our effective tax rate for fiscal years 2023 and 2022 was 19% and 13%, respectively"),
        (2026, "Our effective tax rate for fiscal years 2026 and 2025 was 19% and 18%, respectively")]),
    ("MCHP", "share of net sales made through distributors", range(2022, 2027), [48, 47, 47, 45, 47], [
        (2022, "Distributors accounted for approximately 48% and 50% of our net sales in fiscal 2022 and fiscal 2021"),
        (2026, "Distributors accounted for approximately 47% and 45% of our net sales in fiscal 2026 and fiscal 2025")]),
]
for i, (t, what, yrs, vals, quotes) in enumerate(MULTI, 1):
    yrs = list(yrs)
    q = f"How did {NAME[t]}'s {what} change from fiscal {yrs[0]} to fiscal {yrs[-1]}? Give each year's figure."
    add(f"v3-multi-{i:02d}", "multi_year", q, [t], yrs,
        numbers=pct(*dict.fromkeys(vals)),
        key_points=[", ".join(f"FY{y}: {v}%" for y, v in zip(yrs, vals, strict=True))],
        evidence=[ev(t, qt, y) for y, qt in quotes])

# --- wording_change: same disclosure, reworded between two filings -----------------------------
WORDING = [
    ("AVGO", 2023, 2024, "How did Broadcom's risk-factor statement about its largest end customers change between its fiscal 2023 and fiscal 2024 10-Ks?",
     pct(20, 35, 40),
     ["FY2023 named Apple (about 20% of net revenue) alongside the top five end customers (35%)",
      "FY2024 no longer names Apple and gives only the top five end customers, about 40%"],
     [(2023, "to Apple and our top five end customers, accounted for approximately 20% and 35% of our net revenue for fiscal year 2023"),
      (2024, "to our top five end customers accounted for approximately 40% of our net revenue for fiscal year 2024")]),
    ("AVGO", 2023, 2024, "How did Broadcom describe its largest direct distributor customer in the fiscal 2023 10-K compared with the fiscal 2024 10-K?",
     pct(21, 28),
     ["FY2023 names WT Microelectronics, a distributor, at 21% of net revenue",
      "FY2024 no longer names it: 'one customer, which is a distributor', at 28%"],
     [(2023, "Direct sales to WT Microelectronics Co., Ltd., a distributor, accounted for 21% and 20% of our net revenue"),
      (2024, "Direct sales to one customer, which is a distributor, accounted for 28% and 21% of our net revenue")]),
    ("MU", 2024, 2025, "How did Micron's description of revenue from customers headquartered outside the United States change between its 2024 and 2025 10-Ks?",
     pct(80),
     ["2024: nearly half of revenue from customers headquartered outside the US; over 80% shipped to locations outside the US",
      "2025: approximately one-third from customers headquartered outside the US; approximately 80% shipped outside the US"],
     [(2024, "In 2024, nearly half of our revenue was from sales to customers who have headquarters located outside the United States"),
      (2025, "In 2025, approximately one-third of our revenue was from sales to customers who have headquarters located outside the United States")]),
    ("NVDA", 2025, 2026, "How did NVIDIA change the way it reports revenue from outside the United States between its fiscal 2025 and fiscal 2026 10-Ks?",
     pct(53, 31, 41),
     ["FY2025 filing: customers outside the United States were 53% of revenue for fiscal 2025",
      "FY2026 filing: customers headquartered outside the United States, 31% for fiscal 2026 and 41% for fiscal 2025 on that basis"],
     [(2025, "Revenue from sales to customers outside of the United States accounted for 53% and 56% of total revenue for fiscal years 2025 and 2024"),
      (2026, "Revenue from sales to customers headquartered outside of the United States accounted for 31% and 41% of total revenue for fiscal years 2026 and 2025")]),
    ("MCHP", 2023, 2024, "Which fiscal year do the foreign-sales figures in the risk factors of Microchip's fiscal 2024 10-K refer to, and how do they compare with the fiscal 2023 10-K?",
     pct(78, 21, 14),
     ["The FY2024 risk factor still says 'During fiscal 2023', unchanged from the FY2023 filing",
      "Same figures in both: about 78% foreign, 21% China, 14% Taiwan"],
     [(2024, "During fiscal 2023, approximately 78% of our net sales were made to foreign customers, including 21% in China and 14% in Taiwan")]),
    ("MRVL", 2024, 2025, "How did Marvell's wording about its ten largest customers change between the fiscal 2024 and fiscal 2025 10-Ks, and what share did they represent?",
     pct(72, 81),
     ["FY2024: ten largest customers 'including this distributor' were 72% of net revenue",
      "FY2025: 'inclusive of our distributor and direct customers', 81% of total net revenue"],
     [(2024, "including this distributor, represented 72% of our net revenue for the fiscal year ended February 3, 2024"),
      (2025, "inclusive of our distributor and direct customers, represented 81% of our total net revenue")]),
    ("MU", 2024, 2025, "How did the CHIPS Act investment tax credit rate Micron cites change between its 2024 and 2025 10-Ks?",
     pct(25, 35),
     ["2024: a 25% investment tax credit on qualified US semiconductor manufacturing investments",
      "2025: a 35% credit, now described in addition to CHIPS Act direct funding"],
     [(2024, "In addition, we receive a 25% investment tax credit on qualified investments in U.S. semiconductor manufacturing under the CHIPS Act"),
      (2025, "In addition to the CHIPS Act direct funding, we receive a 35% investment tax credit")]),
    ("AMAT", 2023, 2024, "Applied Materials reports the share of its business from customers outside the United States. What term and figure did it use in the fiscal 2023 10-K versus the fiscal 2024 10-K?",
     pct(85, 86),
     ["FY2023: about 85% of net sales", "FY2024: about 86% of net revenue (the term changed from net sales to net revenue)"],
     [(2023, "In fiscal 2023, approximately 85% of our net sales were to customers in regions outside the United States"),
      (2024, "In fiscal 2024, approximately 86% of our net revenue was to customers in regions outside the United States")]),
    ("MSFT", 2025, 2026, "How did Microsoft's naming and growth of search advertising revenue excluding traffic acquisition costs change between its fiscal 2025 and fiscal 2026 10-Ks?",
     pct(20, 12),
     ["FY2025: 'search and news advertising' revenue ex-TAC grew 20%",
      "FY2026: renamed 'search advertising' revenue ex-TAC, grew 12%"],
     [(2025, "Search and news advertising revenue excluding traffic acquisition costs increased 20%"),
      (2026, "Search advertising revenue excluding traffic acquisition costs increased 12%")]),
    ("MCHP", 2022, 2023, "How did Microchip's name for its largest product line, and that line's share of net sales, change between the fiscal 2022 and fiscal 2023 10-Ks?",
     pct(56.0, 56.3),
     ["FY2022: 'Microcontrollers', about 56.0% of net sales", "FY2023: 'Mixed-signal microcontrollers', about 56.3%"],
     [(2022, "Microcontrollers and associated application development systems accounted for approximately 56.0% and 54.5%"),
      (2023, "Mixed-signal microcontrollers and associated application development systems accounted for approximately 56.3% and 56.0%")]),
]
for i, (t, y0, y1, q, nums, kps, quotes) in enumerate(WORDING, 1):
    add(f"v3-word-{i:02d}", "wording_change", q, [t], [y0, y1], numbers=nums, key_points=kps,
        evidence=[ev(t, qt, y) for y, qt in quotes])

# --- mixed_claim: a narrative claim plus a figure from the same filing --------------------------
MIXED = [
    ("NVDA", 2026, "What fiscal 2026 charge did NVIDIA take because of the new H20 export license requirement, and what was its total fiscal 2026 revenue and growth?",
     [usd(4.5e9), usd(215_938e6), *pct(65)],
     ["USG required a license for H20 exports to China (April 2025)", "$4.5 billion charge in Q1 FY2026 for excess inventory and purchase obligations",
      "Revenue $215.9 billion, up 65%"],
     ["we incurred a $4.5 billion charge in the first quarter of fiscal year 2026 associated with H20", "Revenue | 215,938 | 130,497 | Up 65%"]),
    ("MSFT", 2026, "By how much did Microsoft's Intelligent Cloud revenue grow in fiscal 2026, and what drove it?",
     [usd(31.5e9), *pct(30)],
     ["Intelligent Cloud revenue increased $31.5 billion or 30%", "Driven by Azure and other cloud services (server products and cloud services up 31%)"],
     ["Intelligent Cloud Revenue increased $31.5 billion or 30%"]),
    ("MRVL", 2026, "What was Marvell's fiscal 2026 net revenue and growth, and how concentrated was it among its ten largest customers?",
     [usd(8.2e9), *pct(42, 82)],
     ["Net revenue $8.2 billion, 42% higher than fiscal 2025", "Ten largest customers were 82% of net revenue"],
     ["Net revenue in fiscal 2026 was $8.2 billion, 42% higher than net revenue of $5.8 billion in fiscal 2025",
      "represented 82% of our total net revenue for fiscal 2026"]),
    ("TXN", 2025, "What was Texas Instruments' 2025 free cash flow, and how much did the CHIPS Act investment tax credit add to operating cash flow?",
     [usd(2.94e9), usd(335e6), *pct(16.6)],
     ["Free cash flow $2.94 billion, 16.6% of revenue", "CHIPS Act ITC cash benefit of $335 million in 2025 ($588 million in 2024)"],
     ["Free cash flow was $2.94 billion and represented 16.6% of revenue", "include cash benefits of $335 million and $588 million"]),
    ("MU", 2026, "How much did Micron's total revenue and DRAM sales grow in 2026, and what drove the DRAM increase?",
     pct(256, 252),
     ["Total revenue up 256% vs 2025", "DRAM sales up 252%, mainly from an approximately 180% increase in average selling prices"],
     ["Total revenue for 2026 increased 256% as compared to 2025", "Sales of DRAM products increased 252%"]),
    ("QCOM", 2025, "What were Qualcomm's fiscal 2025 revenues and net income, and what does the filing say about future product revenue from Huawei?",
     [usd(44.3e9, 0.01), usd(5.5e9, 0.02)],
     ["Revenues about $44.3 billion and net income about $5.5 billion",
      "Qualcomm does not expect any further product revenues from Huawei after its export license was revoked"],
     ["Revenues were $44.3 billion, an increase of 14% compared to revenues of $39.0 billion in fiscal 2024",
      "we do not expect to receive any further product revenues from Huawei"]),
    ("LRCX", 2026, "What share of Lam Research's fiscal 2026 revenue came from China, and what supply risk does the filing tie to China?",
     pct(34),
     ["China was about 34% of total revenue in fiscal 2026",
      "China is the primary source of certain rare earth elements critical to manufacturing some of its products"],
     ["represented approximately 34%, 34%, and 42% of our total revenue for fiscal years 2026, 2025, and 2024",
      "China is the primary source of supply of certain rare earth elements"]),
    ("AVGO", 2025, "In fiscal 2025, what share of Broadcom's net revenue came from China, and how much came from its largest direct customer?",
     pct(17, 32),
     ["China (including Hong Kong) was 17% of net revenue", "One semiconductor solutions customer, a distributor, was 32%"],
     ["17% and 20%, respectively, of our net revenue came from shipments or deliveries to China",
      "Direct sales to one semiconductor solutions customer, which is a distributor, accounted for 32% and 28%"]),
    ("INTC", 2025, "What stake in Altera did Intel sell in 2025 and for how much, and what share of its 2025 revenue came from China?",
     [usd(4.3e9), *pct(51, 24)],
     ["Sold 51% of Altera to an affiliate of Silver Lake (SLP)", "Closed September 12, 2025 for net consideration of $4.3 billion",
      "China billings were 24% of total revenue"],
     ["we completed the divestiture of 51% of Altera for net purchase consideration of $4.3 billion",
      "with revenue from billings to China contributing 24% of our total revenue"]),
    ("MCHP", 2026, "What happened with Arrow Electronics in Microchip's fiscal 2026 10-K, and how large a share of net sales did Arrow account for?",
     pct(12),
     ["Certain Arrow subsidiaries were placed on the Entity List in October 2025, interrupting shipments; Commerce later authorized transactions and said they'd be removed",
      "No material impact to Microchip", "Arrow, the largest distributor, accounted for 12% of net sales in fiscal 2026"],
     ["certain subsidiaries of Arrow Electronics (Arrow) were placed on the U.S.",
      "which accounted for 12% and 10% of our net sales in fiscal 2026 and in fiscal 2025"]),
]
for i, (t, y, q, nums, kps, quotes) in enumerate(MIXED, 1):
    add(f"v3-mixed-{i:02d}", "mixed_claim", q, [t], [y], numbers=nums, key_points=kps,
        evidence=[ev(t, qt, y) for qt in quotes])

# --- unanswerable: outside the data's years or companies, or never disclosed -------------------
UNANSWERABLE = [
    ("What did NVIDIA's fiscal 2019 10-K say about export controls?", ["NVDA"], "FY2019 is outside the data (NVDA FY2022-FY2026)"),
    ("How did AMD's risk factors about TSMC change between its 2018 and 2019 10-Ks?", ["AMD"], "2018-2019 are outside the data (AMD FY2021-FY2025)"),
    ("What did Intel's 2020 10-K say about Altera?", ["INTC"], "FY2020 is outside the data (INTC FY2021-FY2025)"),
    ("In which year did onsemi's 10-K first mention silicon carbide in its risk factors?", [], "onsemi is not an indexed company"),
    ("How did GlobalFoundries' customer concentration change from fiscal 2022 to fiscal 2025?", [], "GlobalFoundries is not an indexed company"),
    ("In which year did Samsung's annual report first mention HBM4?", [], "Samsung files no 10-K and is not indexed"),
    ("What share of NVIDIA's revenue will come from China in fiscal 2028?", ["NVDA"], "a forecast no filing gives"),
    ("How many H20 units did NVIDIA ship to China in fiscal 2026?", ["NVDA"], "unit volumes are not disclosed"),
    ("How much revenue did Apple earn from OpenAI in fiscal 2025?", ["AAPL"], "not disclosed in Apple's filings"),
    ("What share of Lam Research's revenue came from China in fiscal 2019?", ["LRCX"], "FY2019 is outside the data; earliest figure given is fiscal 2020"),
    ("What share of Broadcom's contract-manufactured wafers did TSMC make in fiscal 2020?", ["AVGO"], "FY2020 is outside the data; the FY2021 filing gives only fiscal 2021"),
    ("What did Qualcomm's fiscal 2026 10-K say about Huawei?", ["QCOM"], "QCOM's latest filing in the data is FY2025"),
]
for i, (q, cos, why) in enumerate(UNANSWERABLE, 1):
    add(f"v3-una-{i:02d}", "unanswerable", q, cos, [], refuse=True, notes=f"unanswerable: {why}. {NOTE}")


# --- verification -----------------------------------------------------------------------------


def verify() -> list[str]:
    errors = []
    for kind, t, phrase, arg in checks:
        if kind in ("first", "drop"):
            m = drafting.mentions(t, phrase)
            y = arg
            ok = (m.get(y, 0) > 0 and (all(v == 0 for fy, v in m.items() if fy < y) if kind == "first"
                                        else all(v == 0 for fy, v in m.items() if fy > y)))
            if not ok:
                errors.append(f"{kind} {t} {phrase!r} FY{y}: {m}")
        else:
            n = sum(drafting.mentions(t, p, arg).get(LATEST[t], 0) for p in phrase)
            if (n > 0) != (kind == "has"):
                errors.append(f"shared {kind} {t} {phrase}: {n} mentions in FY{LATEST[t]}")
    for it in items:
        for e in it["evidence"]:
            if e["_year"] not in drafting.quote_years(e["ticker"], e["quote"]):
                errors.append(f"{it['id']}: quote not in {e['ticker']} FY{e['_year']}: {e['quote'][:80]!r}")
    return errors


def split(rows: list[dict]) -> None:
    """Stratified: in each category, items 0, 2 of every 5 go to dev (40%), the rest to test."""
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for group in by_cat.values():
        for i, r in enumerate(group):
            r["split"] = "dev" if i % 5 in (0, 2) else "test"


if __name__ == "__main__":
    errors = verify()
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} check(s) failed; nothing written")
    split(items)
    for it in items:
        for e in it["evidence"]:
            e.pop("_year")
        GoldenItem.model_validate(it)
    OUT.write_text("".join(json.dumps(it) + "\n" for it in items))
    counts: dict[str, list[int]] = {}
    for it in items:
        c = counts.setdefault(it["category"], [0, 0])
        c[it["split"] == "test"] += 1
    print(f"wrote {len(items)} items to {OUT}")
    for cat, (dev, test) in counts.items():
        print(f"  {cat:15} dev {dev:2}  test {test:2}")
