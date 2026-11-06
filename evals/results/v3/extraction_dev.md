# Extraction eval, dev split (labels: Claude model labels, no human check)

## v1: extractor e6361d5682b5 (original prompt)

Extraction eval (dev split, extractor e6361d5682b5, 48 sampled blocks)

| Metric | Value | 95% CI |
|---|---|---|
| Precision | 34% | 15% to 62% |
| Recall | 71% | 57% to 84% |
| F1 | 46% | 24% to 68% |

22 matched of 64 extracted and 31 labelled (23 would match ignoring claim type).

| Claim type | Labelled | Extracted | Matched | Precision | Recall |
|---|---|---|---|---|---|
| customer_concentration | 6 | 8 | 5 | 62% | 83% |
| supplier_dependence | 2 | 2 | 0 | 0% | 0% |
| export_controls_trade | 2 | 7 | 2 | 29% | 100% |
| geographic_exposure | 2 | 5 | 2 | 40% | 100% |
| segment_change | 0 | 2 | 0 | 0% | n/a |
| strategic_partnership | 1 | 1 | 1 | 100% | 100% |
| capacity_investment | 1 | 6 | 1 | 17% | 100% |
| legal_regulatory | 17 | 18 | 11 | 61% | 65% |
| cybersecurity_incident | 0 | 4 | 0 | 0% | n/a |
| ai_demand | 0 | 10 | 0 | 0% | n/a |
| competition | 0 | 1 | 0 | 0% | n/a |

| Filing | Labelled | Extracted | Matched | Precision | Recall |
|---|---|---|---|---|---|
| NVDA_2026 | 19 | 38 | 15 | 39% | 79% |
| NVDA_2025 | 10 | 13 | 7 | 54% | 70% |
| AAPL_2025 | 1 | 5 | 0 | 0% | 0% |
| AAPL_2024 | 1 | 8 | 0 | 0% | 0% |


## v2: extractor 990d655db2a7 (anchor and type limits added to the prompt)

Extraction eval (dev split, extractor 990d655db2a7, 48 sampled blocks)

| Metric | Value | 95% CI |
|---|---|---|
| Precision | 65% | 40% to 88% |
| Recall | 77% | 64% to 90% |
| F1 | 71% | 50% to 86% |

24 matched of 37 extracted and 31 labelled (25 would match ignoring claim type).

| Claim type | Labelled | Extracted | Matched | Precision | Recall |
|---|---|---|---|---|---|
| customer_concentration | 6 | 7 | 6 | 86% | 100% |
| supplier_dependence | 2 | 0 | 0 | n/a | 0% |
| export_controls_trade | 2 | 4 | 2 | 50% | 100% |
| geographic_exposure | 2 | 2 | 2 | 100% | 100% |
| strategic_partnership | 1 | 1 | 1 | 100% | 100% |
| capacity_investment | 1 | 2 | 1 | 50% | 100% |
| legal_regulatory | 17 | 16 | 12 | 75% | 71% |
| ai_demand | 0 | 5 | 0 | 0% | n/a |

| Filing | Labelled | Extracted | Matched | Precision | Recall |
|---|---|---|---|---|---|
| NVDA_2026 | 19 | 22 | 16 | 73% | 84% |
| NVDA_2025 | 10 | 12 | 8 | 67% | 80% |
| AAPL_2025 | 1 | 0 | 0 | n/a | 0% |
| AAPL_2024 | 1 | 3 | 0 | 0% | 0% |

