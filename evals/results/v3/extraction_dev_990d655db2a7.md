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
