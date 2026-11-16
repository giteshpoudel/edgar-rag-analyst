Extraction eval (test split, extractor 990d655db2a7, 22 sampled blocks)

| Metric | Value | 95% CI |
|---|---|---|
| Precision | 50% | 11% to 85% |
| Recall | 38% | 8% to 68% |
| F1 | 43% | 13% to 69% |

8 matched of 16 extracted and 21 labelled (10 would match ignoring claim type).

| Claim type | Labelled | Extracted | Matched | Precision | Recall |
|---|---|---|---|---|---|
| customer_concentration | 0 | 1 | 0 | 0% | n/a |
| supplier_dependence | 0 | 1 | 0 | 0% | n/a |
| export_controls_trade | 7 | 5 | 4 | 80% | 57% |
| geographic_exposure | 0 | 3 | 0 | 0% | n/a |
| segment_change | 2 | 2 | 2 | 100% | 100% |
| legal_regulatory | 10 | 2 | 2 | 100% | 20% |
| cybersecurity_incident | 0 | 2 | 0 | 0% | n/a |
| ai_demand | 2 | 0 | 0 | n/a | 0% |

| Filing | Labelled | Extracted | Matched | Precision | Recall |
|---|---|---|---|---|---|
| AMAT_2025 | 13 | 9 | 6 | 67% | 46% |
| AMAT_2024 | 8 | 7 | 2 | 29% | 25% |
