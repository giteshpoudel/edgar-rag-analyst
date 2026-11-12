Link eval (dev split, linker efde3e3b501e)

| View | Items | Correct | Accuracy | 95% CI |
|---|---|---|---|---|
| Given both claims extracted | 17 | 15 | 88% | 71% to 100% |
| End to end | 21 | 15 | 71% | 52% to 90% |

Confusion, given both claims extracted:

| Gold \ predicted | new | changed | unchanged | dropped |
|---|---|---|---|---|
| new | 4 | 1 | 0 | 0 |
| changed | 0 | 5 | 0 | 0 |
| unchanged | 0 | 1 | 5 | 0 |
| dropped | 0 | 0 | 0 | 1 |

Confusion, end to end:

| Gold \ predicted | new | changed | unchanged | dropped | missed |
|---|---|---|---|---|---|
| new | 4 | 1 | 0 | 0 | 0 |
| changed | 0 | 5 | 0 | 0 | 3 |
| unchanged | 0 | 1 | 5 | 0 | 1 |
| dropped | 0 | 0 | 0 | 1 | 0 |

| Pair | Items | Correct | Accuracy |
|---|---|---|---|
| NVDA 2025-2026 | 20 | 15 | 75% |
| AAPL 2024-2025 | 1 | 0 | 0% |

"wrong prior": linked, but to a different earlier claim. "linked": a gold-dropped claim the linker linked forward. "missed": the extractor didn't produce the claim (or its gold prior).
Dropped items are only as complete as the sampled blocks (see link_eval.py).
