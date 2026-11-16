Link eval (test split, linker efde3e3b501e)

| View | Items | Correct | Accuracy | 95% CI |
|---|---|---|---|---|
| Given both claims extracted | 7 | 7 | 100% | 100% to 100% |
| End to end | 16 | 7 | 44% | 19% to 69% |

Confusion, given both claims extracted:

| Gold \ predicted | new | changed | unchanged | dropped |
|---|---|---|---|---|
| new | 4 | 0 | 0 | 0 |
| changed | 0 | 1 | 0 | 0 |
| unchanged | 0 | 0 | 1 | 0 |
| dropped | 0 | 0 | 0 | 1 |

Confusion, end to end:

| Gold \ predicted | new | changed | unchanged | dropped | missed |
|---|---|---|---|---|---|
| new | 4 | 0 | 0 | 0 | 2 |
| changed | 0 | 1 | 0 | 0 | 3 |
| unchanged | 0 | 0 | 1 | 0 | 2 |
| dropped | 0 | 0 | 0 | 1 | 2 |

| Pair | Items | Correct | Accuracy |
|---|---|---|---|
| AMAT 2024-2025 | 16 | 7 | 44% |

"wrong prior": linked, but to a different earlier claim. "linked": a gold-dropped claim the linker linked forward. "missed": the extractor didn't produce the claim (or its gold prior).
Dropped items are only as complete as the sampled blocks (see link_eval.py).
