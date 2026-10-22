Retrieval ablation (v2, split: test, 65 questions with gold evidence)

| Config | R@5 | R@10 | R@20 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.574 | 0.759 | 0.852 | 0.508 | 13 | 16 |
| hybrid | 0.804 | 0.884 | 0.924 | 0.637 | 17 | 20 |
| hybrid+bge | 0.787 | 0.902 | 0.949 | 0.731 | 1040 | 1164 |
| vector+bge | 0.761 | 0.860 | 0.865 | 0.671 | 1042 | 1172 |

Change vs vector (paired bootstrap 95% CI over questions):

| Config | ΔR@5 | ΔMRR |
|---|---|---|
| hybrid | +0.231 [+0.138, +0.331] * | +0.129 [+0.063, +0.198] * |
| hybrid+bge | +0.213 [+0.079, +0.349] * | +0.223 [+0.111, +0.335] * |
| vector+bge | +0.187 [+0.054, +0.321] * | +0.163 [+0.065, +0.263] * |

* = the CI excludes 0
