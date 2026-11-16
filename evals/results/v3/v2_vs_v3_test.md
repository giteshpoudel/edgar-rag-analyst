# v2 vs v3 on the v3 test set

## Judge: `deepseek-v4-pro`

47 test questions × 3 runs per system. Differences are v3 − v2 baseline, with a paired bootstrap 95% CI over questions; * marks a CI that excludes zero (a bound within 1e-9 of zero counts as zero).

| | n | v2 baseline | v3 | Difference (95% CI) |
|---|---|---|---|---|
| **Overall** | 47 | 83.0% | 90.8% | +7.8% [+0.0%, +16.3%] |
| unanswerable | 7 | 100.0% | 100.0% | +0.0% [+0.0%, +0.0%] |
| first_appeared | 10 | 93.3% | 93.3% | +0.0% [-13.3%, +13.3%] |
| wording_change | 6 | 55.6% | 72.2% | +16.7% [-11.1%, +55.6%] |
| dropped | 7 | 95.2% | 95.2% | +0.0% [-14.3%, +14.3%] |
| shared_entity | 4 | 100.0% | 100.0% | +0.0% [+0.0%, +0.0%] |
| multi_year | 7 | 47.6% | 90.5% | +42.9% [+23.8%, +66.7%] * |
| mixed_claim | 6 | 88.9% | 83.3% | -5.6% [-16.7%, +0.0%] |

| Run-to-run | Pass rate per run | Range | Questions that flipped between runs |
|---|---|---|---|
| v2 baseline | 87.2%, 80.8%, 80.8% | 80.9%–87.2% | 19.1% |
| v3 | 93.6%, 89.4%, 89.4% | 89.4%–93.6% | 17.0% |

**Where v3 doesn't help** (questions it passed less often than v2 baseline): v3-first-10 (first_appeared, 100%→67%), v3-first-14 (first_appeared, 100%→67%), v3-drop-02 (dropped, 100%→67%), v3-mixed-09 (mixed_claim, 100%→67%), v3-word-07 (wording_change, 33%→0%)
Categories with no gain: unanswerable, first_appeared, dropped, shared_entity, mixed_claim.

## Cost and latency (answering; judge cost separate)

| | Cost per question | Judge cost per question | p50 s | p95 s | Avg prompt tokens | Avg LLM calls | Answers using a v3 tool |
|---|---|---|---|---|---|---|---|
| v2 baseline | $0.0041 | $0.0024 | 8.8 | 21.5 | 24,520 | 3.65 | 0% |
| v3 | $0.0035 | $0.0019 | 8.3 | 21.5 | 24,916 | 3.42 | 61% |

v3 used a v3 tool in: first_appeared 100%, dropped 48%, shared_entity 67%, multi_year 43%, wording_change 56%, mixed_claim 22%, unanswerable 71%.
