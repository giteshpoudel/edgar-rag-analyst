# Judge calibration (v3 Phase 6)

80 answers (60 natural, 20 perturbed with a known error), 119 key points. Reference grades are Claude's, graded blind; agreement with them is agreement with an independent model, not with a human expert.

| Judge vs | Unit | n | Agreement | Cohen's kappa (95% CI) | Judge 'true' rate | Reference 'true' rate |
|---|---|---|---|---|---|---|
| `deepseek-v4-pro` vs reference | key point | 116 | 97% | 0.89 (0.78 to 0.98) | 78% | 82% |
| `deepseek-v4-pro` vs reference | verdict | 79 | 96% | 0.90 (0.79 to 1.00) | 71% | 75% |

| Judge | Verdict agreement, natural | Verdict agreement, perturbed | Perturbed answers failed | Judge errors | Disagreements with reference |
|---|---|---|---|---|---|
| `deepseek-v4-pro` | 97% | 95% | 100% | 1 | C10, C55, P14 |
