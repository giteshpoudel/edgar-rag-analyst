Retrieval ablation (v2, split: dev, 31 questions with gold evidence)

| Config | R@5 | R@10 | R@20 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.813 | 0.877 | 0.948 | 0.712 | 15 | 17 |
| keyword | 0.774 | 0.829 | 0.900 | 0.662 | 5 | 6 |
| hybrid | 0.867 | 0.904 | 0.927 | 0.780 | 18 | 22 |
| hybrid+minilm | 0.808 | 0.872 | 0.927 | 0.741 | 207 | 233 |
| hybrid+bge | 0.861 | 0.949 | 0.966 | 0.813 | 1117 | 1234 |
| vector+bge | 0.895 | 0.949 | 0.966 | 0.802 | 1146 | 1228 |
