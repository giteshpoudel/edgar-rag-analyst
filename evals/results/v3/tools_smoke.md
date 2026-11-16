# v3 tools smoke run (Phase 4)

Six questions through the agent with `EDGAR_TOOLSET=v3` (deepseek-flash, off-peak, 2026-11-12). Not an eval: no grading, no golden answers. It checks that the model picks the new tools, that the citations resolve and that the number guard stays quiet. Cost $0.026 list (24 LLM calls). Full answers: [tools_smoke.json](tools_smoke.json).

| Question | Tools called | Steps | Sources cited | Invalid citations | Unsupported numbers |
|---|---|---|---|---|---|
| How did NVIDIA's risk disclosures about China export controls change from fiscal 2023 to fiscal 2025? | claim_history ×4, search_filings ×2 | 5 | 28 | 0 | 0 |
| Which companies say in their latest 10-K that they depend on TSMC? | find_entity, compare_claims, graph_neighbors, search_filings ×6 | 5 | 20 | 0 | 0 |
| Who are AMD's named suppliers, and since when has each been named? | graph_neighbors, claim_history ×4 | 4 | 15 | 0 | 0 |
| When did Microsoft first mention AI as a driver of demand or investment in its 10-K? | claim_history | 2 | 10 | 0 | 0 |
| Compare what Applied Materials and Lam Research say about export controls in their latest 10-Ks. | compare_claims, claim_history ×4, search_filings ×3 | 6 | 30 | 0 | 0 |
| What was Apple's revenue in fiscal 2025? | get_financial_metric | 2 | 1 | 0 | 0 |

What to notice:
- **Tool choice:** every history question went to `claim_history`, the TSMC question to `find_entity`, `compare_claims` and `graph_neighbors`, and the revenue question stayed on the v2 tool `get_financial_metric`.
- **Paging:** the NVIDIA question needed four `claim_history` calls, because the 20-item cap made the model page by year range. That works, but it costs steps.
- **Coverage caveat:** the Microsoft answer said FY2023 is the first AI mention *in the data* (which starts at FY2022), as the v3 prompt rule asks.
- **Answer length:** answers are long, with 15–30 sources cited on the history questions. Quality isn't graded here; that's Phase 5a's job.
