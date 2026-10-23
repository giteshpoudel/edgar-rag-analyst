# Eval results (2026-10-22)

140 questions · model `deepseek-flash` · judge `deepseek-v4-pro` · agent: 75 s wall time at concurrency 16

| Metric | agent |
|---|---|
| Overall pass rate | 98% |
| Numeric questions | 93% |
| Narrative questions | 96% |
| Mixed questions | n/a |
| Comparison questions (across companies) | n/a |
| Trend questions (across years) | n/a |
| Sector questions (13 semiconductor companies) | n/a |
| Multi-hop (two sections) | 100% |
| Long-tail sections (legal, market risk, controls) | 100% |
| Text-table facts (not in XBRL) | 100% |
| Ambiguous company names | 94% |
| Near-miss unanswerable (refused correctly) | 100% |
| Calculations on text-table figures | 100% |
| Same topic across several companies | 100% |
| Complete enumerations (4+ items) | 100% |
| Unanswerable (refused correctly) | n/a |
| Gold evidence in passages seen | 99% |
| Number accuracy | 100% |
| Retrieval hit rate (right section retrieved) | n/a |
| Retrieval MRR | n/a |
| Key-point coverage (LLM judge) | 99% |
| False refusals on answerable questions | 2% |
| Answers with invalid citations | 0% |
| Unsupported numbers (share of all numbers stated) | 0% |
| Answers with any unsupported number | 0% |
| Grounding repairs triggered | 1 |
| Avg prompt tokens | 8,701 |
| Avg completion tokens | 421 |
| Avg seconds per question | 4.2 |
| p50 seconds per question | 3.8 |
| p95 seconds per question | 8.6 |
| Share of answer time in LLM calls | 99% |
| Share of answer time in retrieval | 2% |
| Avg LLM calls per question | 2.50 |
| Prompt tokens served from cache | 72% |
| Cost per question (USD, list price) | $0.0013 |
| Cost per 1,000 questions (USD) | $1.27 |
| Judge cost per question (USD, eval only) | $0.0009 |
| Truncated answers (hit token limit) | 0 |
| Malformed answers (tool-call markup) | 0 |
| Judge errors (no usable verdict) | 0 |
| Run errors | 0 |
