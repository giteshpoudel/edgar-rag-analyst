# edgar-rag-analyst

An AI assistant that answers questions about SEC company filings (10-Ks) with cited sources.

- **RAG** over filing text, using local embeddings and LanceDB
- **Function calling** for exact figures from SEC XBRL data
- **Evals** that score retrieval, numeric accuracy and refusals

> 🚧 Work in progress. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the roadmap.

## Eval results

32 hand-labelled questions about Apple's FY2025 10-K: 12 numeric, 11 narrative, 3 mixed and 6 unanswerable. Expected numbers come from SEC XBRL data; narrative answers are graded against key facts by an LLM judge. The table compares the tool-using **agent** with **simple** RAG (one search, no tools).

| Metric | agent | simple |
|---|---|---|
| Overall pass rate | 100% | 84% |
| Numeric questions | 100% | 92% |
| Narrative questions | 100% | 82% |
| Mixed (numbers + explanation) | 100% | 33% |
| Unanswerable, refused correctly | 100% | 100% |
| Retrieval hit rate | 100% | 93% |
| False refusals on answerable questions | 0% | 8% |
| Avg prompt tokens per question | 5,698 | 2,089 |
| Avg seconds per question | 4.4 | 1.9 |

Run it yourself with `edgar eval`. Full results: [evals/results/summary.md](evals/results/summary.md). The agent scoring 100% means this set is now too easy for it; harder multi-company questions come next.
