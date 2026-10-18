# Project Plan v2: measured, gated and observable

v2 turns EDGAR from "works on my eval" into "measured, gated and observable": a harder eval that can still tell good changes from bad, measured retrieval upgrades, and CI that blocks regressions. Target: about 2 weeks part-time.

**Why now:** v1 scores 100% on its own 58 questions, so the eval is saturated and can no longer show whether a change helps. v2 also makes true the capabilities a résumé would claim: reranking, hybrid search, recall@k, a regression gate, tracing and cost observability, and async.

## Success criteria

- A v2 eval set of 150+ questions where the v1 agent scores 70–85%, leaving room to measure gains.
- An ablation table: vector only vs hybrid vs hybrid + reranker, with recall@k, MRR, pass rate, tokens and latency.
- Every number in the final answer, including derived ones in prose, is traced to a tool result and checked.
- A GitHub Actions gate that fails a pull request when pass rate or number accuracy drops beyond a set tolerance.
- Per-run traces with token, cost and latency per tool call, viewable locally.
- README, site project page and résumé bullets updated with the real v2 numbers.

## Where v1 stands

A working agent over 16 companies' latest 10-Ks: LanceDB + bge-small retrieval, XBRL facts, a calculator, and a DeepSeek function-calling loop capped at 6 LLM calls.

| Area | v1 today | v2 gap |
|---|---|---|
| Eval | 58 questions, agent 100% vs plain RAG 60% | Saturated; too small to rank retrieval changes |
| Retrieval | Vector only, ~1,500-char chunks, 200 overlap | No keyword search, no reranker, no ablation |
| Retrieval metric | Hit rate (agent 100%, simple 95%) | No recall@k or MRR per k |
| Numbers | XBRL + calculator tools; 12 annual metrics | Derived numbers in prose not checked; no quarterly or segment data |
| Observability | Token counts printed per run | No traces, cost or per-tool latency |
| CI | None | No tests in CI, no eval gate |
| Runtime | Sync loop | No async, retries or concurrency |

Known v1 limitations that v2 targets: arithmetic sometimes done in prose, annual-only data, and run-to-run variance of a question or two.

## Workstreams

Six workstreams plus one stretch. The harder eval comes first because every other change is judged by it.

### W1. Harder eval set (eval v2)

Build a set that v1 fails often enough to measure progress.

1. Grow to 150–200 questions in `evals/golden_v2.jsonl`, keeping v1's 58 as a regression subset.
2. Add the categories v1 never faced:
   - multi-hop questions that need two or more sections (Risk Factors + MD&A)
   - long-tail sections: Item 3 Legal Proceedings, Item 7A Market Risk, Item 9A Controls
   - facts that live only in text tables, not in XBRL
   - near-miss unanswerable: a company not indexed, a metric not tagged, a future fiscal year
   - ambiguous names and tickers ("Alphabet" vs GOOGL, "Analog" vs ADI)
3. Label each question with gold evidence (for recall@k), expected numbers with a tolerance, and key facts for the judge.
   *Implementation note:* gold evidence is stored as short quotes from the filing, mapped to chunk ids at scoring time, so labels stay valid when chunking changes (W2's chunk-size sweep).
4. Draft candidates with an LLM from sampled chunks, then verify every one by hand. Use a different model from the one answering.
5. Split into dev (tune on it) and test (report only). Never tune on test.
6. Run each config 3 times. Report mean and range, plus a bootstrap 95% CI on pass-rate differences between configs.

**Done when:** v1 scores 70–85% on test, with per-category scores.

### W2. Hybrid search, reranking and ablations

Measure each retrieval upgrade instead of assuming it helps.

1. Add keyword search with LanceDB's built-in full-text index; fuse with vector results by reciprocal rank fusion.
2. Add a local cross-encoder reranker (for example `bge-reranker-base`) over the top 30–50 candidates.
3. Try contextual chunk headers: prepend company, fiscal year and section to each chunk before embedding.
4. Optional: chunk size sweep (800 / 1,500 / 2,500 characters).
5. Report recall@5/10/20, MRR, end-to-end pass rate, tokens and p50/p95 latency for each config.

**Done when:** an ablation table on test picks the default config and states its cost.

### W3. Number grounding checker

Close the known gap where the agent does small arithmetic in prose.

1. Extract every number from the answer, normalizing units and scale ($, B, M, %, bps).
2. Match each against tool outputs in the run's trace, or recompute it (ratio, growth, sum) with the calculator.
3. Use it two ways: as an eval metric (unsupported-number rate) and as a runtime guard that triggers one repair pass.

**Done when:** unsupported-number rate is reported, and is 0 on test with the guard on.

### W4. Tracing and cost

1. Instrument LLM calls, tool calls and retrieval as OpenTelemetry spans.
2. View them locally in an open-source tracer such as Arize Phoenix.
3. Record tokens, cost (from a price table in config) and latency per span.
4. Add cost per question and p50/p95 latency to every eval report.

**Done when:** any eval question opens as a full trace, and the README shows cost per question.

### W5. Async and retries

1. Move to an async LLM client; run parallel tool calls when the model emits several.
2. Run evals concurrently with a semaphore, keeping the SEC limit of 10 requests/second.
3. Add retries with backoff on API and SEC errors.

**Done when:** a full eval run is measurably faster, with the before/after time in the README.

### W6. CI regression gate

1. GitHub Actions on every pull request: ruff, pytest unit tests (offline, no API key).
2. An eval job on a fixed smoke subset (~40 dev questions) against a cached index, using the API key as a repo secret.
3. Compare against a committed baseline file. Fail if pass rate drops more than a set tolerance or number accuracy drops at all.
4. Write the score table to the job summary.
5. Prove it works: open a deliberately broken pull request and show the failed check in the README.

**Done when:** main is protected by the gate and the broken-PR demo is documented.

### W7. Stretch: quarterly and segment data

Quarterly figures are available in the SEC company-facts API already in use. Segment figures are not; they need the filing's XBRL dimensions, which is a bigger parsing job. Do quarterly first; treat segments as a separate follow-up.

## Repo changes

Most work lands in new modules; the agent loop and tools change only for async, tracing and the number guard.

| File | New or changed | Workstream | What it does |
|---|---|---|---|
| `evals/golden_v2.jsonl` | New | W1 | 150–200 labelled questions with gold evidence and a dev/test split field |
| `evals/generate.py` | New | W1 | Drafts candidate questions from sampled chunks for hand review |
| `src/edgar_rag/stats.py` | New | W1 | Repeated runs, bootstrap CIs on pass-rate differences |
| `src/edgar_rag/retriever.py` | Changed | W2 | Vector, keyword and hybrid modes behind one config switch |
| `src/edgar_rag/rerank.py` | New | W2 | Local cross-encoder reranking of top candidates |
| `src/edgar_rag/index.py` | Changed | W2 | Full-text index; optional contextual chunk headers |
| `src/edgar_rag/scoring.py` | Changed | W1, W2, W3 | recall@k, MRR, unsupported-number rate |
| `src/edgar_rag/grounding.py` | New | W3 | Extracts numbers from answers and checks them against the trace |
| `src/edgar_rag/tracing.py` | New | W4 | OpenTelemetry spans, token and cost accounting |
| `src/edgar_rag/agent.py` | Changed | W3, W4, W5 | Async loop, parallel tool calls, spans, repair pass |
| `src/edgar_rag/evaluation.py` | Changed | W4, W5, W6 | Concurrent runner, configs to compare, baseline diff |
| `src/edgar_rag/config.py` | Changed | W2, W4 | Retrieval config and model price table |
| `evals/baseline.json` | New | W6 | Committed scores the CI gate compares against |
| `.github/workflows/ci.yml` | New | W6 | Lint, unit tests, smoke eval gate |
| `tests/` | New | W6 | Offline unit tests for chunking, XBRL parsing, scoring, grounding |
| `src/edgar_rag/xbrl.py` | Changed | W7 | Quarterly facts (stretch) |

## Risks

The main risk is a noisy eval: with LLM variance, a 2–3 point change may be luck, so every comparison needs repeated runs and a confidence interval.

| Risk | Mitigation |
|---|---|
| Eval noise hides real changes | 3 runs per config; bootstrap CI; gate tolerance set from measured run-to-run spread |
| Overfitting to the eval | Tune on dev only; report test once per config |
| Hand-labelling 150+ questions takes too long | LLM drafts, human verifies; cap at 150 if behind schedule |
| CI eval cost and flakiness | Small smoke subset, cheap model, cached index, cost cap per run |
| Reranker too slow on CPU | Rerank top 30 only; report latency; keep it off by default if it does not pay |
| Ablation shows hybrid or reranking does not help | Report it anyway; a measured "no gain" is still a good result |

**Out of scope for v2:** segment-level XBRL, 20-F foreign filers, a new web UI, swapping the LLM provider.

## Deliverables

- [ ] README: v2 eval table, ablation table, cost and latency, the broken-PR demo
- [ ] A short write-up for the site: what the harder eval revealed and which retrieval change paid off
- [ ] Site project page and live demo updated to the v2 default config
- [ ] Résumé bullets rewritten with the real v2 numbers, and stack wording matched to the repo (LanceDB, not pgvector)
- [ ] PROJECT_PLAN.md "Later" items marked done
