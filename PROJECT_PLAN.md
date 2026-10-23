# Project Plan: EDGAR RAG Analyst

An AI assistant that answers questions about public-company filings (SEC 10-Ks). It looks up **text** to explain things, calls **tools** to get exact numbers, and is **measured** against a test set so you know how often it's right.

Built with free, open data and open-source tools. The only paid piece is the LLM API.

---

## 1. The core rule

> **Numbers come from tools. Explanations come from documents. Every answer cites its source.**

The AI engineering skills this project shows:

| Skill | What it means here |
|---|---|
| **RAG** | Find the relevant passages in a filing and give them to the LLM so it answers from real text |
| **Vector database** | Store passages as embeddings and search them by meaning |
| **Function calling** | Let the LLM call typed Python functions to get exact figures instead of guessing |
| **Evals** | A test set plus a score report, so changes can be measured instead of eyeballed |

Example questions it should answer:

- "What were Apple's total net sales in FY2023, and how did they change year over year?"
- "What does NVIDIA say about supply-chain risk? Quote the passage."
- "Which Microsoft segment grew fastest last fiscal year? Cite the filing."

---

## 2. How a question flows

```
Question: "What were Apple's FY2023 sales and what risks did they mention?"
        │
        ▼
   LLM decides: needs a NUMBER + needs TEXT
        │                         │
        ▼                         ▼
  Tool: get_fact()          Search vector DB for
  → exact value from XBRL   "risk factors" passages
        │                         │
        └──────────┬──────────────┘
                   ▼
   LLM writes the answer using only that info, with citations
```

---

## 3. Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for AI work |
| LLM | DeepSeek (4.1 Flash) via its OpenAI-compatible API | Cheap and fast; supports function calling. Model name set in `.env` |
| Embeddings | `sentence-transformers` with `BAAI/bge-small-en-v1.5` | Free, runs locally on CPU |
| Vector DB | **LanceDB** | Embedded (runs inside Python): no server or Docker |
| Parsing | `beautifulsoup4` | Turn filing HTML into clean text |
| Data models | `pydantic` | Typed tool inputs/outputs |
| CLI | `typer` | Simple command-line interface |

Config lives in `.env` (gitignored), with a committed `.env.example`:

```
DEEPSEEK_API_KEY=
LLM_MODEL=
SEC_USER_AGENT="Your Name your@email.com"
```

---

## 4. Data (all free, from the SEC)

- **Filings text** (for RAG): 10-K HTML documents from EDGAR.
- **Company facts** (for tools): structured XBRL numbers at `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`.
- **Filing list:** `https://data.sec.gov/submissions/CIK##########.json`.
- The SEC requires a descriptive `User-Agent` header (name + email) and allows at most 10 requests/second. Cache everything locally in `data/`.

Starting scope: **3 companies** (Apple, Microsoft, NVIDIA), latest 10-K each.

---

## 5. The four phases

### Phase 1: Basic RAG
Answer text questions about one filing, with sources.

- Download one Apple 10-K and cache it.
- Clean the HTML and split it into chunks (by section: Item 1, 1A, 7, etc.).
- Embed chunks locally and store them in LanceDB with metadata (company, year, section).
- CLI: ask a question → retrieve the top 5 chunks → LLM answers using only those → print the answer and its sources.

**Done when:** `edgar ask "What are Apple's main risk factors?"` returns a cited answer.
**You learn:** chunking, embeddings, vector search, grounded prompting.

### Phase 2: Number tools
Get exact figures from SEC data instead of the LLM's memory.

- Tools: `get_fact(company, metric, year)` and `compute_change(a, b)`.
- Define each tool with a Pydantic model and pass it to the LLM as a function schema.
- The LLM decides whether to call a tool, search text, or both.
- Print every tool call so you can see what happened.

**Done when:** "What was Apple's FY2023 revenue vs FY2022?" returns numbers that match the SEC data, and shows the tool call.
**You learn:** function calling, tool schemas, a simple agent loop.

### Phase 3: Evals
Measure quality with a score instead of guessing.

- Write ~30 test questions in `evals/golden.jsonl`: text, numeric, and a few unanswerable ones.
- Each has the expected answer and/or the section that should be retrieved.
- Metrics:
  - **Retrieval hit rate (recall@5):** did the right section come back?
  - **Numeric accuracy:** does the number match the SEC data exactly?
  - **Answer quality:** LLM-as-judge score, spot-checked by hand.
  - **Refusal:** does it say "I don't know" on unanswerable questions?
- One command prints a score table.

**Done when:** `edgar eval` prints a report, and the scores are in the README.
**You learn:** evaluation design. This is what makes the project stand out, since most portfolio projects skip it.

### Phase 4: Polish
Make it portfolio-ready.

- Add Microsoft and NVIDIA.
- Refuse cleanly when retrieval finds nothing relevant.
- README: diagram, setup steps, eval table, limitations.
- Record a short demo (terminal GIF or video).

**Done when:** someone can clone the repo, follow the README, and get it running.

---

## 6. Repo layout

```
edgar-rag-analyst/
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml
├── .env.example
├── data/                  # cached filings + LanceDB (gitignored)
├── src/edgar_rag/
│   ├── sec_client.py      # EDGAR downloads with User-Agent + caching
│   ├── chunking.py        # HTML → clean text → section chunks
│   ├── index.py           # embed + store in LanceDB
│   ├── retriever.py       # vector search with metadata filters
│   ├── tools.py           # get_fact, compute_change (Pydantic)
│   ├── agent.py           # LLM loop: decide → tool/retrieve → answer
│   └── cli.py             # `edgar ingest`, `edgar ask`, `edgar eval`
├── evals/
│   ├── golden.jsonl       # test questions + expected answers
│   └── run_evals.py       # score report
└── tests/
```

---

## 7. Later (only after Phases 1–3 work)

- [x] Hybrid search (keyword + vector) and a reranker, with the lift measured by evals. *Done in v2 W2: hybrid is the default (+0.23 recall@5, +4.0 points); the reranker added nothing and stays off.*
- [x] Regression gate: fail CI if eval scores drop. *Done in v2 W6: GitHub Actions eval gate against a measured baseline; main requires it.*
- [x] Async, retries and token/cost tracking. *Done in v2 W4 (OpenTelemetry tracing, cost per question) and W5 (async, parallel tools, backoff).*
- [ ] A small web UI (Streamlit or FastAPI).
- [ ] Local LLM mode (Ollama) to run fully offline.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| SEC rate limits / blocking | `User-Agent` header, throttle requests, cache locally |
| Filings are long and messy | Split by section, not fixed character counts |
| LLM invents numbers | Numbers only from tools; evals check numeric accuracy |
| Weak test set | Start small, label carefully, expand over time |

---

## 9. Interview talking points

- "I built a RAG pipeline end to end: chunking, local embeddings, a vector DB and cited answers."
- "Numbers come from function calls to SEC data, not from the model's memory."
- "I measured it: retrieval hit rate, numeric accuracy, answer quality and refusal rate."
- "I chose LanceDB over pgvector because it's embedded, and I can explain the tradeoff."

---

## 10. References

- SEC EDGAR APIs: https://www.sec.gov/edgar/sec-api-documentation
- SEC fair-access policy: https://www.sec.gov/os/accessing-edgar-data
- sentence-transformers: https://www.sbert.net
- LanceDB: https://lancedb.com
- DeepSeek API docs: https://api-docs.deepseek.com
