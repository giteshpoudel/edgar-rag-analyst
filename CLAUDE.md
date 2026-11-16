# edgar-rag-analyst: working rules

An agent that answers questions about SEC 10-K filings, with citations, measured by evals. Python 3.11, DeepSeek through an OpenAI-compatible API, local embeddings, LanceDB. Plans: [PROJECT_PLAN.md](PROJECT_PLAN.md) (v1), [PROJECT_PLAN_V2.md](PROJECT_PLAN_V2.md) (v2, done), [PROJECT_PLAN_v3.md](PROJECT_PLAN_v3.md) (v3, current).

## Always

- **Secrets:** `.env` holds the API keys. Never commit it, upload it, print it or put it in logs, traces or the build log. The owner sets secrets (locally and in GitHub); never enter keys yourself.
- **Git:**
  - Commit or push only when asked, and add no AI attribution lines to commits or PRs.
  - `main` is protected: changes go through a PR, and both checks ("Lint and unit tests", "Eval gate") must pass.
- **Numbers come from tools, explanations from documents, and every answer cites its source.** Keep that rule in any new tool: typed Pydantic arguments and outputs, citable results, capped lists.
- **Explain while building:** the owner is learning AI engineering, so say what each step does and why.
- **Build log:** after each phase, update the build log with the `project-build-log` skill (what was built, decisions, bugs, results, spend). Keep the portfolio kit to results merged to main, with measured numbers only.

## v2 is the record

- Don't restructure or re-run finished v2 work. Leave v2's LanceDB table, the `ci-data-v1` snapshot, `evals/baseline.json` and the default toolset as they are.
- If a v2 number looks wrong, flag it to the owner. Never overwrite it.
- The v2 eval gate must pass on every PR.

## v3 rules

- **Budget:**
  - $10 total for API calls and $0 for GPUs; embeddings stay local on CPU.
  - Every LLM call goes through the cost ledger, and the hard cap is `EDGAR_BUDGET_USD`. A run stops before a call that would cross the cap; never bypass the ledger.
  - Cache extraction by filing hash and prompt hash, so a re-run doesn't pay again.
  - Extract with `deepseek-flash` in DeepSeek's off-peak window.
  - **Before any run expected to cost over $1, give the owner the estimate and wait for a go-ahead.**
  - Iterate on dev with single runs, and do 3 runs only for final test numbers.
- **Claims are append-only.**
  - Never update or delete a stored claim or link. Re-extraction adds rows under a new extractor version.
  - Every claim keeps company, fiscal year, section, an exact source quote (verified verbatim against the filing text) and a filing link.
- **Tune on dev only.**
  - Test sets (extraction labels and v3 questions) are scored once per frozen version, and every test score is reported.
  - Don't draft eval questions from the system's own extracted claims.
- **Judge:** use a judge from a provider other than DeepSeek, calibrated against the owner's hand grades (agreement, Cohen's kappa). Keep the `deepseek-v4-pro` scores alongside so v2 numbers stay comparable.
- **Failing evals:** investigate the system before changing a threshold, and tell the owner about any threshold change.
- **Working by phases:** at the start of each phase, give a short plan with a cost estimate and wait for the go-ahead.
- **Storage:** claims and graph go in plain SQLite tables, with no graph database. The v3 tools sit behind `EDGAR_TOOLSET=v3` until a decision to make them the default.
- **Fiscal years** are named by the calendar year the fiscal year ends in, as in v2.
- **Out of scope:** quarterly or segment XBRL (W7), 20-F filers, UI beyond what the demo needs, paid GPUs.
