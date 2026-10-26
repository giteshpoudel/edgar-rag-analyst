# Demo API for giteshpoudel.com/demo/edgar (see EDGAR_DEMO_BACKEND_PLAN.md).
# Everything the service needs is baked in at build time: CPU-only torch, the embedding model
# and the frozen data snapshot, so a cold start downloads nothing.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN useradd --create-home --uid 10001 app && chown app /app

# CPU-only torch first: ~200 MB instead of ~2.5 GB of CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server]"

# Model weights and data are fetched as the app user, so it owns its caches.
USER app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
COPY --chown=app scripts/fetch_ci_data.py scripts/
COPY --chown=app evals/ci-data.sha256 evals/
RUN python scripts/fetch_ci_data.py --dest /app

ENV DEMO_MODE=1 \
    RETRIEVAL_MODE=hybrid \
    RERANKER=none \
    TRACING=off \
    GROUNDING_GUARD=on \
    DATA_DIR=/app/data \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LLM_MODEL=deepseek-flash \
    LLM_BASE_URL=https://api.deepseek.com

EXPOSE 8080
# Cloud Run sets PORT. One worker: the model is loaded once and requests share it.
CMD ["sh", "-c", "exec uvicorn edgar_rag.server:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log"]
