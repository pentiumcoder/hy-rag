# hy-rag API: FastAPI + embedded Chroma + BM25, models downloaded on first boot.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

COPY data/corpus ./data/corpus
COPY eval ./eval
COPY scripts/seed.py ./scripts/seed.py

RUN useradd --create-home --uid 1000 hyrag \
    && mkdir -p /data /models \
    && chown -R hyrag:hyrag /data /models
USER hyrag

# Indexes, caches and uploads live in /data; embedding models in /models (both volumes).
ENV HYRAG_CORPUS_DIR=/app/data/corpus \
    HYRAG_DATA_DIR=/data \
    HYRAG_MODEL_CACHE_DIR=/models \
    HYRAG_EVAL_RESULTS_DIR=/app/eval/results \
    HYRAG_OLLAMA_BASE_URL=http://ollama:11434

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=10s --start-period=600s --retries=5 \
    CMD python -c "import json, urllib.request as u; assert json.load(u.urlopen('http://127.0.0.1:8000/health'))['status'] == 'ok'"

# Seed (index the corpus, pull missing models), then serve. Both steps are idempotent.
CMD ["sh", "-c", "python scripts/seed.py && exec hyrag serve --host 0.0.0.0 --port 8000"]
