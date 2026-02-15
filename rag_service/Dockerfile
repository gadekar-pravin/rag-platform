# Multi-stage build for RAG platform service
# Stage 1: Build dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
COPY pyproject.toml ./
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -e "." --no-cache

# Stage 2: Slim runtime
FROM python:3.12-slim-bookworm

RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY rag_service/ /app/rag_service/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8080

CMD ["uvicorn", "rag_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
