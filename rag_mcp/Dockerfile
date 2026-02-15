# Lightweight image for MCP server
FROM python:3.12-slim-bookworm

RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

WORKDIR /app

RUN pip install --no-cache-dir mcp httpx

COPY rag_mcp/ /app/rag_mcp/

ENV PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8001

CMD ["python", "-m", "rag_mcp.server"]
