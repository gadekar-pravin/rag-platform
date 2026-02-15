# Lightweight image for MCP server
FROM python:3.12-slim-bookworm

RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

WORKDIR /app

RUN pip install --no-cache-dir "mcp>=1.0.0,<2.0.0" "httpx>=0.27.0,<1.0.0"

COPY rag_mcp/ /app/rag_mcp/

ENV PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/')" || exit 1

CMD ["python", "-m", "rag_mcp.server"]
