"""Configuration for the MCP server."""

from __future__ import annotations

import os

RAG_SERVICE_URL: str = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
RAG_MCP_TOKEN: str | None = os.getenv("RAG_MCP_TOKEN")
MCP_PORT: int = int(os.getenv("MCP_PORT", "8001"))
RAG_MCP_FORWARD_CALLER_TOKEN: bool = os.getenv(
    "RAG_MCP_FORWARD_CALLER_TOKEN", "true"
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
