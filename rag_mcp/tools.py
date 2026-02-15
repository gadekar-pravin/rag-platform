"""MCP tool definitions for RAG search and document listing.

Each tool forwards requests to the RAG service HTTP API.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rag_mcp.config import RAG_MCP_FORWARD_CALLER_TOKEN, RAG_MCP_TOKEN, RAG_SERVICE_URL

logger = logging.getLogger(__name__)


def _get_headers(caller_token: str | None = None) -> dict[str, str]:
    """Build auth headers for the RAG service."""
    headers: dict[str, str] = {"Content-Type": "application/json"}

    token = caller_token if (RAG_MCP_FORWARD_CALLER_TOKEN and caller_token) else RAG_MCP_TOKEN
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def extract_bearer_from_context(ctx: Any) -> str | None:
    """Best-effort extraction of Authorization bearer token from FastMCP context."""
    if ctx is None:
        return None

    request_context = getattr(ctx, "request_context", None)
    request = getattr(request_context, "request", None)
    headers = getattr(request, "headers", None)

    auth_header = None
    if headers is not None and (hasattr(headers, "get") or isinstance(headers, dict)):
        auth_header = headers.get("authorization") or headers.get("Authorization")

    if not auth_header:
        metadata = getattr(request_context, "meta", None)
        if isinstance(metadata, dict):
            auth_header = metadata.get("authorization") or metadata.get("Authorization")

    if not isinstance(auth_header, str):
        return None

    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    logger.warning("Authorization header found in MCP context but not in Bearer format")
    return None


async def rag_search(query: str, limit: int = 10, caller_token: str | None = None) -> str:
    """Search the team's document knowledge base.

    Args:
        query: Natural language search query.
        limit: Maximum number of documents to return (1-50).

    Returns:
        Formatted search results with document titles and relevant chunks.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{RAG_SERVICE_URL}/v1/search",
            json={"query": query, "limit": limit},
            headers=_get_headers(caller_token),
        )

    if resp.status_code != 200:
        return f"Search failed: {resp.status_code} {resp.text}"

    data = resp.json()
    results = data.get("results", [])

    if not results:
        return "No documents found matching your query."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        doc_type = r.get("doc_type", "")
        score = r.get("rrf_score", 0)
        lines.append(f"\n## {i}. {title}" + (f" ({doc_type})" if doc_type else ""))
        lines.append(f"Score: {score:.4f}")

        chunks = r.get("chunks", [])
        for c in chunks:
            source = c.get("source", "")
            chunk_text = c.get("chunk_text", "")
            # Truncate long chunks for readability
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            lines.append(f"\n**[{source}]** {chunk_text}")

    return "\n".join(lines)


async def rag_list_documents(
    limit: int = 20,
    offset: int = 0,
    caller_token: str | None = None,
) -> str:
    """List available documents in the knowledge base.

    Args:
        limit: Maximum number of documents to return.
        offset: Number of documents to skip (for pagination).

    Returns:
        Formatted list of documents with titles and metadata.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{RAG_SERVICE_URL}/v1/documents",
            params={"limit": limit, "offset": offset},
            headers=_get_headers(caller_token),
        )

    if resp.status_code != 200:
        return f"List failed: {resp.status_code} {resp.text}"

    data = resp.json()
    documents = data.get("documents", [])
    total = data.get("total", 0)

    if not documents:
        return "No documents found in the knowledge base."

    lines: list[str] = [f"Found {total} documents (showing {len(documents)}):"]
    for d in documents:
        title = d.get("title", "Untitled")
        doc_type = d.get("doc_type", "")
        visibility = d.get("visibility", "TEAM")
        chunks = d.get("total_chunks", 0)
        type_str = f" [{doc_type}]" if doc_type else ""
        lines.append(f"- {title}{type_str} ({visibility}, {chunks} chunks)")

    return "\n".join(lines)
