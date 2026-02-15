"""MCP tool definitions for RAG search and document listing.

Each tool forwards requests to the RAG service HTTP API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from rag_mcp.config import RAG_MCP_FORWARD_CALLER_TOKEN, RAG_MCP_TOKEN, RAG_SERVICE_URL
from rag_mcp.oidc import get_service_token

logger = logging.getLogger(__name__)


def _get_headers(caller_token: str | None = None) -> dict[str, str]:
    """Build auth headers for the RAG service.

    Token priority:
    1. OIDC service token (Cloud Run — minted via service account)
    2. Caller-forwarded token (when RAG_MCP_FORWARD_CALLER_TOKEN is true)
    3. Static RAG_MCP_TOKEN (local dev fallback)
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}

    # 1. OIDC service token (Cloud Run service-to-service)
    oidc_token = get_service_token(RAG_SERVICE_URL)
    if oidc_token:
        headers["Authorization"] = f"Bearer {oidc_token}"
        return headers

    # 2. Caller-forwarded token or 3. Static token
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


_MAX_RETRIES = 2
_RETRY_BACKOFF_BASE = 0.5
_RETRYABLE_STATUS = {502, 503, 504}


async def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """Make an HTTP request with retry on transient failures."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.request(method, url, headers=headers, json=json, params=params)
            if resp.status_code not in _RETRYABLE_STATUS or attempt >= _MAX_RETRIES:
                return resp
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt >= _MAX_RETRIES:
                raise
            logger.warning(
                "HTTP request failed (attempt %d/%d): %s",
                attempt + 1,
                _MAX_RETRIES + 1,
                e,
            )
        await asyncio.sleep(_RETRY_BACKOFF_BASE * (2**attempt))
    raise RuntimeError("Unreachable retry path")


def _sanitize_error(status_code: int) -> str:
    """Return a user-safe error message without leaking internal details."""
    if status_code == 401:
        return "Authentication failed. Please check your credentials."
    if status_code == 403:
        return "Access denied."
    if status_code == 429:
        return "Rate limit exceeded. Please try again later."
    if status_code >= 500:
        return "The search service is temporarily unavailable. Please try again later."
    return f"Request failed with status {status_code}."


async def rag_search(query: str, limit: int = 10, caller_token: str | None = None) -> str:
    """Search the team's document knowledge base.

    Args:
        query: Natural language search query.
        limit: Maximum number of documents to return (1-50).

    Returns:
        Formatted search results with document titles and relevant chunks.
    """
    try:
        resp = await _request_with_retry(
            "POST",
            f"{RAG_SERVICE_URL}/v1/search",
            headers=_get_headers(caller_token),
            json={"query": query, "limit": limit},
        )
    except (httpx.ConnectError, httpx.ReadTimeout):
        return "The search service is temporarily unavailable. Please try again later."

    if resp.status_code != 200:
        return _sanitize_error(resp.status_code)

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
    try:
        resp = await _request_with_retry(
            "GET",
            f"{RAG_SERVICE_URL}/v1/documents",
            headers=_get_headers(caller_token),
            params={"limit": limit, "offset": offset},
        )
    except (httpx.ConnectError, httpx.ReadTimeout):
        return "The search service is temporarily unavailable. Please try again later."

    if resp.status_code != 200:
        return _sanitize_error(resp.status_code)

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
