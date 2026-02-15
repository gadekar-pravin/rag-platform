"""MCP tool definitions for RAG search and document listing.

Each tool forwards requests to the RAG service HTTP API.
"""

from __future__ import annotations

import httpx

from rag_mcp.config import RAG_MCP_TOKEN, RAG_SERVICE_URL


def _get_headers() -> dict[str, str]:
    """Build auth headers for the RAG service."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if RAG_MCP_TOKEN:
        headers["Authorization"] = f"Bearer {RAG_MCP_TOKEN}"
    return headers


async def rag_search(query: str, limit: int = 10) -> str:
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
            headers=_get_headers(),
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


async def rag_list_documents(limit: int = 20, offset: int = 0) -> str:
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
            headers=_get_headers(),
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
