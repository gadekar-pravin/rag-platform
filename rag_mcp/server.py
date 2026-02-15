"""MCP server for RAG platform — exposes rag_search and rag_list_documents tools.

Uses the mcp Python SDK with streamable HTTP transport for VS Code Copilot
integration. Runs as a separate Cloud Run service.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from rag_mcp.config import MCP_PORT
from rag_mcp.tools import rag_list_documents, rag_search

mcp = FastMCP(
    name="rag-search",
    instructions=(
        "Search and browse the team's shared document knowledge base. "
        "Use rag_search to find relevant documents by natural language query. "
        "Use rag_list_documents to browse available documents."
    ),
)


@mcp.tool()
async def search(query: str, limit: int = 10) -> str:
    """Search the team's document knowledge base.

    Args:
        query: Natural language search query describing what you're looking for.
        limit: Maximum number of documents to return (1-50, default 10).
    """
    return await rag_search(query, limit)


@mcp.tool()
async def list_documents(limit: int = 20, offset: int = 0) -> str:
    """List available documents in the knowledge base.

    Args:
        limit: Maximum number of documents to return (default 20).
        offset: Number of documents to skip for pagination (default 0).
    """
    return await rag_list_documents(limit, offset)


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="streamable-http", port=MCP_PORT)


if __name__ == "__main__":
    main()
