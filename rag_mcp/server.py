"""MCP server for RAG platform — exposes rag_search and rag_list_documents tools.

Uses the mcp Python SDK with streamable HTTP transport for VS Code Copilot
integration. Runs as a separate Cloud Run service.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from rag_mcp.config import MCP_PORT
from rag_mcp.tools import extract_bearer_from_context, rag_list_documents, rag_search

F = TypeVar("F", bound=Callable[..., Any])


class _FastMCPServer(Protocol):
    def tool(self) -> Callable[[F], F]: ...

    def run(self, *, transport: str) -> None: ...


if TYPE_CHECKING:

    class Context: ...

    mcp: _FastMCPServer
else:
    try:
        from mcp.server.fastmcp import Context
    except ImportError:  # pragma: no cover - older SDK fallback
        Context = Any

    from mcp.server.fastmcp import FastMCP as _FastMCPRuntime

    mcp = cast(
        _FastMCPServer,
        _FastMCPRuntime(
            name="rag-search",
            instructions=(
                "Search and browse the team's shared document knowledge base. "
                "Use rag_search to find relevant documents by natural language query. "
                "Use rag_list_documents to browse available documents."
            ),
            host="0.0.0.0",
            port=MCP_PORT,
        ),
    )


@mcp.tool()
async def search(query: str, limit: int = 10, ctx: Context | None = None) -> str:
    """Search the team's document knowledge base.

    Args:
        query: Natural language search query describing what you're looking for.
        limit: Maximum number of documents to return (1-50, default 10).
    """
    return await rag_search(query, limit, caller_token=extract_bearer_from_context(ctx))


@mcp.tool()
async def list_documents(limit: int = 20, offset: int = 0, ctx: Context | None = None) -> str:
    """List available documents in the knowledge base.

    Args:
        limit: Maximum number of documents to return (default 20).
        offset: Number of documents to skip for pagination (default 0).
    """
    return await rag_list_documents(limit, offset, caller_token=extract_bearer_from_context(ctx))


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
