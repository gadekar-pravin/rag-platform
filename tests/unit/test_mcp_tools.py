"""Unit tests for MCP token handling and request-context passthrough."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from rag_mcp.tools import _get_headers, extract_bearer_from_context


def _ctx_with_auth(auth_value: str | None):
    headers = {"Authorization": auth_value} if auth_value is not None else {}
    request = SimpleNamespace(headers=headers)
    request_context = SimpleNamespace(request=request, meta={})
    return SimpleNamespace(request_context=request_context)


class TestTokenHeaderSelection:
    @patch("rag_mcp.tools.RAG_MCP_FORWARD_CALLER_TOKEN", True)
    @patch("rag_mcp.tools.RAG_MCP_TOKEN", "service-token")
    def test_prefers_caller_token_when_forwarding_enabled(self):
        headers = _get_headers(caller_token="caller-token")
        assert headers["Authorization"] == "Bearer caller-token"

    @patch("rag_mcp.tools.RAG_MCP_FORWARD_CALLER_TOKEN", False)
    @patch("rag_mcp.tools.RAG_MCP_TOKEN", "service-token")
    def test_uses_service_token_when_forwarding_disabled(self):
        headers = _get_headers(caller_token="caller-token")
        assert headers["Authorization"] == "Bearer service-token"


class TestOidcServiceToken:
    """OIDC service token takes priority over caller/static tokens."""

    @patch("rag_mcp.tools.get_service_token", return_value="oidc-token-123")
    @patch("rag_mcp.tools.RAG_MCP_FORWARD_CALLER_TOKEN", True)
    @patch("rag_mcp.tools.RAG_MCP_TOKEN", "static-token")
    def test_oidc_token_takes_priority(self, _mock_oidc):
        headers = _get_headers(caller_token="caller-token")
        assert headers["Authorization"] == "Bearer oidc-token-123"

    @patch("rag_mcp.tools.get_service_token", return_value=None)
    @patch("rag_mcp.tools.RAG_MCP_FORWARD_CALLER_TOKEN", True)
    @patch("rag_mcp.tools.RAG_MCP_TOKEN", "static-token")
    def test_falls_back_to_caller_when_oidc_none(self, _mock_oidc):
        headers = _get_headers(caller_token="caller-token")
        assert headers["Authorization"] == "Bearer caller-token"

    @patch("rag_mcp.tools.get_service_token", return_value=None)
    @patch("rag_mcp.tools.RAG_MCP_FORWARD_CALLER_TOKEN", False)
    @patch("rag_mcp.tools.RAG_MCP_TOKEN", "static-token")
    def test_falls_back_to_static_when_oidc_none_and_forward_off(self, _mock_oidc):
        headers = _get_headers(caller_token="caller-token")
        assert headers["Authorization"] == "Bearer static-token"


class TestContextExtraction:
    def test_extracts_bearer_from_request_headers(self):
        ctx = _ctx_with_auth("Bearer abc123")
        assert extract_bearer_from_context(ctx) == "abc123"

    def test_returns_none_when_no_token(self):
        ctx = _ctx_with_auth(None)
        assert extract_bearer_from_context(ctx) is None
