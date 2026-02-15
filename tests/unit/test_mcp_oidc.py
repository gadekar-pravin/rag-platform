"""Unit tests for rag_mcp.oidc — OIDC token minting and caching."""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from rag_mcp.oidc import _decode_jwt_exp, _mint_token, clear_cache, get_service_token


def _make_jwt(exp: float, sub: str = "sa@project.iam") -> str:
    """Build a minimal unsigned JWT with the given exp claim."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub, "exp": exp}).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


class TestDecodeJwtExp:
    def test_valid_jwt(self):
        exp = time.time() + 3600
        token = _make_jwt(exp)
        assert _decode_jwt_exp(token) == exp

    def test_missing_exp_raises(self):
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(json.dumps({"sub": "x"}).encode()).rstrip(b"=").decode()
        token = f"{header}.{payload}.sig"
        with pytest.raises(ValueError, match="missing 'exp'"):
            _decode_jwt_exp(token)

    def test_malformed_jwt_raises(self):
        with pytest.raises(ValueError, match="not a valid JWT"):
            _decode_jwt_exp("not-a-jwt")


class TestGetServiceToken:
    def setup_method(self):
        clear_cache()

    def teardown_method(self):
        clear_cache()

    @patch("rag_mcp.oidc._ON_CLOUD_RUN", False)
    def test_returns_none_when_not_on_cloud_run(self):
        assert get_service_token("https://rag.example.com") is None

    @patch("rag_mcp.oidc._ON_CLOUD_RUN", True)
    @patch("rag_mcp.oidc._mint_token")
    def test_mints_and_caches_token(self, mock_mint):
        exp = time.time() + 3600
        token = _make_jwt(exp)
        mock_mint.return_value = token

        result = get_service_token("https://rag.example.com")
        assert result == token
        assert mock_mint.call_count == 1

        # Second call should use cache, not mint again.
        result2 = get_service_token("https://rag.example.com")
        assert result2 == token
        assert mock_mint.call_count == 1

    @patch("rag_mcp.oidc._ON_CLOUD_RUN", True)
    @patch("rag_mcp.oidc._mint_token")
    def test_refreshes_near_expiry(self, mock_mint):
        now = time.time()
        # Token that expires in 2 minutes (within the 5-min refresh margin).
        old_token = _make_jwt(now + 120)
        new_token = _make_jwt(now + 3600)
        mock_mint.side_effect = [old_token, new_token]

        # First call: mints old_token.
        result1 = get_service_token("https://rag.example.com")
        assert result1 == old_token

        # Second call: old_token is within refresh margin, so re-mints.
        result2 = get_service_token("https://rag.example.com")
        assert result2 == new_token
        assert mock_mint.call_count == 2

    @patch("rag_mcp.oidc._ON_CLOUD_RUN", True)
    @patch("rag_mcp.oidc._mint_token")
    def test_stale_fallback_on_mint_failure(self, mock_mint):
        now = time.time()
        # Token expires in 2 minutes — within refresh margin but still valid.
        stale_token = _make_jwt(now + 120)
        mock_mint.side_effect = [stale_token, RuntimeError("metadata server down")]

        result1 = get_service_token("https://rag.example.com")
        assert result1 == stale_token

        # Refresh attempt fails → falls back to stale token.
        result2 = get_service_token("https://rag.example.com")
        assert result2 == stale_token

    @patch("rag_mcp.oidc._ON_CLOUD_RUN", True)
    @patch("rag_mcp.oidc._mint_token")
    def test_returns_none_when_truly_expired_and_mint_fails(self, mock_mint):
        now = time.time()
        # Token already expired.
        expired_token = _make_jwt(now - 10)
        mock_mint.side_effect = [expired_token, RuntimeError("metadata server down")]

        # First call: mints (expired, but still stored in cache).
        _ = get_service_token("https://rag.example.com")

        # Second call: refresh fails, stale token is truly expired → None.
        result = get_service_token("https://rag.example.com")
        assert result is None

    @patch("rag_mcp.oidc._ON_CLOUD_RUN", True)
    @patch("rag_mcp.oidc._mint_token")
    def test_returns_none_on_first_mint_failure(self, mock_mint):
        mock_mint.side_effect = RuntimeError("no metadata server")
        assert get_service_token("https://rag.example.com") is None


class TestUrllibParseImport:
    """Verify that urllib.parse is explicitly imported (Fix 1)."""

    @patch("rag_mcp.oidc.urllib.request.urlopen")
    def test_mint_token_encodes_audience(self, mock_urlopen):
        """_mint_token URL-encodes the audience parameter."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_jwt(time.time() + 3600).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _mint_token("https://rag.example.com/path?q=1")

        # Verify the URL was constructed with encoded audience
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "https%3A%2F%2Frag.example.com" in req.full_url
