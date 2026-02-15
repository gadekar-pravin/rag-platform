"""Unit tests for auth module — OIDC verification, shared token safety."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from rag_service.auth import Identity, get_identity, is_public_path


def _make_request(
    auth_header: str | None = None,
    query_token: str | None = None,
    path: str = "/v1/search",
) -> MagicMock:
    """Create a mock FastAPI Request."""
    request = MagicMock()
    request.url.path = path
    request.headers = {}
    if auth_header:
        request.headers["authorization"] = auth_header
    request.query_params = {}
    if query_token:
        request.query_params["token"] = query_token
    return request


class TestPublicPaths:
    def test_liveness_is_public(self):
        assert is_public_path("/liveness")

    def test_readiness_is_public(self):
        assert is_public_path("/readiness")

    def test_docs_is_public(self):
        assert is_public_path("/docs")

    def test_search_is_not_public(self):
        assert not is_public_path("/v1/search")

    def test_documents_is_not_public(self):
        assert not is_public_path("/v1/documents")


class TestGetIdentity:
    async def test_missing_token_raises_401(self):
        """Request without any token raises 401."""
        request = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            await get_identity(request)
        assert exc_info.value.status_code == 401

    @patch("rag_service.auth.IS_CLOUD_RUN", False)
    @patch("rag_service.auth.RAG_SHARED_TOKEN", "test-secret-token")
    @patch("rag_service.auth.TENANT_ID", "test-tenant")
    async def test_shared_token_dev_mode(self):
        """Shared token works in local dev (not Cloud Run)."""
        request = _make_request(auth_header="Bearer test-secret-token")
        identity = await get_identity(request)
        assert identity.tenant_id == "test-tenant"
        assert identity.user_id == "dev-user"

    @patch("rag_service.auth.IS_CLOUD_RUN", True)
    @patch("rag_service.auth.RAG_SHARED_TOKEN", "test-secret-token")
    @patch("rag_service.auth.id_token")
    async def test_shared_token_rejected_on_cloud_run(self, mock_id_token):
        """Shared token is NOT accepted on Cloud Run — must use OIDC."""
        mock_id_token.verify_token.side_effect = Exception("Invalid token")
        request = _make_request(auth_header="Bearer test-secret-token")
        with pytest.raises(HTTPException) as exc_info:
            await get_identity(request)
        assert exc_info.value.status_code == 401

    @patch("rag_service.auth.IS_CLOUD_RUN", False)
    @patch("rag_service.auth.RAG_SHARED_TOKEN", None)
    @patch("rag_service.auth.id_token")
    @patch("rag_service.auth.TENANT_ID", "my-tenant")
    async def test_oidc_token_verification(self, mock_id_token):
        """Valid OIDC token extracts email as user_id."""
        mock_id_token.verify_token.return_value = {
            "email": "user@company.com",
            "sub": "12345",
            "iss": "https://accounts.google.com",
        }
        request = _make_request(auth_header="Bearer valid-oidc-token")
        identity = await get_identity(request)
        assert identity.tenant_id == "my-tenant"
        assert identity.user_id == "user@company.com"
        assert identity.principal == "user@company.com"
        mock_id_token.verify_token.assert_called_once()

    @patch("rag_service.auth.IS_CLOUD_RUN", False)
    @patch("rag_service.auth.RAG_SHARED_TOKEN", None)
    @patch("rag_service.auth.id_token")
    async def test_oidc_falls_back_to_sub(self, mock_id_token):
        """If no email claim, falls back to sub claim."""
        mock_id_token.verify_token.return_value = {
            "sub": "12345",
            "iss": "https://accounts.google.com",
        }
        request = _make_request(auth_header="Bearer valid-oidc-token")
        identity = await get_identity(request)
        assert identity.user_id == "12345"

    @patch("rag_service.auth.IS_CLOUD_RUN", False)
    @patch("rag_service.auth.RAG_SHARED_TOKEN", None)
    @patch("rag_service.auth.RAG_OIDC_AUDIENCE", "https://rag-service.example.com")
    @patch("rag_service.auth.id_token")
    async def test_oidc_audience_passed_to_verifier(self, mock_id_token):
        mock_id_token.verify_token.return_value = {
            "sub": "12345",
            "iss": "https://accounts.google.com",
        }
        request = _make_request(auth_header="Bearer valid-oidc-token")

        await get_identity(request)

        _, kwargs = mock_id_token.verify_token.call_args
        assert kwargs["audience"] == "https://rag-service.example.com"

    @patch("rag_service.auth.IS_CLOUD_RUN", False)
    @patch("rag_service.auth.RAG_SHARED_TOKEN", None)
    @patch("rag_service.auth.RAG_TENANT_CLAIM", "tenant")
    @patch("rag_service.auth.id_token")
    async def test_tenant_claim_used_when_available(self, mock_id_token):
        mock_id_token.verify_token.return_value = {
            "sub": "12345",
            "tenant": "tenant-from-claim",
            "iss": "https://accounts.google.com",
        }
        request = _make_request(auth_header="Bearer valid-oidc-token")

        identity = await get_identity(request)
        assert identity.tenant_id == "tenant-from-claim"

    async def test_query_param_token_extraction(self):
        """Token can be extracted from query parameter."""
        request = _make_request(query_token="my-token")
        # Will fail at verification, but tests extraction path
        with pytest.raises(HTTPException):
            await get_identity(request)


class TestIdentityDataclass:
    def test_identity_fields(self):
        identity = Identity(
            tenant_id="t1",
            user_id="u1",
            principal="u1@test.com",
        )
        assert identity.tenant_id == "t1"
        assert identity.user_id == "u1"
        assert identity.principal == "u1@test.com"
