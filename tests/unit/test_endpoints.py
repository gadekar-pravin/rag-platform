"""Unit tests for FastAPI endpoints in the RAG service.

Tests cover:
- POST /v1/index (successful indexing, blank content, dedup)
- GET /v1/documents (list with pagination, limit clamping)
- DELETE /v1/documents/{id} (success, 404 for non-existent)
- POST /v1/search (successful search, blank query)
- GET /liveness and GET /readiness (health endpoints)
- Auth middleware (missing token returns 401)
- Body size middleware (oversized content-length returns 413)
- Request ID middleware (echo and auto-generate)

Uses httpx.AsyncClient with ASGITransport for testing the FastAPI app.
All external dependencies (DB, embedding, stores) are mocked.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from rag_service.auth import Identity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_IDENTITY = Identity(
    tenant_id="test-tenant",
    user_id="test-user",
    principal="test-user@example.com",
)

TEST_TOKEN = "test-dev-token"


@asynccontextmanager
async def _fake_rls_connection(tenant_id: str, user_id: str):
    """Fake rls_connection that yields a MagicMock connection."""
    yield MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_lifespan():
    """Patch get_pool / close_pool so the lifespan does not touch the DB."""
    with (
        patch("rag_service.app.get_pool", new_callable=AsyncMock) as mock_pool,
        patch("rag_service.app.close_pool", new_callable=AsyncMock),
        patch("rag_service.app.require_auth_on_cloud_run"),
        patch("rag_service.app.setup_logging"),
    ):
        mock_pool.return_value = MagicMock()
        yield


@pytest.fixture()
def _mock_auth():
    """Patch get_identity in auth middleware to accept TEST_TOKEN."""

    async def _patched_get_identity(request: Any) -> Identity:
        from rag_service.auth import _extract_token

        token = _extract_token(request)
        if token == TEST_TOKEN:
            return TEST_IDENTITY
        if not token:
            raise HTTPException(status_code=401, detail="Missing authorization token")
        raise HTTPException(status_code=401, detail="Invalid token")

    return patch("rag_service.app.get_identity", side_effect=_patched_get_identity)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Standard auth header using the test token."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture()
async def client(_mock_lifespan, _mock_auth):
    """Async httpx client wired to the FastAPI app with mocked deps."""
    with _mock_auth:
        from rag_service.app import app

        # Reset rate limiter state between tests to avoid cross-test interference
        app.state.limiter.reset()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


class TestLiveness:
    async def test_liveness_returns_ok(self, client: AsyncClient):
        resp = await client.get("/liveness")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    async def test_liveness_does_not_require_auth(self, client: AsyncClient):
        """Liveness is a public path -- no auth header needed."""
        resp = await client.get("/liveness")
        assert resp.status_code == 200


class TestReadiness:
    @patch(
        "rag_service.app.check_embedding_service",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch(
        "rag_service.app.check_db_connection", new_callable=AsyncMock, return_value=True
    )
    async def test_readiness_ok(self, mock_db, mock_emb, client: AsyncClient):
        resp = await client.get("/readiness")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @patch(
        "rag_service.app.check_embedding_service",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch(
        "rag_service.app.check_db_connection",
        new_callable=AsyncMock,
        return_value=False,
    )
    async def test_readiness_db_down(self, mock_db, mock_emb, client: AsyncClient):
        resp = await client.get("/readiness")
        assert resp.status_code == 503
        assert "Database unavailable" in resp.json()["detail"]

    @patch(
        "rag_service.app.check_embedding_service",
        new_callable=AsyncMock,
        return_value=False,
    )
    @patch(
        "rag_service.app.check_db_connection", new_callable=AsyncMock, return_value=True
    )
    async def test_readiness_embedding_down(
        self, mock_db, mock_emb, client: AsyncClient
    ):
        """When DB is OK but embedding is down, status is degraded (not 503)."""
        resp = await client.get("/readiness")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert "Embedding service unavailable" in body["error"]


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    async def test_missing_token_returns_401(self, client: AsyncClient):
        """Requests to protected endpoints without a token get 401."""
        resp = await client.get("/v1/documents")
        assert resp.status_code == 401
        assert "x-request-id" in resp.headers
        assert "Missing authorization token" in resp.json()["detail"]

    async def test_invalid_token_returns_401(self, client: AsyncClient):
        resp = await client.get(
            "/v1/documents",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        assert "x-request-id" in resp.headers
        assert "Invalid token" in resp.json()["detail"]

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_valid_token_passes(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Valid token passes auth middleware and reaches the endpoint."""
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.list_documents = AsyncMock(return_value=([], 0))
            resp = await client.get("/v1/documents", headers=auth_headers)
        assert resp.status_code == 200

    async def test_public_paths_skip_auth(self, client: AsyncClient):
        """Public endpoints do not require authentication."""
        for path in ("/liveness", "/readiness", "/docs", "/openapi.json"):
            resp = await client.get(path)
            # Should not be 401 (readiness may be 503 if DB mock is missing, but not 401)
            assert resp.status_code != 401, f"{path} should skip auth"


# ---------------------------------------------------------------------------
# Body size middleware
# ---------------------------------------------------------------------------


class TestBodySizeMiddleware:
    async def test_oversized_content_length_returns_413(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Content-Length exceeding 10 MB triggers 413."""
        headers = {**auth_headers, "Content-Length": str(11 * 1024 * 1024)}
        resp = await client.post(
            "/v1/index",
            content=b"{}",
            headers=headers,
        )
        assert resp.status_code == 413

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_normal_content_length_passes(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Small Content-Length should not be rejected by size middleware."""
        # This will be rejected by validation (missing fields), but NOT by size middleware.
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            resp = await client.post(
                "/v1/index",
                json={"title": "T", "content": "C"},
                headers=auth_headers,
            )
        # Should get past the body size check (not 413).
        assert resp.status_code != 413


# ---------------------------------------------------------------------------
# POST /v1/index
# ---------------------------------------------------------------------------


class TestIndexEndpoint:
    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_success(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Successful document indexing returns 200 with document_id."""
        doc_id = str(uuid.uuid4())
        mock_chunk.return_value = ["chunk-1", "chunk-2"]
        mock_embed.return_value = [[0.1] * 768, [0.2] * 768]

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            mock_ds.upsert_document = AsyncMock(
                return_value={
                    "document_id": doc_id,
                    "status": "indexed",
                    "total_chunks": 2,
                }
            )
            resp = await client.post(
                "/v1/index",
                json={"title": "Test Doc", "content": "Some meaningful content here."},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["document_id"] == doc_id
        assert body["status"] == "indexed"
        assert body["total_chunks"] == 2

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_blank_content_returns_400(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Content that is only whitespace should be rejected."""
        resp = await client.post(
            "/v1/index",
            json={"title": "Test Doc", "content": "   "},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "blank" in resp.json()["detail"].lower()

    async def test_index_missing_title_returns_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Missing required 'title' field returns validation error."""
        resp = await client.post(
            "/v1/index",
            json={"content": "Some content"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_index_missing_content_returns_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Missing required 'content' field returns validation error."""
        resp = await client.post(
            "/v1/index",
            json={"title": "Title"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_dedup_returns_deduplicated(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """When check_dedup detects a duplicate, status is 'deduplicated'."""
        doc_id = str(uuid.uuid4())

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(
                return_value={"document_id": doc_id, "total_chunks": 1}
            )
            resp = await client.post(
                "/v1/index",
                json={"title": "Dup Doc", "content": "Duplicate content."},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deduplicated"
        assert body["document_id"] == doc_id

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_private_visibility(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """PRIVATE visibility passes owner_user_id from identity."""
        doc_id = str(uuid.uuid4())
        mock_chunk.return_value = ["chunk-1"]
        mock_embed.return_value = [[0.1] * 768]

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            mock_ds.upsert_document = AsyncMock(
                return_value={
                    "document_id": doc_id,
                    "status": "indexed",
                    "total_chunks": 1,
                }
            )
            resp = await client.post(
                "/v1/index",
                json={
                    "title": "Private Doc",
                    "content": "Secret content.",
                    "visibility": "PRIVATE",
                },
                headers=auth_headers,
            )

            assert resp.status_code == 200
            # Verify owner_user_id was passed to the store
            call_kwargs = mock_ds.upsert_document.call_args.kwargs
            assert call_kwargs["owner_user_id"] == TEST_IDENTITY.user_id
            assert call_kwargs["visibility"] == "PRIVATE"

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_team_visibility_no_owner(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """TEAM visibility sets owner_user_id to None."""
        doc_id = str(uuid.uuid4())
        mock_chunk.return_value = ["chunk-1"]
        mock_embed.return_value = [[0.1] * 768]

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            mock_ds.upsert_document = AsyncMock(
                return_value={
                    "document_id": doc_id,
                    "status": "indexed",
                    "total_chunks": 1,
                }
            )
            resp = await client.post(
                "/v1/index",
                json={"title": "Team Doc", "content": "Shared content."},
                headers=auth_headers,
            )

            assert resp.status_code == 200
            call_kwargs = mock_ds.upsert_document.call_args.kwargs
            assert call_kwargs["owner_user_id"] is None
            assert call_kwargs["visibility"] == "TEAM"

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch(
        "rag_service.app.embed_chunks",
        new_callable=AsyncMock,
        side_effect=RuntimeError("API down"),
    )
    @patch(
        "rag_service.app.chunk_document",
        new_callable=AsyncMock,
        return_value=["chunk-1"],
    )
    async def test_index_embedding_failure_returns_503(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Embedding failure returns 503."""
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            resp = await client.post(
                "/v1/index",
                json={"title": "Doc", "content": "Content."},
                headers=auth_headers,
            )
        assert resp.status_code == 503
        assert "Embedding service unavailable" in resp.json()["detail"]

    async def test_index_invalid_visibility_returns_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Invalid visibility value (not TEAM/PRIVATE) returns 422."""
        resp = await client.post(
            "/v1/index",
            json={"title": "Doc", "content": "Content.", "visibility": "PUBLIC"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_with_metadata(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Metadata dict is passed through to the store."""
        doc_id = str(uuid.uuid4())
        mock_chunk.return_value = ["chunk-1"]
        mock_embed.return_value = [[0.1] * 768]
        metadata = {"source": "upload", "author": "tester"}

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            mock_ds.upsert_document = AsyncMock(
                return_value={
                    "document_id": doc_id,
                    "status": "indexed",
                    "total_chunks": 1,
                }
            )
            resp = await client.post(
                "/v1/index",
                json={
                    "title": "Meta Doc",
                    "content": "Content with metadata.",
                    "metadata": metadata,
                },
                headers=auth_headers,
            )

            assert resp.status_code == 200
            call_kwargs = mock_ds.upsert_document.call_args.kwargs
            assert call_kwargs["metadata"] == metadata

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_dedup_precheck_skips_embedding(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Fix 4: When check_dedup finds a match, embed_chunks is NOT called."""
        doc_id = str(uuid.uuid4())

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(
                return_value={"document_id": doc_id, "total_chunks": 3}
            )
            resp = await client.post(
                "/v1/index",
                json={"title": "Dup Doc", "content": "Duplicate content."},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deduplicated"
        assert body["document_id"] == doc_id
        assert body["total_chunks"] == 3
        # Embedding and chunking should NOT have been called
        mock_embed.assert_not_called()
        mock_chunk.assert_not_called()

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock)
    async def test_index_no_dedup_match_proceeds_to_embed(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Fix 4: When check_dedup returns None, chunking and embedding proceed."""
        doc_id = str(uuid.uuid4())
        mock_chunk.return_value = ["chunk-1"]
        mock_embed.return_value = [[0.1] * 768]

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            mock_ds.upsert_document = AsyncMock(
                return_value={
                    "document_id": doc_id,
                    "status": "indexed",
                    "total_chunks": 1,
                }
            )
            resp = await client.post(
                "/v1/index",
                json={"title": "New Doc", "content": "Fresh content."},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "indexed"
        mock_embed.assert_called_once()
        mock_chunk.assert_called_once()

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_chunks", new_callable=AsyncMock)
    @patch("rag_service.app.chunk_document", new_callable=AsyncMock, return_value=[])
    async def test_index_no_chunks_returns_400(
        self,
        mock_chunk: AsyncMock,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Document that produces zero chunks is rejected with 400."""
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.check_dedup = AsyncMock(return_value=None)
            resp = await client.post(
                "/v1/index",
                json={"title": "Empty Doc", "content": "x"},
                headers=auth_headers,
            )
        assert resp.status_code == 400
        assert "no chunks" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /v1/documents
# ---------------------------------------------------------------------------


class TestListDocuments:
    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_list_documents_success(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        doc_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        mock_docs = [
            {
                "id": doc_id,
                "title": "Doc One",
                "doc_type": "markdown",
                "source_uri": None,
                "visibility": "TEAM",
                "total_chunks": 3,
                "embedding_model": "gemini-embedding-001",
                "created_at": now,
                "updated_at": now,
            },
        ]

        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.list_documents = AsyncMock(return_value=(mock_docs, 1))
            resp = await client.get("/v1/documents", headers=auth_headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["documents"]) == 1
        assert body["documents"][0]["id"] == doc_id
        assert body["documents"][0]["title"] == "Doc One"

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_list_documents_empty(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.list_documents = AsyncMock(return_value=([], 0))
            resp = await client.get("/v1/documents", headers=auth_headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["documents"] == []

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_list_documents_pagination(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Limit and offset query params are forwarded to the store."""
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.list_documents = AsyncMock(return_value=([], 50))
            resp = await client.get(
                "/v1/documents?limit=10&offset=20",
                headers=auth_headers,
            )

        assert resp.status_code == 200
        # Verify the store was called with the specified pagination
        call_kwargs = mock_ds.list_documents.call_args.kwargs
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 20

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_list_documents_limit_clamped_to_200(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Limit above 200 is clamped to 200."""
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.list_documents = AsyncMock(return_value=([], 0))
            resp = await client.get(
                "/v1/documents?limit=500",
                headers=auth_headers,
            )

        assert resp.status_code == 200
        call_kwargs = mock_ds.list_documents.call_args.kwargs
        assert call_kwargs["limit"] == 200

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_list_documents_limit_clamped_to_minimum_1(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Limit below 1 is clamped to 1."""
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.list_documents = AsyncMock(return_value=([], 0))
            resp = await client.get(
                "/v1/documents?limit=0",
                headers=auth_headers,
            )

        assert resp.status_code == 200
        call_kwargs = mock_ds.list_documents.call_args.kwargs
        assert call_kwargs["limit"] == 1

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_list_documents_negative_offset_clamped_to_0(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Negative offset is clamped to 0."""
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.list_documents = AsyncMock(return_value=([], 0))
            resp = await client.get(
                "/v1/documents?offset=-5",
                headers=auth_headers,
            )

        assert resp.status_code == 200
        call_kwargs = mock_ds.list_documents.call_args.kwargs
        assert call_kwargs["offset"] == 0


# ---------------------------------------------------------------------------
# DELETE /v1/documents/{id}
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_delete_success(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        doc_id = str(uuid.uuid4())
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.soft_delete = AsyncMock(return_value=True)
            resp = await client.delete(f"/v1/documents/{doc_id}", headers=auth_headers)

        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] is True
        assert body["document_id"] == doc_id

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    async def test_delete_not_found_returns_404(
        self,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        doc_id = str(uuid.uuid4())
        with patch("rag_service.app._doc_store") as mock_ds:
            mock_ds.soft_delete = AsyncMock(return_value=False)
            resp = await client.delete(f"/v1/documents/{doc_id}", headers=auth_headers)

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    async def test_delete_without_auth_returns_401(self, client: AsyncClient):
        doc_id = str(uuid.uuid4())
        resp = await client.delete(f"/v1/documents/{doc_id}")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /v1/search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_query", new_callable=AsyncMock)
    async def test_search_success(
        self,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Successful search returns formatted results."""
        mock_embed.return_value = [0.1] * 768
        doc_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())

        mock_raw = {
            "results": [
                {
                    "document_id": doc_id,
                    "title": "Result Doc",
                    "doc_type": "markdown",
                    "rrf_score": 0.032,
                    "vector_score": 0.016,
                    "text_score": 0.016,
                    "chunks": [
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": 0,
                            "chunk_text": "Matching chunk text.",
                            "source": "vector",
                            "score": 0.95,
                        },
                    ],
                },
            ],
            "debug": None,
        }

        with patch("rag_service.app._search_store") as mock_ss:
            mock_ss.search_hybrid = AsyncMock(return_value=mock_raw)
            resp = await client.post(
                "/v1/search",
                json={"query": "test query", "limit": 5},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["document_id"] == doc_id
        assert body["results"][0]["rrf_score"] == 0.032
        assert len(body["results"][0]["chunks"]) == 1
        assert body["results"][0]["chunks"][0]["chunk_text"] == "Matching chunk text."
        assert body["debug"] is None

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_query", new_callable=AsyncMock)
    async def test_search_blank_query_returns_400(
        self,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Query that is only whitespace returns 400."""
        resp = await client.post(
            "/v1/search",
            json={"query": "   "},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "blank" in resp.json()["detail"].lower()

    async def test_search_empty_query_returns_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Empty string query is caught by Pydantic min_length=1 validation."""
        resp = await client.post(
            "/v1/search",
            json={"query": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_search_missing_query_returns_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Missing query field returns 422."""
        resp = await client.post(
            "/v1/search",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_query", new_callable=AsyncMock)
    async def test_search_with_debug(
        self,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """When include_debug=True, response includes debug metrics."""
        mock_embed.return_value = [0.1] * 768
        mock_raw = {
            "results": [],
            "debug": {
                "vector_pool_size": 15,
                "text_pool_size": 8,
                "vector_has_more": True,
                "text_has_more": False,
                "vector_cutoff_score": 0.72,
                "text_cutoff_score": 0.15,
            },
        }

        with patch("rag_service.app._search_store") as mock_ss:
            mock_ss.search_hybrid = AsyncMock(return_value=mock_raw)
            resp = await client.post(
                "/v1/search",
                json={"query": "debug query", "include_debug": True},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["debug"] is not None
        assert body["debug"]["vector_pool_size"] == 15
        assert body["debug"]["text_has_more"] is False
        assert body["debug"]["vector_cutoff_score"] == 0.72
        assert body["debug"]["text_cutoff_score"] == 0.15

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch("rag_service.app.embed_query", new_callable=AsyncMock)
    async def test_search_empty_results(
        self,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """No matching documents returns empty results list."""
        mock_embed.return_value = [0.1] * 768
        mock_raw = {"results": []}

        with patch("rag_service.app._search_store") as mock_ss:
            mock_ss.search_hybrid = AsyncMock(return_value=mock_raw)
            resp = await client.post(
                "/v1/search",
                json={"query": "unmatched query"},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        assert resp.json()["results"] == []

    @patch("rag_service.app.rls_connection", side_effect=_fake_rls_connection)
    @patch(
        "rag_service.app.embed_query",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Embedding API down"),
    )
    async def test_search_embedding_failure_returns_503(
        self,
        mock_embed: AsyncMock,
        mock_rls: MagicMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Embedding failure during search returns 503."""
        resp = await client.post(
            "/v1/search",
            json={"query": "test query"},
            headers=auth_headers,
        )
        assert resp.status_code == 503
        assert "Embedding service unavailable" in resp.json()["detail"]

    async def test_search_without_auth_returns_401(self, client: AsyncClient):
        resp = await client.post(
            "/v1/search",
            json={"query": "test"},
        )
        assert resp.status_code == 401

    async def test_search_limit_above_max_returns_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Limit > 50 is rejected by Pydantic validation (le=50)."""
        resp = await client.post(
            "/v1/search",
            json={"query": "test", "limit": 100},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_search_limit_below_min_returns_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ):
        """Limit < 1 is rejected by Pydantic validation (ge=1)."""
        resp = await client.post(
            "/v1/search",
            json={"query": "test", "limit": 0},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class TestRequestIdMiddleware:
    async def test_response_has_request_id_header(self, client: AsyncClient):
        """Every response should include an x-request-id header."""
        resp = await client.get("/liveness")
        assert "x-request-id" in resp.headers

    async def test_provided_request_id_is_echoed(self, client: AsyncClient):
        """If the client sends x-request-id, the server echoes it back."""
        resp = await client.get(
            "/liveness",
            headers={"x-request-id": "custom-trace-123"},
        )
        assert resp.headers["x-request-id"] == "custom-trace-123"
