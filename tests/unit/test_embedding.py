"""Unit tests for embedding module — mock genai, verify dim guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_service.config import RAG_EMBEDDING_DIM


class TestGetEmbedding:
    @patch("rag_service.embedding._get_gemini_client")
    def test_successful_embedding(self, mock_client_fn):
        """Normal embedding returns L2-normalized vector of correct dimension."""
        from rag_service.embedding import get_embedding

        mock_values = [0.5] * RAG_EMBEDDING_DIM
        mock_embedding = MagicMock()
        mock_embedding.values = mock_values

        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = get_embedding("test text")

        assert len(result) == RAG_EMBEDDING_DIM
        assert isinstance(result, list)
        # Verify L2 normalization (norm should be ~1.0)
        import numpy as np
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    @patch("rag_service.embedding._get_gemini_client")
    def test_dim_guard_fails_loudly(self, mock_client_fn):
        """Dim mismatch raises ValueError — no silent zero-vector fallback."""
        from rag_service.embedding import get_embedding

        wrong_dim = RAG_EMBEDDING_DIM + 100
        mock_values = [0.5] * wrong_dim
        mock_embedding = MagicMock()
        mock_embedding.values = mock_values

        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        with pytest.raises(ValueError, match="dimension mismatch"):
            get_embedding("test text")

    @patch("rag_service.embedding._get_gemini_client")
    def test_empty_response_raises(self, mock_client_fn):
        """Empty embedding response raises AssertionError."""
        from rag_service.embedding import get_embedding

        mock_response = MagicMock()
        mock_response.embeddings = None

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        with pytest.raises(AssertionError, match="empty"):
            get_embedding("test text")

    @patch("rag_service.embedding._get_gemini_client")
    def test_task_type_passed_to_api(self, mock_client_fn):
        """Task type is correctly forwarded to the Gemini API."""
        from rag_service.embedding import get_embedding

        mock_values = [0.5] * RAG_EMBEDDING_DIM
        mock_embedding = MagicMock()
        mock_embedding.values = mock_values

        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        get_embedding("test text", "RETRIEVAL_QUERY")

        call_kwargs = mock_client.models.embed_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config["task_type"] == "RETRIEVAL_QUERY"


class TestGeminiClientDetection:
    @patch.dict("os.environ", {"K_SERVICE": "test-service"}, clear=False)
    def test_gcp_detection(self):
        from rag_service.embedding import _is_gcp_environment
        assert _is_gcp_environment()

    @patch.dict("os.environ", {}, clear=True)
    def test_local_detection(self):
        from rag_service.embedding import _is_gcp_environment
        assert not _is_gcp_environment()
