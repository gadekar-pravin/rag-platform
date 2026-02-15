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


class TestGetEmbeddingsBatch:
    """Fix 5: Batch embedding sends multiple texts in one API call."""

    @patch("rag_service.embedding._get_gemini_client")
    def test_batch_returns_correct_count(self, mock_client_fn):
        """Batch embedding returns one vector per input text."""
        from rag_service.embedding import get_embeddings_batch

        texts = ["text one", "text two", "text three"]
        mock_embeddings = []
        for _ in texts:
            emb = MagicMock()
            emb.values = [0.5] * RAG_EMBEDDING_DIM
            mock_embeddings.append(emb)

        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        results = get_embeddings_batch(texts)

        assert len(results) == 3
        # Verify single API call
        mock_client.models.embed_content.assert_called_once()
        # Verify contents was the list of texts
        call_kwargs = mock_client.models.embed_content.call_args
        assert call_kwargs.kwargs["contents"] == texts

    @patch("rag_service.embedding._get_gemini_client")
    def test_batch_l2_normalizes_each(self, mock_client_fn):
        """Each embedding in the batch is L2-normalized."""
        import numpy as np

        from rag_service.embedding import get_embeddings_batch

        texts = ["a", "b"]
        mock_embeddings = []
        for _ in texts:
            emb = MagicMock()
            emb.values = [3.0, 4.0] + [0.0] * (RAG_EMBEDDING_DIM - 2)
            mock_embeddings.append(emb)

        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        results = get_embeddings_batch(texts)

        for vec in results:
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 0.01

    @patch("rag_service.embedding._get_gemini_client")
    def test_batch_dim_guard(self, mock_client_fn):
        """Batch embedding raises on dimension mismatch."""
        from rag_service.embedding import get_embeddings_batch

        emb = MagicMock()
        emb.values = [0.5] * (RAG_EMBEDDING_DIM + 10)  # wrong dim

        mock_response = MagicMock()
        mock_response.embeddings = [emb]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        with pytest.raises(ValueError, match="dimension mismatch"):
            get_embeddings_batch(["test"])

    @patch("rag_service.embedding._get_gemini_client")
    def test_batch_count_mismatch_raises(self, mock_client_fn):
        """Batch embedding raises when API returns wrong number of embeddings."""
        from rag_service.embedding import get_embeddings_batch

        emb = MagicMock()
        emb.values = [0.5] * RAG_EMBEDDING_DIM

        mock_response = MagicMock()
        mock_response.embeddings = [emb]  # Only 1, but we sent 2 texts

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_fn.return_value = mock_client

        with pytest.raises(ValueError, match="count mismatch"):
            get_embeddings_batch(["text1", "text2"])


class TestEmbedChunksBatching:
    """Fix 5: embed_chunks splits into batches."""

    @patch("rag_service.embedding.RAG_EMBED_BATCH_SIZE", 100)
    @patch("rag_service.embedding._embed_batch_with_retries")
    async def test_single_batch_for_small_input(self, mock_batch):
        """Fewer than batch_size chunks → single batch call."""
        from rag_service.embedding import embed_chunks

        mock_batch.return_value = [[0.1] * 768] * 50
        result = await embed_chunks(["text"] * 50)

        assert len(result) == 50
        assert mock_batch.call_count == 1

    @patch("rag_service.embedding.RAG_EMBED_BATCH_SIZE", 100)
    @patch("rag_service.embedding._embed_batch_with_retries")
    async def test_multiple_batches(self, mock_batch):
        """150 chunks with batch_size=100 → 2 API calls."""
        from rag_service.embedding import embed_chunks

        mock_batch.side_effect = [
            [[0.1] * 768] * 100,  # First batch: 100
            [[0.2] * 768] * 50,   # Second batch: 50
        ]
        result = await embed_chunks(["text"] * 150)

        assert len(result) == 150
        assert mock_batch.call_count == 2

    async def test_empty_chunks_returns_empty(self):
        """Empty input returns empty list without API calls."""
        from rag_service.embedding import embed_chunks

        result = await embed_chunks([])
        assert result == []


class TestGeminiClientDetection:
    @patch.dict("os.environ", {"K_SERVICE": "test-service"}, clear=False)
    def test_gcp_detection(self):
        from rag_service.embedding import _is_gcp_environment
        assert _is_gcp_environment()

    @patch.dict("os.environ", {}, clear=True)
    def test_local_detection(self):
        from rag_service.embedding import _is_gcp_environment
        assert not _is_gcp_environment()
