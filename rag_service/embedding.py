"""Embedding utilities using Gemini embedding models.

Ported from ApexFlow's remme/utils.py with key changes:
- Model upgraded to gemini-embedding-001 (from text-embedding-004)
- output_dimensionality=768 explicitly set
- Dim guard: fails loudly instead of returning zero vector
- Removed legacy Nomic task-type mapping
- Gemini client factory inlined (no separate module needed)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import cast

import numpy as np
from google import genai

from rag_service.config import (
    RAG_EMBED_MAX_CONCURRENCY,
    RAG_EMBED_MAX_RETRIES,
    RAG_EMBED_RETRY_BASE_SECONDS,
    RAG_EMBEDDING_DIM,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_TASK_DOC,
    RAG_EMBEDDING_TASK_QUERY,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
)

logger = logging.getLogger(__name__)


def _is_gcp_environment() -> bool:
    """Detect if running on GCP (Cloud Run, GCE, etc.)."""
    return bool(os.getenv("K_SERVICE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))


@lru_cache(maxsize=1)
def _get_gemini_client() -> genai.Client:
    """Cached Gemini client with automatic credential detection."""
    if _is_gcp_environment():
        return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Set it for local dev or run on GCP for ADC.")
    return genai.Client(api_key=api_key)


def get_embedding(text: str, task_type: str = RAG_EMBEDDING_TASK_DOC) -> list[float]:
    """Generate embedding for text using the configured Gemini embedding model.

    Args:
        text: The text to embed.
        task_type: RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for search.

    Returns:
        L2-normalized embedding vector as a list of floats.

    Raises:
        ValueError: If the returned embedding dimension doesn't match config.
        RuntimeError: If the embedding API call fails.
    """
    client = _get_gemini_client()

    response = client.models.embed_content(
        model=RAG_EMBEDDING_MODEL,
        contents=text,
        config={"task_type": task_type, "output_dimensionality": RAG_EMBEDDING_DIM},
    )

    assert response.embeddings is not None, "Embedding response was empty"
    embedding = response.embeddings[0].values
    assert embedding is not None, "Embedding values were None"

    if len(embedding) != RAG_EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {RAG_EMBEDDING_DIM}. "
            f"Model {RAG_EMBEDDING_MODEL} returned unexpected dimensions."
        )

    vec = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return cast(list[float], vec.tolist())


async def embed_query(query_text: str) -> list[float]:
    """Embed a search query (async wrapper, RETRIEVAL_QUERY task type)."""
    return await _embed_with_retries(query_text, RAG_EMBEDDING_TASK_QUERY)


async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed multiple chunks with bounded concurrency and retries."""
    if not chunks:
        return []

    concurrency = max(1, RAG_EMBED_MAX_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency)

    async def _embed_one(text: str) -> list[float]:
        async with semaphore:
            return await _embed_with_retries(text, RAG_EMBEDDING_TASK_DOC)

    return list(await asyncio.gather(*(_embed_one(text) for text in chunks)))


async def _embed_with_retries(text: str, task_type: str) -> list[float]:
    """Run blocking embedding call in executor with bounded retries."""
    loop = asyncio.get_running_loop()
    retries = max(0, RAG_EMBED_MAX_RETRIES)

    for attempt in range(retries + 1):
        try:
            result = await loop.run_in_executor(None, get_embedding, text, task_type)
            return list(cast(Sequence[float], result))
        except Exception:
            if attempt >= retries:
                raise
            backoff_seconds = RAG_EMBED_RETRY_BASE_SECONDS * (2**attempt)
            logger.warning(
                "Embedding attempt %d/%d failed; retrying in %.2fs",
                attempt + 1,
                retries + 1,
                backoff_seconds,
            )
            await asyncio.sleep(backoff_seconds)

    raise RuntimeError("Unreachable embedding retry path")


async def check_embedding_service() -> bool:
    """Quick health check: verify the embedding API is reachable."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_embedding, "health check", RAG_EMBEDDING_TASK_QUERY)
        return True
    except Exception:
        logger.warning("Embedding health check failed", exc_info=True)
        return False
