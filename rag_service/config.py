"""Environment-variable-driven configuration for the RAG service.

No dependency on ApexFlow's settings.json — all config comes from env vars.
"""

from __future__ import annotations

import os


# -- Tenant -------------------------------------------------------------------
TENANT_ID: str = os.getenv("TENANT_ID", "default")

# -- Embedding ----------------------------------------------------------------
RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
RAG_EMBEDDING_DIM: int = int(os.getenv("RAG_EMBEDDING_DIM", "768"))
RAG_EMBEDDING_TASK_DOC: str = "RETRIEVAL_DOCUMENT"
RAG_EMBEDDING_TASK_QUERY: str = "RETRIEVAL_QUERY"

# -- Search -------------------------------------------------------------------
RAG_RRF_K: int = int(os.getenv("RAG_RRF_K", "60"))
RAG_SEARCH_EXPANSION: int = int(os.getenv("RAG_SEARCH_EXPANSION", "3"))

# -- Ingestion ----------------------------------------------------------------
RAG_INGESTION_VERSION: str = os.getenv("RAG_INGESTION_VERSION", "v1")

# -- Chunking -----------------------------------------------------------------
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "2000"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# -- Auth ---------------------------------------------------------------------
RAG_SHARED_TOKEN: str | None = os.getenv("RAG_SHARED_TOKEN")

# -- GCP ----------------------------------------------------------------------
VERTEX_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "apexflow-ai")
VERTEX_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# -- Server -------------------------------------------------------------------
IS_CLOUD_RUN: bool = bool(os.getenv("K_SERVICE"))
