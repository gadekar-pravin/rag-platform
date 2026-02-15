"""Environment-variable-driven configuration for the RAG service.

No dependency on ApexFlow's settings.json — all config comes from env vars.
"""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# -- Tenant -------------------------------------------------------------------
TENANT_ID: str = os.getenv("TENANT_ID", "default")
RAG_TENANT_CLAIM: str = os.getenv("RAG_TENANT_CLAIM", "tenant_id")
RAG_REQUIRE_TENANT_CLAIM: bool = _env_bool("RAG_REQUIRE_TENANT_CLAIM", False)

# -- Embedding ----------------------------------------------------------------
RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
RAG_EMBEDDING_DIM: int = int(os.getenv("RAG_EMBEDDING_DIM", "768"))
RAG_EMBEDDING_TASK_DOC: str = "RETRIEVAL_DOCUMENT"
RAG_EMBEDDING_TASK_QUERY: str = "RETRIEVAL_QUERY"
RAG_EMBED_BATCH_SIZE: int = int(os.getenv("RAG_EMBED_BATCH_SIZE", "100"))
RAG_EMBED_MAX_CONCURRENCY: int = int(os.getenv("RAG_EMBED_MAX_CONCURRENCY", "8"))
RAG_EMBED_MAX_RETRIES: int = int(os.getenv("RAG_EMBED_MAX_RETRIES", "2"))
RAG_EMBED_RETRY_BASE_SECONDS: float = float(os.getenv("RAG_EMBED_RETRY_BASE_SECONDS", "0.5"))

# -- Search -------------------------------------------------------------------
RAG_RRF_K: int = int(os.getenv("RAG_RRF_K", "60"))
RAG_SEARCH_EXPANSION: int = int(os.getenv("RAG_SEARCH_EXPANSION", "3"))
RAG_SEARCH_PER_DOC_CAP: int = int(os.getenv("RAG_SEARCH_PER_DOC_CAP", "3"))
RAG_SEARCH_CANDIDATE_MULTIPLIER: int = int(os.getenv("RAG_SEARCH_CANDIDATE_MULTIPLIER", "4"))

# -- Ingestion ----------------------------------------------------------------
RAG_INGESTION_VERSION: str = os.getenv("RAG_INGESTION_VERSION", "v1")

# -- Chunking -----------------------------------------------------------------
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "2000"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# -- Auth ---------------------------------------------------------------------
RAG_SHARED_TOKEN: str | None = os.getenv("RAG_SHARED_TOKEN")
RAG_OIDC_AUDIENCE: str | None = os.getenv("RAG_OIDC_AUDIENCE")
RAG_ALLOWED_ISSUERS: set[str] = set(
    _env_csv("RAG_ALLOWED_ISSUERS", "https://accounts.google.com,accounts.google.com")
)

# -- GCP ----------------------------------------------------------------------
VERTEX_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "apexflow-ai")
VERTEX_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# -- CORS ---------------------------------------------------------------------
RAG_CORS_ALLOW_ORIGINS: list[str] = _env_csv(
    "RAG_CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
)
RAG_CORS_ALLOW_METHODS: list[str] = _env_csv(
    "RAG_CORS_ALLOW_METHODS",
    "GET,POST,DELETE,OPTIONS",
)
RAG_CORS_ALLOW_HEADERS: list[str] = _env_csv(
    "RAG_CORS_ALLOW_HEADERS",
    "Authorization,Content-Type",
)
RAG_CORS_ALLOW_CREDENTIALS: bool = _env_bool("RAG_CORS_ALLOW_CREDENTIALS", False)

# -- Server -------------------------------------------------------------------
IS_CLOUD_RUN: bool = bool(os.getenv("K_SERVICE"))
