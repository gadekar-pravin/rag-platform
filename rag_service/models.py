"""Pydantic request/response schemas for the RAG service API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# -- Search -------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=10_000, description="Search query text"
    )
    limit: int = Field(10, ge=1, le=50, description="Max documents to return")
    include_debug: bool = Field(False, description="Include debug metrics in response")


class ChunkResult(BaseModel):
    chunk_id: str
    chunk_index: int
    chunk_text: str
    source: str  # "vector" or "text"
    score: float


class SearchResult(BaseModel):
    document_id: str
    title: str
    doc_type: str | None = None
    rrf_score: float
    vector_score: float
    text_score: float
    chunks: list[ChunkResult]


class SearchDebug(BaseModel):
    vector_pool_size: int
    text_pool_size: int
    vector_has_more: bool
    text_has_more: bool
    vector_cutoff_score: float | None = None
    text_cutoff_score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    debug: SearchDebug | None = None


# -- Documents ----------------------------------------------------------------


class DocumentSummary(BaseModel):
    id: str
    title: str
    doc_type: str | None = None
    source_uri: str | None = None
    visibility: str
    total_chunks: int | None = None
    embedding_model: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int


class DeleteResponse(BaseModel):
    deleted: bool
    document_id: str


# -- Index (MVP manual upload) ------------------------------------------------


class IndexRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(
        ..., min_length=1, max_length=2_000_000, description="Document text content"
    )
    doc_type: str | None = Field(
        None, max_length=100, description="Document type (e.g., 'markdown', 'pdf')"
    )
    visibility: str = Field(
        "TEAM", pattern="^(TEAM|PRIVATE)$", description="TEAM or PRIVATE"
    )
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class IndexResponse(BaseModel):
    document_id: str
    status: str  # "indexed" or "deduplicated"
    total_chunks: int


# -- Health -------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    error: str | None = None
