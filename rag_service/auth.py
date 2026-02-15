"""Authentication for the RAG service.

Two auth modes:
1. Cloud Run OIDC: Verifies Google identity tokens (Cloud Run sets
   X-Serverless-Authorization or Authorization header with OIDC tokens).
2. Shared bearer token: For local dev only (disabled when K_SERVICE is set).

The auth middleware extracts an Identity dataclass with tenant_id, user_id,
and principal, which is used to set RLS session variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from fastapi import HTTPException, Request
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from rag_service.config import IS_CLOUD_RUN, RAG_SHARED_TOKEN, TENANT_ID

logger = logging.getLogger(__name__)

# Cache the Google transport session for token verification
_transport = google_requests.Request()

# Paths that skip auth
_PUBLIC_PATHS = {"/liveness", "/readiness", "/docs", "/openapi.json"}


@dataclass
class Identity:
    """Authenticated caller identity."""

    tenant_id: str
    user_id: str
    principal: str  # email or sub claim


async def get_identity(request: Request) -> Identity:
    """Extract and verify caller identity from the request.

    Returns an Identity with tenant_id (from env), user_id (from token),
    and principal (email or sub claim).

    Raises HTTPException 401 if no valid credentials are provided.
    """
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token")

    # Try shared token first (dev only)
    if not IS_CLOUD_RUN and RAG_SHARED_TOKEN and token == RAG_SHARED_TOKEN:
        return Identity(
            tenant_id=TENANT_ID,
            user_id="dev-user",
            principal="dev-user@local",
        )

    # Verify OIDC token
    try:
        claims = id_token.verify_token(token, _transport)
        email = claims.get("email", "")
        sub = claims.get("sub", "")
        principal = email or sub
        if not principal:
            raise HTTPException(status_code=401, detail="Token missing email and sub claims")

        return Identity(
            tenant_id=TENANT_ID,
            user_id=principal,
            principal=principal,
        )
    except Exception as e:
        logger.warning("Token verification failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid token") from e


def _extract_token(request: Request) -> str | None:
    """Extract bearer token from Authorization header or query param."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    # Fallback: query param (for SSE clients that can't send headers)
    token = request.query_params.get("token")
    if token:
        return token

    return None


def is_public_path(path: str) -> bool:
    """Check if the request path skips authentication."""
    return path in _PUBLIC_PATHS or path.startswith("/docs")


def require_auth_on_cloud_run() -> None:
    """Safety check: shared token must not be usable on Cloud Run."""
    if IS_CLOUD_RUN and RAG_SHARED_TOKEN:
        logger.warning(
            "RAG_SHARED_TOKEN is set on Cloud Run — it will be ignored. "
            "Use OIDC tokens for authentication in production."
        )
