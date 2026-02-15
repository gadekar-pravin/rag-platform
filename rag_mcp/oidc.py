"""OIDC token minting for Cloud Run service-to-service auth.

On Cloud Run the MCP server mints its own ID token targeting the RAG service URL
as audience.  The token is cached and refreshed 5 minutes before expiry.

Uses the GCE metadata server directly via ``urllib.request`` (stdlib) so there
are **no extra dependencies** beyond the Python standard library.  On local dev
(no metadata server) ``get_service_token()`` returns ``None`` so callers fall
back to static tokens.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cloud Run sets K_SERVICE automatically.
_ON_CLOUD_RUN = bool(os.getenv("K_SERVICE"))

_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience}"
)
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


@dataclass
class _CachedToken:
    token: str
    expires_at: float  # unix timestamp


_cache: _CachedToken | None = None

# Refresh this many seconds before the token actually expires.
_REFRESH_MARGIN_SECONDS = 300  # 5 minutes


def _decode_jwt_exp(token: str) -> float:
    """Decode the ``exp`` claim from a JWT *without* signature verification.

    The RAG service performs full verification; we only need the expiry time
    for cache management.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Token is not a valid JWT (expected 3 parts)")

    payload_b64 = parts[1]
    # Add padding
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding

    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    payload = json.loads(payload_bytes)
    exp = payload.get("exp")
    if exp is None:
        raise ValueError("JWT payload missing 'exp' claim")
    return float(exp)


def _mint_token(audience: str) -> str:
    """Mint a fresh OIDC ID token via the GCE metadata server."""
    url = _METADATA_URL.format(audience=urllib.parse.quote(audience, safe=""))
    req = urllib.request.Request(url, headers=_METADATA_HEADERS)
    with urllib.request.urlopen(req, timeout=5) as resp:
        token = resp.read().decode("utf-8").strip()
    if not token:
        raise RuntimeError("Metadata server returned empty token")
    return token


def get_service_token(audience: str) -> str | None:
    """Return an OIDC ID token for service-to-service auth, or ``None``.

    On Cloud Run this mints a token using the service account's identity,
    targeting *audience* (the RAG service URL).  The token is cached and
    refreshed 5 minutes before expiry.  If the cached token cannot be
    refreshed but hasn't actually expired yet, the stale token is returned
    as a fallback.

    Returns ``None`` when not running on Cloud Run (no metadata server).
    """
    global _cache  # noqa: PLW0603

    if not _ON_CLOUD_RUN:
        return None

    now = time.time()

    # Return cached token if still fresh (not within refresh margin).
    if _cache is not None and now < _cache.expires_at - _REFRESH_MARGIN_SECONDS:
        return _cache.token

    # Attempt to mint a new token.
    try:
        token = _mint_token(audience)
        exp = _decode_jwt_exp(token)
        _cache = _CachedToken(token=token, expires_at=exp)
        return token
    except Exception:
        logger.warning("Failed to mint OIDC token for %s", audience, exc_info=True)

        # Stale fallback: return the old token if it hasn't truly expired.
        if _cache is not None and now < _cache.expires_at:
            logger.info("Using stale cached OIDC token (still valid)")
            return _cache.token

        return None


def clear_cache() -> None:
    """Clear the cached token. Exposed for tests."""
    global _cache  # noqa: PLW0603
    _cache = None
