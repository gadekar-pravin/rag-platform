"""Shared test fixtures for the rag-platform test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def test_tenant_id() -> str:
    return "test-tenant"


@pytest.fixture
def test_user_id() -> str:
    return "test-user@example.com"


@pytest.fixture
def other_user_id() -> str:
    return "other-user@example.com"


@pytest.fixture
def other_tenant_id() -> str:
    return "other-tenant"
