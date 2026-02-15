"""Seed development data into the RAG platform.

Usage:
    python scripts/seed-dev-data.py

Requires:
    - Database running (default: localhost:5432)
    - GEMINI_API_KEY set for embedding generation
    - Migration applied (alembic upgrade head)
"""

from __future__ import annotations

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SAMPLE_DOCUMENTS = [
    {
        "title": "API Design Guidelines",
        "content": (
            "REST API Design Best Practices\n\n"
            "1. Use nouns for resource URLs, not verbs.\n"
            "2. Use HTTP methods correctly: GET for reads, POST for creates, "
            "PUT for full updates, PATCH for partial updates, DELETE for removal.\n"
            "3. Return appropriate HTTP status codes: 200 OK, 201 Created, "
            "204 No Content, 400 Bad Request, 401 Unauthorized, 404 Not Found.\n"
            "4. Version your API using URL path (e.g., /v1/users) or headers.\n"
            "5. Use pagination for list endpoints: offset/limit or cursor-based.\n"
            "6. Support filtering, sorting, and field selection via query parameters.\n"
            "7. Use HATEOAS for discoverability where appropriate.\n"
            "8. Document your API with OpenAPI/Swagger specifications."
        ),
        "doc_type": "guidelines",
    },
    {
        "title": "Database Migration Runbook",
        "content": (
            "Database Migration Procedures\n\n"
            "Pre-migration checklist:\n"
            "- Take a full database backup\n"
            "- Review migration SQL in a staging environment\n"
            "- Notify the team about the planned downtime window\n"
            "- Ensure rollback scripts are tested\n\n"
            "Migration steps:\n"
            "1. Enable maintenance mode on the application\n"
            "2. Run alembic upgrade head\n"
            "3. Verify schema changes with \\dt and \\d+ commands\n"
            "4. Run smoke tests against the migrated database\n"
            "5. Disable maintenance mode\n"
            "6. Monitor application logs for errors\n\n"
            "Rollback procedure:\n"
            "1. Enable maintenance mode\n"
            "2. Run alembic downgrade -1\n"
            "3. Restore from backup if needed\n"
            "4. Disable maintenance mode"
        ),
        "doc_type": "runbook",
    },
    {
        "title": "Python Coding Standards",
        "content": (
            "Python Coding Standards for the Team\n\n"
            "Formatting:\n"
            "- Use ruff for linting and formatting\n"
            "- Line length: 120 characters maximum\n"
            "- Use type hints for all function signatures\n\n"
            "Naming conventions:\n"
            "- snake_case for functions and variables\n"
            "- PascalCase for classes\n"
            "- UPPER_CASE for constants\n\n"
            "Architecture patterns:\n"
            "- Prefer async/await for I/O operations\n"
            "- Use dependency injection over global state\n"
            "- Write stateless service classes\n"
            "- Use Pydantic for data validation\n\n"
            "Testing:\n"
            "- Write unit tests with pytest\n"
            "- Use pytest-asyncio for async tests\n"
            "- Mock external dependencies\n"
            "- Aim for 80%+ code coverage"
        ),
        "doc_type": "standards",
    },
]


async def main() -> None:
    from rag_service.db import get_pool, close_pool, rls_connection
    from rag_service.embedding import embed_chunks
    from rag_service.chunking.chunker import chunk_document
    from rag_service.stores.rag_document_store import RagDocumentStore

    store = RagDocumentStore()
    tenant_id = os.getenv("TENANT_ID", "default")
    user_id = "seed-script@dev"

    print(f"Seeding {len(SAMPLE_DOCUMENTS)} documents for tenant '{tenant_id}'...")

    for doc in SAMPLE_DOCUMENTS:
        title = doc["title"]
        content = doc["content"]
        doc_type = doc["doc_type"]

        print(f"  Chunking '{title}'...")
        chunks = await chunk_document(content)
        if not chunks:
            print(f"  Skipping '{title}' — no chunks produced")
            continue

        print(f"  Embedding {len(chunks)} chunks...")
        embeddings = await embed_chunks(chunks)

        async with rls_connection(tenant_id, user_id) as conn:
            result = await store.upsert_document(
                conn,
                tenant_id=tenant_id,
                title=title,
                content=content,
                chunks=chunks,
                embeddings=embeddings,
                doc_type=doc_type,
            )
            print(f"  {title}: {result['status']} ({result['total_chunks']} chunks)")

    await close_pool()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
