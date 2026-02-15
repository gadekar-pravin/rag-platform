"""Enable and enforce RLS on child and ingestion tables.

Revision ID: 002
Revises: 001
Create Date: 2026-02-15
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- rag_document_chunks -------------------------------------------------
    op.execute("ALTER TABLE rag_document_chunks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_document_chunks FORCE ROW LEVEL SECURITY")

    op.execute("""
        CREATE POLICY rag_chunks_select ON rag_document_chunks
        FOR SELECT
        USING (
            EXISTS (
                SELECT 1
                FROM rag_documents d
                WHERE d.id = rag_document_chunks.document_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_chunks_insert ON rag_document_chunks
        FOR INSERT
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM rag_documents d
                WHERE d.id = rag_document_chunks.document_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_chunks_update ON rag_document_chunks
        FOR UPDATE
        USING (
            EXISTS (
                SELECT 1
                FROM rag_documents d
                WHERE d.id = rag_document_chunks.document_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM rag_documents d
                WHERE d.id = rag_document_chunks.document_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_chunks_delete ON rag_document_chunks
        FOR DELETE
        USING (
            EXISTS (
                SELECT 1
                FROM rag_documents d
                WHERE d.id = rag_document_chunks.document_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    # -- rag_chunk_embeddings ------------------------------------------------
    op.execute("ALTER TABLE rag_chunk_embeddings ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_chunk_embeddings FORCE ROW LEVEL SECURITY")

    op.execute("""
        CREATE POLICY rag_embeddings_select ON rag_chunk_embeddings
        FOR SELECT
        USING (
            EXISTS (
                SELECT 1
                FROM rag_document_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id = rag_chunk_embeddings.chunk_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_embeddings_insert ON rag_chunk_embeddings
        FOR INSERT
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM rag_document_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id = rag_chunk_embeddings.chunk_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_embeddings_update ON rag_chunk_embeddings
        FOR UPDATE
        USING (
            EXISTS (
                SELECT 1
                FROM rag_document_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id = rag_chunk_embeddings.chunk_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM rag_document_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id = rag_chunk_embeddings.chunk_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_embeddings_delete ON rag_chunk_embeddings
        FOR DELETE
        USING (
            EXISTS (
                SELECT 1
                FROM rag_document_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                WHERE c.id = rag_chunk_embeddings.chunk_id
                  AND d.tenant_id = current_setting('app.tenant_id', true)
                  AND d.deleted_at IS NULL
                  AND (
                      d.visibility = 'TEAM'
                      OR (
                          d.visibility = 'PRIVATE'
                          AND d.owner_user_id = current_setting('app.user_id', true)
                      )
                  )
            )
        )
    """)

    # -- rag_ingestion_runs --------------------------------------------------
    op.execute("ALTER TABLE rag_ingestion_runs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_ingestion_runs FORCE ROW LEVEL SECURITY")

    op.execute("""
        CREATE POLICY rag_ingestion_runs_select ON rag_ingestion_runs
        FOR SELECT
        USING (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """)

    op.execute("""
        CREATE POLICY rag_ingestion_runs_insert ON rag_ingestion_runs
        FOR INSERT
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """)

    op.execute("""
        CREATE POLICY rag_ingestion_runs_update ON rag_ingestion_runs
        FOR UPDATE
        USING (
            tenant_id = current_setting('app.tenant_id', true)
        )
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """)

    op.execute("""
        CREATE POLICY rag_ingestion_runs_delete ON rag_ingestion_runs
        FOR DELETE
        USING (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """)

    # -- rag_ingestion_items -------------------------------------------------
    op.execute("ALTER TABLE rag_ingestion_items ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_ingestion_items FORCE ROW LEVEL SECURITY")

    op.execute("""
        CREATE POLICY rag_ingestion_items_select ON rag_ingestion_items
        FOR SELECT
        USING (
            EXISTS (
                SELECT 1
                FROM rag_ingestion_runs r
                WHERE r.id = rag_ingestion_items.run_id
                  AND r.tenant_id = current_setting('app.tenant_id', true)
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_ingestion_items_insert ON rag_ingestion_items
        FOR INSERT
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM rag_ingestion_runs r
                WHERE r.id = rag_ingestion_items.run_id
                  AND r.tenant_id = current_setting('app.tenant_id', true)
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_ingestion_items_update ON rag_ingestion_items
        FOR UPDATE
        USING (
            EXISTS (
                SELECT 1
                FROM rag_ingestion_runs r
                WHERE r.id = rag_ingestion_items.run_id
                  AND r.tenant_id = current_setting('app.tenant_id', true)
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM rag_ingestion_runs r
                WHERE r.id = rag_ingestion_items.run_id
                  AND r.tenant_id = current_setting('app.tenant_id', true)
            )
        )
    """)

    op.execute("""
        CREATE POLICY rag_ingestion_items_delete ON rag_ingestion_items
        FOR DELETE
        USING (
            EXISTS (
                SELECT 1
                FROM rag_ingestion_runs r
                WHERE r.id = rag_ingestion_items.run_id
                  AND r.tenant_id = current_setting('app.tenant_id', true)
            )
        )
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS rag_ingestion_items_delete ON rag_ingestion_items")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_items_update ON rag_ingestion_items")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_items_insert ON rag_ingestion_items")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_items_select ON rag_ingestion_items")
    op.execute("ALTER TABLE rag_ingestion_items DISABLE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_delete ON rag_ingestion_runs")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_update ON rag_ingestion_runs")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_insert ON rag_ingestion_runs")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_select ON rag_ingestion_runs")
    op.execute("ALTER TABLE rag_ingestion_runs DISABLE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS rag_embeddings_delete ON rag_chunk_embeddings")
    op.execute("DROP POLICY IF EXISTS rag_embeddings_update ON rag_chunk_embeddings")
    op.execute("DROP POLICY IF EXISTS rag_embeddings_insert ON rag_chunk_embeddings")
    op.execute("DROP POLICY IF EXISTS rag_embeddings_select ON rag_chunk_embeddings")
    op.execute("ALTER TABLE rag_chunk_embeddings DISABLE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS rag_chunks_delete ON rag_document_chunks")
    op.execute("DROP POLICY IF EXISTS rag_chunks_update ON rag_document_chunks")
    op.execute("DROP POLICY IF EXISTS rag_chunks_insert ON rag_document_chunks")
    op.execute("DROP POLICY IF EXISTS rag_chunks_select ON rag_document_chunks")
    op.execute("ALTER TABLE rag_document_chunks DISABLE ROW LEVEL SECURITY")
