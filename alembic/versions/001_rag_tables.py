"""Create rag_* tables with RLS and indexes.

Revision ID: 001
Create Date: 2026-02-15

Consolidated migration: creates all 5 tables, partial unique indexes for
dedup/upsert, and RLS policies on every table in a single step.
"""

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- Extensions -----------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # -- rag_documents --------------------------------------------------------
    op.execute(
        """
        CREATE TABLE rag_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            visibility TEXT NOT NULL DEFAULT 'TEAM'
                CHECK (visibility IN ('TEAM', 'PRIVATE')),
            owner_user_id TEXT,

            title TEXT NOT NULL,
            doc_type TEXT,
            source_uri TEXT,
            metadata JSONB DEFAULT '{}',
            content TEXT,

            source_hash TEXT,
            content_hash TEXT NOT NULL,

            embedding_model TEXT NOT NULL,
            embedding_dim INT NOT NULL,
            embedding_task_type_doc TEXT DEFAULT 'RETRIEVAL_DOCUMENT',
            embedding_task_type_query TEXT DEFAULT 'RETRIEVAL_QUERY',

            ingestion_version TEXT DEFAULT 'v1',
            chunk_method TEXT DEFAULT 'rule_based',
            total_chunks INT DEFAULT 0,

            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            deleted_at TIMESTAMPTZ,

            -- PRIVATE must have owner; TEAM must not
            CHECK (
                (visibility = 'TEAM' AND owner_user_id IS NULL)
                OR (visibility = 'PRIVATE' AND owner_user_id IS NOT NULL)
            )
        )
    """
    )

    # -- Dedup indexes (3 partial unique indexes) -----------------------------

    # TEAM docs with source_uri: one canonical doc per (tenant, source_uri)
    op.execute(
        """
        CREATE UNIQUE INDEX ux_rag_docs_team_source_uri
        ON rag_documents (tenant_id, source_uri)
        WHERE deleted_at IS NULL
          AND visibility = 'TEAM'
          AND source_uri IS NOT NULL
    """
    )

    # TEAM ad-hoc docs (no source_uri): dedup by content_hash
    op.execute(
        """
        CREATE UNIQUE INDEX ux_rag_docs_team_dedup_adhoc
        ON rag_documents (tenant_id, content_hash)
        WHERE deleted_at IS NULL
          AND visibility = 'TEAM'
          AND source_uri IS NULL
    """
    )

    # PRIVATE docs: dedup by (tenant, owner, content_hash)
    op.execute(
        """
        CREATE UNIQUE INDEX ux_rag_docs_private_dedup
        ON rag_documents (tenant_id, owner_user_id, content_hash)
        WHERE deleted_at IS NULL
          AND visibility = 'PRIVATE'
    """
    )

    # Filter index for common queries
    op.execute(
        """
        CREATE INDEX ix_rag_docs_filter
        ON rag_documents (tenant_id, visibility, owner_user_id, deleted_at)
    """
    )

    # -- rag_document_chunks --------------------------------------------------
    op.execute(
        """
        CREATE TABLE rag_document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL
                REFERENCES rag_documents(id) ON DELETE CASCADE,
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_start INT,
            chunk_end INT,

            fts TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('english', chunk_text)
            ) STORED,

            UNIQUE (document_id, chunk_index)
        )
    """
    )

    op.execute(
        """
        CREATE INDEX ix_rag_chunks_doc
        ON rag_document_chunks (document_id)
    """
    )

    op.execute(
        """
        CREATE INDEX ix_rag_chunks_fts
        ON rag_document_chunks USING GIN (fts)
    """
    )

    # -- rag_chunk_embeddings -------------------------------------------------
    op.execute(
        """
        CREATE TABLE rag_chunk_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            chunk_id UUID NOT NULL
                REFERENCES rag_document_chunks(id) ON DELETE CASCADE,
            embedding VECTOR(768) NOT NULL
        )
    """
    )

    op.execute(
        """
        CREATE INDEX ix_rag_emb_chunk
        ON rag_chunk_embeddings (chunk_id)
    """
    )

    # -- rag_ingestion_runs (tracking, populated later) -----------------------
    op.execute(
        """
        CREATE TABLE rag_ingestion_runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            status TEXT DEFAULT 'running'
                CHECK (status IN ('running', 'completed', 'failed')),
            total_files INT DEFAULT 0,
            processed_files INT DEFAULT 0,
            error_message TEXT,
            metadata JSONB DEFAULT '{}'
        )
    """
    )

    # -- rag_ingestion_items (tracking, populated later) ----------------------
    op.execute(
        """
        CREATE TABLE rag_ingestion_items (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id UUID NOT NULL
                REFERENCES rag_ingestion_runs(id) ON DELETE CASCADE,
            document_id UUID REFERENCES rag_documents(id) ON DELETE SET NULL,
            source_uri TEXT,
            status TEXT DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
            error_message TEXT,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ
        )
    """
    )

    # -- Row-Level Security: rag_documents ------------------------------------

    op.execute("ALTER TABLE rag_documents ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_documents FORCE ROW LEVEL SECURITY")

    # SELECT: tenant match + (TEAM visible to all, PRIVATE only to owner) + not deleted
    op.execute(
        """
        CREATE POLICY rag_documents_select ON rag_documents
        FOR SELECT
        USING (
            tenant_id = current_setting('app.tenant_id', true)
            AND deleted_at IS NULL
            AND (
                visibility = 'TEAM'
                OR (visibility = 'PRIVATE'
                    AND owner_user_id = current_setting('app.user_id', true))
            )
        )
    """
    )

    # INSERT: tenant match + visibility/owner consistency
    op.execute(
        """
        CREATE POLICY rag_documents_insert ON rag_documents
        FOR INSERT
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
            AND (
                (visibility = 'TEAM' AND owner_user_id IS NULL)
                OR (visibility = 'PRIVATE'
                    AND owner_user_id = current_setting('app.user_id', true))
            )
        )
    """
    )

    # UPDATE: same as SELECT for USING, same as INSERT for WITH CHECK
    op.execute(
        """
        CREATE POLICY rag_documents_update ON rag_documents
        FOR UPDATE
        USING (
            tenant_id = current_setting('app.tenant_id', true)
            AND deleted_at IS NULL
            AND (
                visibility = 'TEAM'
                OR (visibility = 'PRIVATE'
                    AND owner_user_id = current_setting('app.user_id', true))
            )
        )
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
            AND (
                (visibility = 'TEAM' AND owner_user_id IS NULL)
                OR (visibility = 'PRIVATE'
                    AND owner_user_id = current_setting('app.user_id', true))
            )
        )
    """
    )

    # DELETE: same as SELECT
    op.execute(
        """
        CREATE POLICY rag_documents_delete ON rag_documents
        FOR DELETE
        USING (
            tenant_id = current_setting('app.tenant_id', true)
            AND deleted_at IS NULL
            AND (
                visibility = 'TEAM'
                OR (visibility = 'PRIVATE'
                    AND owner_user_id = current_setting('app.user_id', true))
            )
        )
    """
    )

    # -- Row-Level Security: rag_document_chunks ------------------------------

    op.execute("ALTER TABLE rag_document_chunks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_document_chunks FORCE ROW LEVEL SECURITY")

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    # -- Row-Level Security: rag_chunk_embeddings -----------------------------

    op.execute("ALTER TABLE rag_chunk_embeddings ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_chunk_embeddings FORCE ROW LEVEL SECURITY")

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    # -- Row-Level Security: rag_ingestion_runs -------------------------------

    op.execute("ALTER TABLE rag_ingestion_runs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_ingestion_runs FORCE ROW LEVEL SECURITY")

    op.execute(
        """
        CREATE POLICY rag_ingestion_runs_select ON rag_ingestion_runs
        FOR SELECT
        USING (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """
    )

    op.execute(
        """
        CREATE POLICY rag_ingestion_runs_insert ON rag_ingestion_runs
        FOR INSERT
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """
    )

    op.execute(
        """
        CREATE POLICY rag_ingestion_runs_update ON rag_ingestion_runs
        FOR UPDATE
        USING (
            tenant_id = current_setting('app.tenant_id', true)
        )
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """
    )

    op.execute(
        """
        CREATE POLICY rag_ingestion_runs_delete ON rag_ingestion_runs
        FOR DELETE
        USING (
            tenant_id = current_setting('app.tenant_id', true)
        )
    """
    )

    # -- Row-Level Security: rag_ingestion_items ------------------------------

    op.execute("ALTER TABLE rag_ingestion_items ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE rag_ingestion_items FORCE ROW LEVEL SECURITY")

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )

    op.execute(
        """
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
    """
    )


def downgrade() -> None:
    # -- Drop RLS policies (reverse order) ------------------------------------

    # rag_ingestion_items
    op.execute(
        "DROP POLICY IF EXISTS rag_ingestion_items_delete ON rag_ingestion_items"
    )
    op.execute(
        "DROP POLICY IF EXISTS rag_ingestion_items_update ON rag_ingestion_items"
    )
    op.execute(
        "DROP POLICY IF EXISTS rag_ingestion_items_insert ON rag_ingestion_items"
    )
    op.execute(
        "DROP POLICY IF EXISTS rag_ingestion_items_select ON rag_ingestion_items"
    )
    op.execute("ALTER TABLE rag_ingestion_items DISABLE ROW LEVEL SECURITY")

    # rag_ingestion_runs
    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_delete ON rag_ingestion_runs")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_update ON rag_ingestion_runs")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_insert ON rag_ingestion_runs")
    op.execute("DROP POLICY IF EXISTS rag_ingestion_runs_select ON rag_ingestion_runs")
    op.execute("ALTER TABLE rag_ingestion_runs DISABLE ROW LEVEL SECURITY")

    # rag_chunk_embeddings
    op.execute("DROP POLICY IF EXISTS rag_embeddings_delete ON rag_chunk_embeddings")
    op.execute("DROP POLICY IF EXISTS rag_embeddings_update ON rag_chunk_embeddings")
    op.execute("DROP POLICY IF EXISTS rag_embeddings_insert ON rag_chunk_embeddings")
    op.execute("DROP POLICY IF EXISTS rag_embeddings_select ON rag_chunk_embeddings")
    op.execute("ALTER TABLE rag_chunk_embeddings DISABLE ROW LEVEL SECURITY")

    # rag_document_chunks
    op.execute("DROP POLICY IF EXISTS rag_chunks_delete ON rag_document_chunks")
    op.execute("DROP POLICY IF EXISTS rag_chunks_update ON rag_document_chunks")
    op.execute("DROP POLICY IF EXISTS rag_chunks_insert ON rag_document_chunks")
    op.execute("DROP POLICY IF EXISTS rag_chunks_select ON rag_document_chunks")
    op.execute("ALTER TABLE rag_document_chunks DISABLE ROW LEVEL SECURITY")

    # rag_documents
    op.execute("DROP POLICY IF EXISTS rag_documents_delete ON rag_documents")
    op.execute("DROP POLICY IF EXISTS rag_documents_update ON rag_documents")
    op.execute("DROP POLICY IF EXISTS rag_documents_insert ON rag_documents")
    op.execute("DROP POLICY IF EXISTS rag_documents_select ON rag_documents")
    op.execute("ALTER TABLE rag_documents DISABLE ROW LEVEL SECURITY")

    # -- Drop tables (reverse FK order) ---------------------------------------
    op.execute("DROP TABLE IF EXISTS rag_ingestion_items CASCADE")
    op.execute("DROP TABLE IF EXISTS rag_ingestion_runs CASCADE")
    op.execute("DROP TABLE IF EXISTS rag_chunk_embeddings CASCADE")
    op.execute("DROP TABLE IF EXISTS rag_document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS rag_documents CASCADE")
