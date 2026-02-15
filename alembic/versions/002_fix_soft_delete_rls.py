"""Remove deleted_at IS NULL from rag_documents SELECT policy.

Revision ID: 002
Create Date: 2026-02-16

PostgreSQL applies the SELECT USING expression to the **new row** during
UPDATE.  The original SELECT policy included ``deleted_at IS NULL``, which
made ``soft_delete`` (UPDATE … SET deleted_at = NOW()) always fail with
``InsufficientPrivilegeError`` because the new row has ``deleted_at IS NOT
NULL``.

Fix: move the ``deleted_at IS NULL`` filter from the RLS SELECT policy to
the application layer (store queries).  Tenant/visibility isolation
remains in RLS.  The chunk and embedding SELECT policies independently
check ``d.deleted_at IS NULL`` via their EXISTS sub-queries, so search
results are still correctly filtered.
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the old SELECT policy that blocks soft_delete
    op.execute("DROP POLICY rag_documents_select ON rag_documents")

    # Recreate without deleted_at IS NULL
    op.execute(
        """
        CREATE POLICY rag_documents_select ON rag_documents
        FOR SELECT
        USING (
            tenant_id = current_setting('app.tenant_id', true)
            AND (
                visibility = 'TEAM'
                OR (visibility = 'PRIVATE'
                    AND owner_user_id = current_setting('app.user_id', true))
            )
        )
    """
    )


def downgrade() -> None:
    op.execute("DROP POLICY rag_documents_select ON rag_documents")

    # Restore original policy with deleted_at IS NULL
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
