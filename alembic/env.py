"""Alembic environment configuration for rag-platform.

Uses psycopg2 (sync) for migrations. Mirrors the 3-priority connection
logic from rag_service/db.py with a postgresql+psycopg2:// prefix.
"""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _get_url() -> str:
    """Build database URL from environment."""
    if url := os.environ.get("DATABASE_URL"):
        # Ensure psycopg2 driver prefix for Alembic
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    sslmode = os.environ.get("DB_SSLMODE", "disable")
    return (
        f"postgresql+psycopg2://{os.environ.get('DB_USER', 'apexflow')}:"
        f"{os.environ.get('DB_PASSWORD', 'apexflow')}@"
        f"{os.environ.get('DB_HOST', 'localhost')}:"
        f"{os.environ.get('DB_PORT', '5432')}/"
        f"{os.environ.get('DB_NAME', 'apexflow')}?sslmode={sslmode}"
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
