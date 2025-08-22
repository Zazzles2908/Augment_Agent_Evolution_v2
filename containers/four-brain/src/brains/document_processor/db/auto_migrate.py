"""
Minimal auto-migration for Brain-4 (Docling) database objects.
Runs at startup if MIGRATE_ON_START=true. Zero fabrication: logs real errors.
"""
from __future__ import annotations
import asyncpg
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "migrations_sql"

async def run_auto_migrations(database_url: str) -> bool:
    """
    Execute minimal SQL migrations required for Brain-4 endpoints to work.
    Currently: ensure augment_agent schema and documents table exist.
    """
    try:
        conn = await asyncpg.connect(database_url, timeout=10.0)
        try:
            for sql_file in sorted(SQL_DIR.glob("*.sql")):
                sql = sql_file.read_text(encoding="utf-8")
                await conn.execute(sql)
                logger.info(f"Applied migration: {sql_file.name}")
        finally:
            await conn.close()
        return True
    except Exception as e:
        logger.error(f"Auto-migration failed: {e}")
        return False

