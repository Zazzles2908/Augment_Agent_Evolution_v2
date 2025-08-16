#!/usr/bin/env python3
"""
Supabase Client Helper for Brain-3 Augment
Provides functions to connect to Postgres and manage agentic_memory table

This module provides helper functions for Brain-3 to interact with the Supabase
PostgreSQL database, specifically for managing the agentic_memory table.

Zero Fabrication Policy: ENFORCED
All implementations use real psycopg2 and Supabase connections.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import structlog

logger = structlog.get_logger(__name__)

class SupabaseClient:
    """Supabase client for Brain-3 agentic memory operations"""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize Supabase client"""
        self.connection_string = connection_string or os.getenv("POSTGRES_CONNECTION_STRING")
        self.connection = None

        if not self.connection_string:
            logger.warning("No PostgreSQL connection string provided")

    def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            if not self.connection_string:
                return False

            self.connection = psycopg2.connect(self.connection_string)
            logger.info("✅ Supabase PostgreSQL connection established")
            return True

        except Exception as e:
            logger.error("❌ Supabase PostgreSQL connection failed", error=str(e))
            return False

    def disconnect(self):
        """Disconnect from PostgreSQL database"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Supabase PostgreSQL connection closed")

    def insert_agentic_memory(self, task_id: str, prompt: str,
                            outcome_score: float, metadata: Dict[str, Any] = None) -> bool:
        """Insert a record into the agentic_memory table"""
        try:
            if not self.connection:
                if not self.connect():
                    return False

            cursor = self.connection.cursor()

            cursor.execute("""
                INSERT INTO agentic_memory (task_id, prompt, outcome_score, metadata)
                VALUES (%s, %s, %s, %s)
            """, (
                task_id,
                prompt,
                outcome_score,
                json.dumps(metadata or {})
            ))

            self.connection.commit()
            cursor.close()

            logger.debug("Agentic memory record inserted",
                        task_id=task_id, score=outcome_score)
            return True

        except Exception as e:
            logger.error("Failed to insert agentic memory record",
                        task_id=task_id, error=str(e))
            if self.connection:
                self.connection.rollback()
            return False

    def get_agentic_memory(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agentic memory record by task_id"""
        try:
            if not self.connection:
                if not self.connect():
                    return None

            cursor = self.connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM agentic_memory WHERE task_id = %s
            """, (task_id,))

            result = cursor.fetchone()
            cursor.close()

            if result:
                return dict(result)
            return None

        except Exception as e:
            logger.error("Failed to get agentic memory record",
                        task_id=task_id, error=str(e))
            return None

    def get_recent_agentic_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent agentic memory records"""
        try:
            if not self.connection:
                if not self.connect():
                    return []

            cursor = self.connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM agentic_memory
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error("Failed to get recent agentic memories", error=str(e))
            return []

    def get_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for the last N hours"""
        try:
            if not self.connection:
                if not self.connect():
                    return {}

            cursor = self.connection.cursor()

            # Get stats for the specified time period
            cursor.execute("""
                SELECT
                    COUNT(*) as total_tasks,
                    AVG(outcome_score) as average_score,
                    MAX(outcome_score) as best_score,
                    MIN(outcome_score) as worst_score,
                    COUNT(CASE WHEN outcome_score > 0.7 THEN 1 END) as high_score_tasks,
                    COUNT(CASE WHEN outcome_score < 0.3 THEN 1 END) as low_score_tasks
                FROM agentic_memory
                WHERE created_at >= NOW() - INTERVAL '%s hours'
            """, (hours,))

            result = cursor.fetchone()
            cursor.close()

            if result:
                total_tasks, avg_score, best_score, worst_score, high_score, low_score = result
                return {
                    "total_tasks": total_tasks or 0,
                    "average_score": float(avg_score) if avg_score else 0.0,
                    "best_score": float(best_score) if best_score else 0.0,
                    "worst_score": float(worst_score) if worst_score else 0.0,
                    "high_score_tasks": high_score or 0,
                    "low_score_tasks": low_score or 0,
                    "time_period_hours": hours
                }

            return {}

        except Exception as e:
            logger.error("Failed to get performance stats", error=str(e))
            return {}

    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            if not self.connection:
                return self.connect()

            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()

            return True

        except Exception as e:
            logger.warning("Supabase health check failed", error=str(e))
            return False

# Global Supabase client instance
_supabase_client: Optional[SupabaseClient] = None

def get_supabase_client() -> SupabaseClient:
    """Get or create the global Supabase client instance"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client