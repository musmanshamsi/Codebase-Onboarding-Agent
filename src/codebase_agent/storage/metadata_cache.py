"""SQLite Metadata Cache wrapper (Database Design Section 5)."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


class MetadataCache:
    """Manages SQLite cache database storing file index states and run logs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_db_dir()
        self.initialize_schema()

    def _ensure_db_dir(self):
        """Creates parent directory for SQLite database if missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connected sqlite3 database object with row factory set."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self):
        """Initializes database schema per Database Design Document Section 5."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table: file_index_state (Database Design Section 5.2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_index_state (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    language TEXT NOT NULL,
                    last_indexed_at TEXT NOT NULL,
                    symbol_count INTEGER NOT NULL DEFAULT 0,
                    parse_status TEXT NOT NULL,
                    parse_error TEXT
                )
            """)

            # Table: index_runs (Database Design Section 5.3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS index_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    files_processed INTEGER NOT NULL DEFAULT 0,
                    files_failed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL
                )
            """)

            # Table: query_log (Database Design Section 5.4)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    asked_at TEXT NOT NULL,
                    top_k_chunk_ids TEXT,
                    graph_expanded_ids TEXT,
                    answer_returned TEXT,
                    insufficient_context BOOLEAN NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    # --- Index Run Operations ---

    def start_index_run(self, run_type: str = "full") -> int:
        """Records the start of an index run into index_runs table."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO index_runs (run_type, started_at, files_processed, files_failed, status)
                VALUES (?, ?, 0, 0, 'running')
            """, (run_type, now_iso))
            conn.commit()
            return cursor.lastrowid

    def finish_index_run(self, run_id: int, files_processed: int, files_failed: int, status: str = "completed"):
        """Updates the completion of an index run."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE index_runs
                SET completed_at = ?, files_processed = ?, files_failed = ?, status = ?
                WHERE run_id = ?
            """, (now_iso, files_processed, files_failed, status, run_id))
            conn.commit()

    def get_last_run(self) -> Optional[Dict[str, Any]]:
        """Fetches the most recent index run record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM index_runs ORDER BY started_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- File Index State Operations ---

    def get_file_state(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieves index state for a given relative file path."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM file_index_state WHERE file_path = ?
            """, (file_path,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_file_states(self) -> Dict[str, Dict[str, Any]]:
        """Returns mapping of relative file paths to their index state records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM file_index_state")
            rows = cursor.fetchall()
            return {row["file_path"]: dict(row) for row in rows}

    def upsert_file_state(
        self,
        file_path: str,
        content_hash: str,
        language: str,
        symbol_count: int = 0,
        parse_status: str = "success",
        parse_error: Optional[str] = None
    ):
        """Inserts or replaces file index state record."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO file_index_state (
                    file_path, content_hash, language, last_indexed_at,
                    symbol_count, parse_status, parse_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path, content_hash, language, now_iso,
                symbol_count, parse_status, parse_error
            ))
            conn.commit()

    def delete_file_state(self, file_path: str):
        """Deletes file record upon file removal from repo."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM file_index_state WHERE file_path = ?", (file_path,))
            conn.commit()
