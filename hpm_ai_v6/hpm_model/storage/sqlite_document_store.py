from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional


class SQLiteDocumentStore:
    """Tiny SQLite-backed JSON document store."""

    def __init__(self, db_path: str, table_name: str = "documents") -> None:
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    name TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def save(self, name: str, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name} (name, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (name, json.dumps(payload), time.time()),
            )

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload FROM {self.table_name} WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
