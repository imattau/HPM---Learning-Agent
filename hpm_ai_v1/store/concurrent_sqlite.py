"""Concurrent SQLite Pattern Store for HPM AI v1.

Uses Write-Ahead Logging (WAL) and timeout retries to support multiple 
agents (and the refactoring sandbox) reading/writing concurrently.
"""
import json
import sqlite3
import time
from typing import Optional

class ConcurrentSQLiteStore:
    """
    Thread-safe, multi-process SQLite store.
    Schema: patterns(id TEXT, agent_id TEXT, pattern_json TEXT, weight REAL, PRIMARY KEY (id, agent_id))
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS patterns (
        id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        pattern_json TEXT NOT NULL,
        weight REAL NOT NULL,
        PRIMARY KEY (id, agent_id)
    );
    CREATE INDEX IF NOT EXISTS idx_agent_id ON patterns(agent_id);
    """

    def __init__(self, db_path: str, timeout: float = 10.0):
        self.db_path = db_path
        self.timeout = timeout
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        # Enable WAL mode for concurrent reads/writes
        conn.execute("PRAGMA journal_mode=WAL;")
        # Synchronous NORMAL is faster in WAL mode while remaining safe
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    def save(self, pattern, weight: float, agent_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patterns (id, agent_id, pattern_json, weight) VALUES (?, ?, ?, ?)",
                (pattern.id, agent_id, json.dumps(pattern.to_dict()), weight),
            )

    def load_json(self, pattern_id: str, agent_id: str) -> tuple[dict, float]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE id = ? AND agent_id = ?",
                (pattern_id, agent_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Pattern {pattern_id} for agent {agent_id} not found.")
        return json.loads(row[0]), row[1]

    def query_all(self, agent_id: Optional[str] = None) -> list[tuple[dict, float, str]]:
        with self._conn() as conn:
            if agent_id:
                rows = conn.execute(
                    "SELECT pattern_json, weight, agent_id FROM patterns WHERE agent_id = ?", 
                    (agent_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT pattern_json, weight, agent_id FROM patterns").fetchall()
        return [(json.loads(r[0]), r[1], r[2]) for r in rows]

    def delete(self, pattern_id: str, agent_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM patterns WHERE id = ? AND agent_id = ?", (pattern_id, agent_id))

    def update_weight(self, pattern_id: str, agent_id: str, weight: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE patterns SET weight = ? WHERE id = ? AND agent_id = ?",
                (weight, pattern_id, agent_id),
            )
