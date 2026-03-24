"""Concurrent SQLite Pattern Store for HPM AI v1.

Uses Write-Ahead Logging (WAL) and timeout retries to support multiple 
agents (and the refactoring sandbox) reading/writing concurrently.
"""
import json
import sqlite3
import time
from typing import Optional, List, Tuple, Any
from hpm.patterns.factory import pattern_from_dict

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
        parent_l3_id TEXT,
        recombination_op TEXT,
        PRIMARY KEY (id, agent_id)
    );
    CREATE INDEX IF NOT EXISTS idx_agent_id ON patterns(agent_id);

    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str, timeout: float = 10.0):
        self.db_path = db_path
        self.timeout = timeout
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    def set_metadata(self, key: str, value: Any) -> None:
        """Store arbitrary JSON-serializable metadata."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value))
            )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve stored metadata."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value_json FROM metadata WHERE key = ?",
                (key,)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return default

    def save(self, pattern, weight: float, agent_id: str, parent_l3_id: Optional[str] = None, recombination_op: Optional[str] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO patterns 
                   (id, agent_id, pattern_json, weight, parent_l3_id, recombination_op) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pattern.id, agent_id, json.dumps(pattern.to_dict()), weight, parent_l3_id, recombination_op),
            )

    def load(self, pattern_id: str, agent_id: str) -> Tuple[object, float]:
        """Load a pattern object and its weight."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE id = ? AND agent_id = ?",
                (pattern_id, agent_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Pattern {pattern_id} for agent {agent_id} not found.")
        return pattern_from_dict(json.loads(row[0])), float(row[1])

    def query(self, agent_id: str) -> List[Tuple[object, float]]:
        """Return all patterns for a specific agent."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE agent_id = ?", 
                (agent_id,)
            ).fetchall()
        return [(pattern_from_dict(json.loads(r[0])), float(r[1])) for r in rows]

    def query_all(self, agent_id: Optional[str] = None) -> List[Tuple[dict, float, str]]:
        """Return raw dicts for all patterns (or filtered by agent)."""
        with self._conn() as conn:
            if agent_id:
                rows = conn.execute(
                    "SELECT pattern_json, weight, agent_id FROM patterns WHERE agent_id = ?", 
                    (agent_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT pattern_json, weight, agent_id FROM patterns").fetchall()
        return [(json.loads(r[0]), float(r[1]), r[2]) for r in rows]

    def delete(self, pattern_id: str, agent_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM patterns WHERE id = ? AND agent_id = ?", (pattern_id, agent_id))

    def update_weight(self, pattern_id: str, agent_id: str, weight: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE patterns SET weight = ? WHERE id = ? AND agent_id = ?",
                (weight, pattern_id, agent_id),
            )
