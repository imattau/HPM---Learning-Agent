from __future__ import annotations

import os
import pickle
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ReasoningGraphSnapshot:
    signature: str
    payload: bytes
    updated_at: float


class ReasoningGraphStore:
    """SQLite-backed cache for a reasoning graph snapshot."""

    def __init__(self, cache_dir: str, agent_name: str = "reasoning") -> None:
        self.cache_dir = cache_dir
        self.agent_name = agent_name
        self.archive_dir = os.path.join(cache_dir, agent_name)
        os.makedirs(self.archive_dir, exist_ok=True)
        self.db_path = os.path.join(self.archive_dir, "reasoning_graph.db")
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_snapshot (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    signature TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def save(self, signature: str, snapshot: dict[str, Any]) -> None:
        payload = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_snapshot (id, signature, payload, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    signature = excluded.signature,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (signature, payload, time.time()),
            )

    def load(self) -> Optional[ReasoningGraphSnapshot]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT signature, payload, updated_at FROM graph_snapshot WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        return ReasoningGraphSnapshot(signature=row[0], payload=row[1], updated_at=float(row[2]))

    @staticmethod
    def unpack(snapshot: ReasoningGraphSnapshot) -> dict[str, Any]:
        return pickle.loads(snapshot.payload)
