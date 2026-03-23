import json
import sqlite3
from hpm.patterns import pattern_from_dict


class SQLiteStore:
    """
    SQLite-backed PatternStore. Persists patterns across sessions.
    Schema: patterns(id TEXT, agent_id TEXT, pattern_json TEXT, weight REAL, PRIMARY KEY (id, agent_id)).
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS patterns (
        id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        pattern_json TEXT NOT NULL,
        weight REAL NOT NULL,
        PRIMARY KEY (id, agent_id)
    )
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(self._SCHEMA)

    def save(self, pattern, weight: float, agent_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patterns (id, agent_id, pattern_json, weight) "
                "VALUES (?, ?, ?, ?)",
                (pattern.id, agent_id, json.dumps(pattern.to_dict()), weight),
            )

    def load(self, pattern_id: str, agent_id: str) -> tuple:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE id = ? AND agent_id = ?",
                (pattern_id, agent_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"{pattern_id} for {agent_id}")
        return pattern_from_dict(json.loads(row[0])), row[1]

    def query(self, agent_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
        return [(pattern_from_dict(json.loads(r[0])), r[1]) for r in rows]

    def query_all(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pattern_json, weight, agent_id FROM patterns"
            ).fetchall()
        return [(pattern_from_dict(json.loads(r[0])), r[1], r[2]) for r in rows]

    def delete(self, pattern_id: str, agent_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM patterns WHERE id = ? AND agent_id = ?", (pattern_id, agent_id))

    def update_weight(self, pattern_id: str, agent_id: str, weight: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE patterns SET weight = ? WHERE id = ? AND agent_id = ?",
                (weight, pattern_id, agent_id),
            )
