import json
import sqlite3
from typing import Optional, List, Tuple
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

    def _find_agent_id(self, pattern_id: str, agent_id: str) -> str:
        """Helper to find the correct agent_id for a pattern_id, with fallback."""
        if agent_id != "default_agent":
            return agent_id
            
        with self._conn() as conn:
            row = conn.execute(
                "SELECT agent_id FROM patterns WHERE id = ?",
                (pattern_id,)
            ).fetchone()
            if row:
                return row[0]
        return agent_id

    def save(self, pattern, weight: float, agent_id: str = "default_agent") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patterns (id, agent_id, pattern_json, weight) "
                "VALUES (?, ?, ?, ?)",
                (pattern.id, agent_id, json.dumps(pattern.to_dict()), weight),
            )

    def load(self, pattern_id: str, agent_id: str = "default_agent") -> Tuple[object, float]:
        actual_aid = self._find_agent_id(pattern_id, agent_id)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE id = ? AND agent_id = ?",
                (pattern_id, actual_aid),
            ).fetchone()
        if row is None:
            raise KeyError(f"{pattern_id} for {agent_id}")
        return pattern_from_dict(json.loads(row[0])), float(row[1])

    def query(self, agent_id: str = "default_agent") -> List[Tuple[object, float]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
        return [(pattern_from_dict(json.loads(r[0])), float(r[1])) for r in rows]

    def query_all(self) -> List[Tuple[object, float, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pattern_json, weight, agent_id FROM patterns"
            ).fetchall()
        return [(pattern_from_dict(json.loads(r[0])), float(r[1]), r[2]) for r in rows]

    def delete(self, pattern_id: str, agent_id: str = "default_agent") -> None:
        actual_aid = self._find_agent_id(pattern_id, agent_id)
        with self._conn() as conn:
            conn.execute("DELETE FROM patterns WHERE id = ? AND agent_id = ?", (pattern_id, actual_aid))

    def update_weight(self, pattern_id: str, agent_id: str, weight: Optional[float] = None) -> None:
        # Backward compatibility for (pattern_id, weight)
        if weight is None:
            actual_weight = float(agent_id)
            target_aid = "default_agent"
        else:
            actual_weight = weight
            target_aid = agent_id

        actual_aid = self._find_agent_id(pattern_id, target_aid)
        with self._conn() as conn:
            conn.execute(
                "UPDATE patterns SET weight = ? WHERE id = ? AND agent_id = ?",
                (actual_weight, pattern_id, actual_aid),
            )
            
    def has(self, pattern_id: str, agent_id: str = "default_agent") -> bool:
        actual_aid = self._find_agent_id(pattern_id, agent_id)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM patterns WHERE id = ? AND agent_id = ?",
                (pattern_id, actual_aid),
            ).fetchone()
        return row is not None
