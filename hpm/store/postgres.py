import json
from ..patterns.gaussian import GaussianPattern


class PostgreSQLStore:
    """
    PatternStore backed by PostgreSQL. Drop-in replacement for SQLiteStore.
    Schema matches SQLiteStore: patterns(id, agent_id, pattern_json, weight).

    Parameters
    ----------
    dsn : str
        libpq connection string, e.g. "postgresql://user:pass@host/dbname"
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS patterns (
        id           TEXT PRIMARY KEY,
        agent_id     TEXT NOT NULL,
        pattern_json TEXT NOT NULL,
        weight       REAL NOT NULL
    )
    """
    _INDEX = "CREATE INDEX IF NOT EXISTS idx_patterns_agent ON patterns(agent_id)"

    def __init__(self, dsn: str):
        import psycopg2
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._SCHEMA)
            cur.execute(self._INDEX)
        self._conn.commit()

    def save(self, pattern, weight: float, agent_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patterns (id, agent_id, pattern_json, weight) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (id) DO UPDATE SET"
                " weight = EXCLUDED.weight, pattern_json = EXCLUDED.pattern_json",
                (pattern.id, agent_id, json.dumps(pattern.to_dict()), weight),
            )
        self._conn.commit()

    def query(self, agent_id: str) -> list:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT pattern_json, weight FROM patterns WHERE agent_id = %s", (agent_id,)
            )
            return [
                (GaussianPattern.from_dict(json.loads(row[0])), row[1])
                for row in cur.fetchall()
            ]

    def query_all(self) -> list:
        """Return all (pattern, weight, agent_id) triples. Matches SQLiteStore signature."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT pattern_json, weight, agent_id FROM patterns")
            return [
                (GaussianPattern.from_dict(json.loads(row[0])), row[1], row[2])
                for row in cur.fetchall()
            ]

    def delete(self, pattern_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM patterns WHERE id = %s", (pattern_id,))
        self._conn.commit()

    def update_weight(self, pattern_id: str, weight: float) -> None:
        """Silent no-op if pattern_id does not exist -- matches SQLiteStore behaviour."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE patterns SET weight = %s WHERE id = %s", (weight, pattern_id)
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
