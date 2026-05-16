from __future__ import annotations

import atexit
import struct
import threading
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence

import numpy as np

from hpm_ai_v6.hpm_model.core.cell import Cell


def _load_sqlite_vec(con):
    """Load the sqlite-vec extension into an open connection."""
    try:
        import sqlite_vec
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        return True
    except Exception:
        return False


def _emb_to_bytes(arr: np.ndarray) -> bytes:
    """Encode a float64 numpy array as float32 little-endian bytes for sqlite-vec."""
    return arr.astype(np.float32).tobytes()


def _bytes_to_emb(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=float)


class PatternPager:
    """
    Async disk-backed pattern archive using SQLite + sqlite-vec.

    Drops the lowest-weight patterns when active patterns exceed max_active_patterns.
    Public API is identical to the previous JSONL-based implementation.
    """

    def __init__(
        self,
        cache_dir: str,
        agent_name: str,
        max_active_patterns: int = 5000,
        spill_fraction: float = 0.25,
    ):
        import os, sqlite3
        self.cache_dir = cache_dir
        self.agent_name = agent_name
        self.max_active_patterns = max_active_patterns
        self.spill_fraction = spill_fraction

        self.archive_dir = os.path.join(cache_dir, agent_name)
        os.makedirs(self.archive_dir, exist_ok=True)
        self.db_path = os.path.join(self.archive_dir, "patterns.db")

        self._lock = threading.Lock()
        self._pending: Dict[str, Dict[str, object]] = {}
        self._vec_ready = False  # True once vec_patterns table is created
        self._dim: Optional[int] = None  # fixed once first pattern written

        self._con = self._open_connection()
        self._init_schema()

        self._queue: "Queue[Optional[Dict[str, object]]]" = Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker.start()
        atexit.register(self.close)

    def _open_connection(self):
        import sqlite3
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        _load_sqlite_vec(con)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_schema(self):
        with self._lock:
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
                    name    TEXT UNIQUE NOT NULL,
                    dim     INTEGER NOT NULL,
                    source  TEXT,
                    target  TEXT,
                    weight  REAL NOT NULL DEFAULT 1.0,
                    embedding BLOB NOT NULL
                )
            """)
            self._con.commit()
            # Restore dim and vec table if patterns already exist
            row = self._con.execute("SELECT value FROM meta WHERE key='dim'").fetchone()
            if row:
                self._dim = int(row[0])
                self._ensure_vec_table(self._dim)

    def _ensure_vec_table(self, dim: int) -> None:
        """Create vec_patterns virtual table for given dim if not already created."""
        if self._vec_ready:
            return
        try:
            self._con.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_patterns USING vec0(
                    embedding float[{dim}]
                )
            """)
            self._con.commit()
            self._vec_ready = True
        except Exception:
            pass  # sqlite-vec not available; load_nearest falls back to cosine scan

    @staticmethod
    def _serialize_cell(cell: Cell) -> Dict[str, object]:
        emb = cell.as_numpy()
        return {
            "name": cell.name,
            "dim": cell.dim,
            "embedding": emb.tolist(),
            "source": cell.source.name if cell.source is not None else None,
            "target": cell.target.name if cell.target is not None else None,
            "weight": float(cell.weight),
        }

    @staticmethod
    def _deserialize_cell(payload: Dict[str, object], lookup: Dict[str, Cell]) -> Cell:
        source_name = payload.get("source")
        target_name = payload.get("target")
        emb = payload.get("embedding")
        if isinstance(emb, (bytes, bytearray)):
            emb = _bytes_to_emb(emb)
        else:
            emb = np.asarray(emb, dtype=float)
        return Cell(
            name=str(payload["name"]),
            dim=int(payload["dim"]),
            embedding=emb,
            source=lookup.get(str(source_name)) if source_name else None,
            target=lookup.get(str(target_name)) if target_name else None,
            weight=float(payload.get("weight", 1.0)),
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _to_numpy(vector: object) -> np.ndarray:
        if isinstance(vector, np.ndarray):
            return vector.astype(float, copy=False)
        try:
            import torch
            if isinstance(vector, torch.Tensor):
                return vector.detach().cpu().numpy().astype(float, copy=False)
        except Exception:
            pass
        return np.asarray(vector, dtype=float)

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if payload is not None:
                    self._write_payload(payload)
                    with self._lock:
                        self._pending.pop(str(payload["name"]), None)
            finally:
                self._queue.task_done()

    def _write_payload(self, payload: Dict[str, object]) -> None:
        name = str(payload["name"])
        dim = int(payload["dim"])
        emb_list = payload.get("embedding")
        if isinstance(emb_list, (bytes, bytearray)):
            emb_arr = _bytes_to_emb(emb_list)
        else:
            emb_arr = np.asarray(emb_list, dtype=float)
        emb_bytes = _emb_to_bytes(emb_arr)

        with self._lock:
            if self._dim is None:
                self._dim = len(emb_arr)
                self._con.execute("INSERT OR IGNORE INTO meta VALUES ('dim', ?)", (str(self._dim),))
                self._ensure_vec_table(self._dim)

            if len(emb_arr) != self._dim:
                return  # incompatible dim, drop silently

            self._con.execute(
                "INSERT OR REPLACE INTO patterns(name, dim, source, target, weight, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, dim, payload.get("source"), payload.get("target"),
                 float(payload.get("weight", 1.0)), emb_bytes),
            )
            if self._vec_ready:
                rowid = self._con.execute(
                    "SELECT rowid FROM patterns WHERE name=?", (name,)
                ).fetchone()[0]
                self._con.execute(
                    "INSERT OR REPLACE INTO vec_patterns(rowid, embedding) VALUES (?, ?)",
                    (rowid, emb_bytes),
                )
            self._con.commit()

    def enqueue_save(self, cell: Cell) -> None:
        payload = self._serialize_cell(cell)
        with self._lock:
            self._pending[str(payload["name"])] = payload
        self._queue.put(payload)

    def save(self, cell: Cell) -> None:
        self.enqueue_save(cell)

    def has(self, name: str) -> bool:
        with self._lock:
            if name in self._pending:
                return True
            row = self._con.execute(
                "SELECT 1 FROM patterns WHERE name=? LIMIT 1", (name,)
            ).fetchone()
            return row is not None

    def load(self, name: str, lookup: Dict[str, Cell]) -> Optional[Cell]:
        with self._lock:
            pending = self._pending.get(name)
        if pending is not None:
            return self._deserialize_cell(pending, lookup)
        with self._lock:
            row = self._con.execute(
                "SELECT name, dim, source, target, weight, embedding FROM patterns WHERE name=?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        payload = {"name": row[0], "dim": row[1], "source": row[2],
                   "target": row[3], "weight": row[4], "embedding": row[5]}
        return self._deserialize_cell(payload, lookup)

    def load_nearest(
        self,
        query_embedding: object,
        lookup: Dict[str, Cell],
        min_similarity: float = 0.85,
        exclude_names: Optional[Sequence[str]] = None,
    ) -> Optional[Cell]:
        query = self._to_numpy(query_embedding)
        if query.size == 0:
            return None
        exclude = set(exclude_names or [])

        # Try sqlite-vec KNN first
        if self._vec_ready and self._dim and len(query) == self._dim:
            try:
                q_bytes = _emb_to_bytes(query)
                rows = self._con.execute(
                    "SELECT p.name, p.dim, p.source, p.target, p.weight, p.embedding "
                    "FROM vec_patterns v JOIN patterns p ON v.rowid = p.rowid "
                    "WHERE v.embedding MATCH ? AND k = 20 "
                    "ORDER BY distance",
                    (q_bytes,),
                ).fetchall()
                for row in rows:
                    name = row[0]
                    if name in exclude:
                        continue
                    emb = _bytes_to_emb(row[5])
                    score = self._cosine_similarity(query, emb)
                    if score >= min_similarity:
                        payload = {"name": row[0], "dim": row[1], "source": row[2],
                                   "target": row[3], "weight": row[4], "embedding": emb}
                        return self._deserialize_cell(payload, lookup)
                return None
            except Exception:
                pass  # fall through to full scan

        # Fallback: full cosine scan
        payloads = self.iter_index_payloads()
        best_payload = None
        best_score = float("-inf")
        for payload in payloads:
            name = str(payload.get("name", ""))
            if name in exclude:
                continue
            emb_raw = payload.get("embedding")
            if emb_raw is None:
                continue
            if isinstance(emb_raw, (bytes, bytearray)):
                candidate = _bytes_to_emb(emb_raw)
            else:
                candidate = np.asarray(emb_raw, dtype=float)
            if candidate.shape != query.shape:
                continue
            score = self._cosine_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_payload = payload
        if best_payload is None or best_score < min_similarity:
            return None
        return self._deserialize_cell(best_payload, lookup)

    def iter_index_payloads(self) -> List[Dict[str, object]]:
        with self._lock:
            pending_vals = list(self._pending.values())
            rows = self._con.execute(
                "SELECT name, dim, source, target, weight, embedding FROM patterns"
            ).fetchall()

        persisted_names = {str(p["name"]) for p in pending_vals}
        result = list(pending_vals)
        for row in rows:
            name = row[0]
            if name not in persisted_names:
                emb = _bytes_to_emb(row[5])
                result.append({
                    "name": name, "dim": row[1], "source": row[2],
                    "target": row[3], "weight": row[4], "embedding": emb.tolist(),
                })
        return result

    def load_from_payload(self, payload: Dict[str, object], lookup: Dict[str, Cell]) -> Cell:
        return self._deserialize_cell(payload, lookup)

    def select_evictions(self, patterns: Sequence[Cell], weights: Sequence[float]) -> List[int]:
        if len(patterns) <= self.max_active_patterns:
            return []
        excess = len(patterns) - self.max_active_patterns
        spill_count = max(excess, int(len(patterns) * self.spill_fraction))
        spill_count = max(1, min(spill_count, len(patterns)))
        weighted = sorted(enumerate(weights), key=lambda item: float(item[1]))
        return [idx for idx, _ in weighted[:spill_count]]

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self.flush()
        self._stop_event.set()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        with self._lock:
            try:
                self._con.close()
            except Exception:
                pass
