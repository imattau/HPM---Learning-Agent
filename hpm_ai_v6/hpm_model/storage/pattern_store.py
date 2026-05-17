"""Shared cross-agent cross-corpus PatternStore backed by SQLite.

Stores patterns from all agents by name. Each pattern has its own embedding
dimension — no single-dim constraint. Running-average weights accumulate
across training calls. Name-based lookup enables scoring without dim matching.
"""
from __future__ import annotations

import os
import sqlite3
import struct
from typing import List, Optional, Tuple

import numpy as np

from hpm_ai_v6.hpm_model.core.cell import Cell


def _emb_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _bytes_to_emb(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


class PatternStore:
    """Shared, cross-agent, cross-corpus pattern store."""

    def __init__(self, cache_dir: str, similarity_threshold: float = 0.85) -> None:
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold

        shared_dir = os.path.join(cache_dir, "shared")
        os.makedirs(shared_dir, exist_ok=True)
        self._db_path = os.path.join(shared_dir, "patterns.db")
        self._npz_path = os.path.join(shared_dir, "patterns.npz")

        self._con = self._open_connection()
        self._init_schema()
        self._migrate_npz()

    def _open_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS shared_patterns (
                name      TEXT PRIMARY KEY NOT NULL,
                agent     TEXT,
                weight    REAL NOT NULL DEFAULT 1.0,
                count     INTEGER NOT NULL DEFAULT 1,
                dim       INTEGER NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        self._con.commit()

    def _migrate_npz(self) -> None:
        if not os.path.exists(self._npz_path):
            return
        count = self._con.execute("SELECT COUNT(*) FROM shared_patterns").fetchone()[0]
        if count > 0:
            return
        try:
            data = np.load(self._npz_path, allow_pickle=True)
            vectors = data["vectors"].astype(np.float32)
            weights_arr = data["weights"].astype(np.float32)
            counts_arr = data["counts"].astype(np.int32)
            names = data["names"].astype(object)
            agents = data["agents"].astype(object)
            for i in range(len(names)):
                vec = vectors[i]
                self._upsert(str(names[i]), str(agents[i]),
                             float(weights_arr[i]), int(counts_arr[i]), vec)
            self._con.commit()
        except Exception:
            pass

    def _upsert(self, name: str, agent: str, weight: float,
                count: int, vec: np.ndarray) -> None:
        emb_bytes = _emb_to_bytes(vec)
        dim = len(vec)
        existing = self._con.execute(
            "SELECT weight, count FROM shared_patterns WHERE name=?", (name,)
        ).fetchone()
        if existing:
            old_w, old_c = existing
            new_c = old_c + count
            new_w = (old_w * old_c + weight * count) / new_c
            self._con.execute(
                "UPDATE shared_patterns SET weight=?, count=?, agent=?, dim=?, embedding=? WHERE name=?",
                (new_w, new_c, agent, dim, emb_bytes, name),
            )
        else:
            self._con.execute(
                "INSERT INTO shared_patterns(name, agent, weight, count, dim, embedding) VALUES (?,?,?,?,?,?)",
                (name, agent, weight, count, dim, emb_bytes),
            )

    def merge(self, patterns: List[Cell], weights: List[float], agent_name: str) -> None:
        """Merge patterns with running-average weight update."""
        for cell, w in zip(patterns, weights):
            try:
                vec = cell.as_numpy().astype(np.float32)
                if vec.size == 0:
                    continue
                self._upsert(cell.name, agent_name, float(w), 1, vec)
            except Exception:
                continue
        self._con.commit()

    def load(self) -> Tuple[List[Cell], List[float]]:
        rows = self._con.execute(
            "SELECT name, weight, embedding FROM shared_patterns"
        ).fetchall()
        cells, weights = [], []
        for name, weight, emb_b in rows:
            emb = _bytes_to_emb(emb_b).astype(float)
            cells.append(Cell(name=str(name), embedding=emb))
            weights.append(float(weight))
        return cells, weights

    def find_similar(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Cell]]:
        """Return top-k most similar same-dim patterns by cosine similarity."""
        vec = np.asarray(vector, dtype=np.float32)
        dim = len(vec)
        rows = self._con.execute(
            "SELECT name, embedding FROM shared_patterns WHERE dim=?", (dim,)
        ).fetchall()
        results = []
        for name, emb_b in rows:
            emb = _bytes_to_emb(emb_b)
            sim = _cosine_sim(vec, emb)
            results.append((sim, Cell(name=str(name), embedding=emb.astype(float))))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def iter_payloads(self) -> List[dict]:
        rows = self._con.execute(
            "SELECT name, agent, weight, count, embedding FROM shared_patterns"
        ).fetchall()
        return [
            {"name": r[0], "agent": r[1], "weight": r[2],
             "count": r[3], "embedding": _bytes_to_emb(r[4]).tolist()}
            for r in rows
        ]

    def save(self) -> None:
        pass  # SQLite commits immediately; kept for API compatibility

    def clear(self) -> None:
        self._con.execute("DELETE FROM shared_patterns")
        self._con.commit()

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass
