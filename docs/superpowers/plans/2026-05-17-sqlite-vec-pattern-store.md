# sqlite-vec Pattern Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PatternPager's JSONL/JSON file storage with SQLite + sqlite-vec, preserving the exact public API so no callers change.

**Architecture:** Each agent gets a single `patterns.db` SQLite file (WAL mode) with a `patterns` metadata table and a `vec_patterns` virtual table for KNN search. The background writer thread is kept but now does `INSERT OR REPLACE` SQL instead of file writes. All existing callers (`base_agent.py`, `reasoning_agent.py`, `quiz_cli.py`) require zero changes.

**Tech Stack:** Python 3.10+, sqlite3 (stdlib), sqlite-vec>=0.1.6, numpy

---

## File Structure

- **Modify:** `hpm_ai_v6/hpm_model/storage/pattern_pager.py` — full internal rewrite, same public API
- **Modify:** `requirements.txt` — add `sqlite-vec>=0.1.6`
- **Create:** `hpm_ai_v6/hpm_model/storage/migrate_pager_to_sqlite.py` — one-time migration script
- **Create:** `hpm_ai_v6/tests/test_pattern_pager_sqlite.py` — tests for new backend

---

### Task 1: Add sqlite-vec dependency and verify install

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependency**

In `requirements.txt`, add after the existing entries:
```
sqlite-vec>=0.1.6
```

- [ ] **Step 2: Install**

```bash
pip install sqlite-vec
```

- [ ] **Step 3: Verify**

```bash
python3 -c "import sqlite_vec; import sqlite3; con = sqlite3.connect(':memory:'); con.enable_load_extension(True); sqlite_vec.load(con); con.enable_load_extension(False); print('sqlite-vec OK:', con.execute('SELECT vec_version()').fetchone())"
```

Expected output: `sqlite-vec OK: ('v0.1.x',)` (exact version may vary)

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add sqlite-vec for vector pattern storage"
```

---

### Task 2: Write failing tests for the new PatternPager

**Files:**
- Create: `hpm_ai_v6/tests/test_pattern_pager_sqlite.py`

- [ ] **Step 1: Create test file**

```python
"""Tests for PatternPager with sqlite-vec backend."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.storage.pattern_pager import PatternPager


def _make_cell(name: str, dim: int, vec: list[float], weight: float = 1.0) -> Cell:
    return Cell(name=name, dim=dim, embedding=np.array(vec, dtype=float), weight=weight)


def _make_pager(tmp_path: str, agent: str = "test_agent") -> PatternPager:
    return PatternPager(cache_dir=tmp_path, agent_name=agent)


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


def test_save_and_load(tmp):
    pager = _make_pager(tmp)
    cell = _make_cell("pat1", 1, [0.1, 0.2, 0.3])
    pager.save(cell)
    pager.flush()
    loaded = pager.load("pat1", {})
    assert loaded is not None
    assert loaded.name == "pat1"
    assert np.allclose(loaded.as_numpy(), [0.1, 0.2, 0.3], atol=1e-5)
    pager.close()


def test_load_unknown_returns_none(tmp):
    pager = _make_pager(tmp)
    assert pager.load("nonexistent", {}) is None
    pager.close()


def test_has(tmp):
    pager = _make_pager(tmp)
    cell = _make_cell("p1", 1, [1.0, 0.0])
    pager.save(cell)
    pager.flush()
    assert pager.has("p1") is True
    assert pager.has("p2") is False
    pager.close()


def test_duplicate_upsert(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("p1", 1, [1.0, 0.0], weight=1.0))
    pager.flush()
    pager.save(_make_cell("p1", 1, [1.0, 0.0], weight=9.9))
    pager.flush()
    loaded = pager.load("p1", {})
    assert abs(loaded.weight - 9.9) < 0.01
    payloads = pager.iter_index_payloads()
    names = [p["name"] for p in payloads]
    assert names.count("p1") == 1
    pager.close()


def test_iter_index_payloads(tmp):
    pager = _make_pager(tmp)
    for i in range(3):
        pager.save(_make_cell(f"p{i}", 1, [float(i), float(i + 1)]))
    pager.flush()
    payloads = pager.iter_index_payloads()
    assert len(payloads) == 3
    names = {p["name"] for p in payloads}
    assert names == {"p0", "p1", "p2"}
    assert all("embedding" in p for p in payloads)
    pager.close()


def test_load_nearest(tmp):
    pager = _make_pager(tmp)
    # Three unit vectors; query is closest to p1
    pager.save(_make_cell("p0", 1, [1.0, 0.0, 0.0]))
    pager.save(_make_cell("p1", 1, [0.0, 1.0, 0.0]))
    pager.save(_make_cell("p2", 1, [0.0, 0.0, 1.0]))
    pager.flush()
    query = np.array([0.01, 0.99, 0.01])
    result = pager.load_nearest(query, {}, min_similarity=0.5)
    assert result is not None
    assert result.name == "p1"
    pager.close()


def test_load_nearest_excludes(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("p0", 1, [1.0, 0.0, 0.0]))
    pager.save(_make_cell("p1", 1, [0.99, 0.1, 0.0]))
    pager.flush()
    query = np.array([1.0, 0.0, 0.0])
    result = pager.load_nearest(query, {}, min_similarity=0.5, exclude_names=["p0"])
    assert result is not None
    assert result.name == "p1"
    pager.close()


def test_load_nearest_below_threshold_returns_none(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("p0", 1, [1.0, 0.0, 0.0]))
    pager.flush()
    query = np.array([0.0, 0.0, 1.0])
    result = pager.load_nearest(query, {}, min_similarity=0.99)
    assert result is None
    pager.close()


def test_select_evictions_under_limit(tmp):
    pager = _make_pager(tmp, agent="evict_agent")
    pager.max_active_patterns = 10
    patterns = [_make_cell(f"p{i}", 1, [float(i)]) for i in range(5)]
    weights = [1.0] * 5
    assert pager.select_evictions(patterns, weights) == []
    pager.close()


def test_select_evictions_over_limit(tmp):
    pager = _make_pager(tmp, agent="evict_agent")
    pager.max_active_patterns = 3
    patterns = [_make_cell(f"p{i}", 1, [float(i)]) for i in range(5)]
    weights = [5.0, 1.0, 3.0, 0.5, 2.0]
    evictions = pager.select_evictions(patterns, weights)
    assert len(evictions) >= 2
    # Lowest-weight indices should be evicted first (weights 0.5 and 1.0 → indices 3 and 1)
    assert 3 in evictions
    assert 1 in evictions
    pager.close()


def test_persistence_across_open_close(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("persist1", 1, [0.5, 0.5]))
    pager.flush()
    pager.close()

    pager2 = _make_pager(tmp)
    loaded = pager2.load("persist1", {})
    assert loaded is not None
    assert loaded.name == "persist1"
    pager2.close()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
python3 -m pytest hpm_ai_v6/tests/test_pattern_pager_sqlite.py -v 2>&1 | head -40
```

Expected: most tests fail (PatternPager still uses JSONL backend)

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v6/tests/test_pattern_pager_sqlite.py
git commit -m "test: add failing tests for sqlite-vec PatternPager backend"
```

---

### Task 3: Rewrite PatternPager internals with sqlite-vec backend

**Files:**
- Modify: `hpm_ai_v6/hpm_model/storage/pattern_pager.py`

Keep the exact same class name, `__init__` signature, and all public methods. Replace all JSONL/JSON file logic with SQLite.

- [ ] **Step 1: Replace pattern_pager.py**

```python
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
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest hpm_ai_v6/tests/test_pattern_pager_sqlite.py -v
```

Expected: all 11 tests pass

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
python3 -m pytest hpm_ai_v6/tests/ -v --ignore=hpm_ai_v6/tests/test_dataset_training_agent_wikipedia.py -x 2>&1 | tail -20
```

Expected: no new failures

- [ ] **Step 4: Commit**

```bash
git add hpm_ai_v6/hpm_model/storage/pattern_pager.py
git commit -m "feat: rewrite PatternPager with sqlite-vec backend"
```

---

### Task 4: Write migration script

**Files:**
- Create: `hpm_ai_v6/hpm_model/storage/migrate_pager_to_sqlite.py`

- [ ] **Step 1: Create migration script**

```python
"""One-time migration: convert per-agent JSONL/JSON archives to SQLite + sqlite-vec."""
from __future__ import annotations

import json
import os
import sys


def migrate_agent_dir(agent_dir: str) -> int:
    """Migrate one agent directory. Returns count of patterns migrated."""
    from hpm_ai_v6.hpm_model.core.cell import Cell
    from hpm_ai_v6.hpm_model.storage.pattern_pager import PatternPager
    import numpy as np

    agent_name = os.path.basename(agent_dir)
    cache_dir = os.path.dirname(agent_dir)

    pager = PatternPager(cache_dir=cache_dir, agent_name=agent_name)

    payloads: dict[str, dict] = {}

    # Read index.jsonl
    index_path = os.path.join(agent_dir, "index.jsonl")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    payloads[str(p["name"])] = p
                except (json.JSONDecodeError, KeyError):
                    continue

    # Read individual JSON files (may fill gaps not in index)
    for fname in os.listdir(agent_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(agent_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                p = json.load(f)
            name = str(p.get("name", ""))
            if name and name not in payloads:
                payloads[name] = p
        except Exception:
            continue

    count = 0
    for payload in payloads.values():
        try:
            emb = payload.get("embedding")
            if emb is None:
                continue
            cell = Cell(
                name=str(payload["name"]),
                dim=int(payload.get("dim", 1)),
                embedding=np.asarray(emb, dtype=float),
                weight=float(payload.get("weight", 1.0)),
            )
            pager.save(cell)
            count += 1
        except Exception as e:
            print(f"  Skipping {payload.get('name')}: {e}")

    pager.flush()
    pager.close()
    return count


def main(cache_dir: str) -> None:
    if not os.path.isdir(cache_dir):
        print(f"Error: {cache_dir} is not a directory")
        sys.exit(1)

    for agent_name in os.listdir(cache_dir):
        agent_dir = os.path.join(cache_dir, agent_name)
        if not os.path.isdir(agent_dir):
            continue
        if agent_name.startswith("."):
            continue
        # Skip if already migrated (only patterns.db, no JSONL files)
        has_jsonl = os.path.exists(os.path.join(agent_dir, "index.jsonl"))
        has_json = any(f.endswith(".json") for f in os.listdir(agent_dir))
        if not has_jsonl and not has_json:
            print(f"  {agent_name}: already migrated, skipping")
            continue

        print(f"Migrating {agent_name}...")
        count = migrate_agent_dir(agent_dir)
        print(f"  {agent_name}: {count} patterns migrated")

    print("\nDone. Old JSON/JSONL files are preserved — delete them manually when satisfied.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 migrate_pager_to_sqlite.py <cache_dir>")
        sys.exit(1)
    main(sys.argv[1])
```

- [ ] **Step 2: Test migration with a temp directory**

```bash
python3 -c "
import tempfile, os, json, numpy as np
tmp = tempfile.mkdtemp()
agent_dir = os.path.join(tmp, 'testagent')
os.makedirs(agent_dir)
# Write a fake index.jsonl
with open(os.path.join(agent_dir, 'index.jsonl'), 'w') as f:
    f.write(json.dumps({'name': 'pat1', 'dim': 1, 'embedding': [0.1, 0.2], 'weight': 2.5}) + '\n')
print('Test dir:', tmp)

from hpm_ai_v6.hpm_model.storage.migrate_pager_to_sqlite import main
main(tmp)

from hpm_ai_v6.hpm_model.storage.pattern_pager import PatternPager
pager = PatternPager(tmp, 'testagent')
loaded = pager.load('pat1', {})
print('Loaded:', loaded.name, 'weight:', loaded.weight)
assert loaded.name == 'pat1'
assert abs(loaded.weight - 2.5) < 0.01
print('Migration test PASSED')
pager.close()
"
```

Expected: `Migration test PASSED`

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v6/hpm_model/storage/migrate_pager_to_sqlite.py
git commit -m "feat: add migration script from JSONL to sqlite-vec pattern store"
```

---

### Task 5: Run migration on live cache

> **Note:** This is a one-time operation on the live pattern cache. It is non-destructive — old files are preserved. Only run after Task 3 tests pass.

- [ ] **Step 1: Find the pattern cache directory**

```bash
python3 -c "
from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
import inspect, os
src = inspect.getfile(MultiAgentReader)
base = os.path.dirname(os.path.dirname(src))
cache = os.path.join(base, 'data', 'pattern_cache')
print('Cache dir:', cache)
print('Contents:', os.listdir(cache) if os.path.exists(cache) else 'not found')
"
```

- [ ] **Step 2: Run migration**

Replace `<cache_dir>` with the path printed in Step 1:

```bash
python3 hpm_ai_v6/hpm_model/storage/migrate_pager_to_sqlite.py <cache_dir>
```

Expected output: one line per agent showing pattern count migrated.

- [ ] **Step 3: Verify a patterns.db was created per agent**

```bash
find <cache_dir> -name "patterns.db" -exec ls -lh {} \;
```

Expected: one `patterns.db` per agent subdirectory.

- [ ] **Step 4: Spot-check via Python**

```bash
python3 -c "
from hpm_ai_v6.hpm_model.storage.pattern_pager import PatternPager
import sys
cache_dir = sys.argv[1]
agent = sys.argv[2]
pager = PatternPager(cache_dir, agent)
payloads = pager.iter_index_payloads()
print(f'{agent}: {len(payloads)} patterns in sqlite-vec store')
pager.close()
" <cache_dir> <first_agent_name>
```

Expected: count matches (or is close to) what the migration reported.

- [ ] **Step 5: Commit**

```bash
git commit --allow-empty -m "chore: live cache migrated to sqlite-vec pattern store"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Single `patterns.db` per agent — Task 3
- ✅ WAL mode + NORMAL sync — Task 3 (`_open_connection`)
- ✅ `vec_patterns` virtual table for KNN — Task 3 (`_ensure_vec_table`)
- ✅ `meta` table stores dim — Task 3 (`_init_schema`)
- ✅ Incompatible dim rejected silently — Task 3 (`_write_payload`)
- ✅ All public API methods preserved — Task 3
- ✅ Fallback to cosine scan if sqlite-vec unavailable — Task 3 (`load_nearest`)
- ✅ Migration script — Task 4
- ✅ 10 tests covering all API methods — Task 2
- ✅ `requirements.txt` updated — Task 1

**Placeholder scan:** None found.

**Type consistency:** All method signatures match across tasks.
