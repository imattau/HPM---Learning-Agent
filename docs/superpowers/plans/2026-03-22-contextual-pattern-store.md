# Contextual Pattern Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap `TieredStore` with `ContextualPatternStore` to archive and warm-start Tier 2 state across ARC episodes, accumulate globally-useful patterns in SQLite, and refactor archival logic into specialist `Librarian`/`Forecaster` classes.

**Architecture:** `ContextualPatternStore` is a transparent wrapper around `TieredStore` — all `TieredStore` public methods are delegated unchanged so `Agent.step()` is unaffected. The wrapper adds `begin_context`/`end_context` lifecycle hooks called by the benchmark harness. Phases 1–3 are validated sequentially: each phase is benchmarked before moving to the next.

**Tech Stack:** Python 3, `numpy`, `pickle` (stdlib, used for Tier 2 state as specified in the spec), `json` (stdlib), `uuid` (stdlib), `sqlite3` (stdlib), `pytest` for tests. Run tests with `python3 -m pytest tests/ -v`.

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `hpm/store/contextual_store.py` | Create | `SubstrateSignature`, `extract_signature`, `ContextualPatternStore` |
| `hpm/config.py` | Modify | Add `global_weight_threshold`, `global_promotion_n`, `fingerprint_nll_threshold` fields |
| `benchmarks/multi_agent_arc.py` | Modify | Replace bare `TieredStore` with `ContextualPatternStore` in `run_persistent` |
| `tests/store/test_contextual_store.py` | Create | Unit + integration tests for Phase 1 and Phase 2 |
| `hpm/agents/librarian.py` | Create | Phase 3: `Librarian` — coarse filter, global pass |
| `hpm/agents/forecaster.py` | Create | Phase 3: `Forecaster` — NLL ranking |
| `tests/agents/test_librarian.py` | Create | Phase 3: `Librarian` unit tests |
| `tests/agents/test_forecaster.py` | Create | Phase 3: `Forecaster` unit tests |

---

## Phase 1: Archive and Warm-Start

### Task 1: `SubstrateSignature` and `extract_signature`

**Files:**
- Create: `hpm/store/contextual_store.py`
- Create: `tests/store/test_contextual_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/store/test_contextual_store.py
import numpy as np
import pytest
from hpm.store.contextual_store import SubstrateSignature, extract_signature


def make_grid(rows, cols, colors):
    """Helper: grid with specified dimensions and colour values."""
    g = np.zeros((rows, cols), dtype=int)
    for i, c in enumerate(colors):
        g[i % rows, i % cols] = c
    return g


def test_grid_size():
    grid = np.zeros((5, 8), dtype=int)
    sig = extract_signature(grid)
    assert sig.grid_size == (5, 8)


def test_unique_color_count_excludes_background():
    # Background is 0. Colors 1, 2, 3 appear -> unique_color_count = 3
    grid = np.array([[0, 1, 2], [3, 0, 0]], dtype=int)
    sig = extract_signature(grid)
    assert sig.unique_color_count == 3


def test_unique_color_count_all_background():
    grid = np.zeros((4, 4), dtype=int)
    sig = extract_signature(grid)
    assert sig.unique_color_count == 0


def test_object_count_single_blob():
    grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=int)
    sig = extract_signature(grid)
    assert sig.object_count == 1


def test_object_count_two_separated_blobs():
    grid = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=int)
    sig = extract_signature(grid)
    assert sig.object_count == 2


def test_aspect_ratio_square():
    grid = np.zeros((5, 5), dtype=int)
    sig = extract_signature(grid)
    assert sig.aspect_ratio_bucket == "square"


def test_aspect_ratio_landscape():
    # rows/cols < 0.8 -> landscape
    grid = np.zeros((3, 10), dtype=int)
    sig = extract_signature(grid)
    assert sig.aspect_ratio_bucket == "landscape"


def test_aspect_ratio_portrait():
    # rows/cols > 1.25 -> portrait
    grid = np.zeros((10, 3), dtype=int)
    sig = extract_signature(grid)
    assert sig.aspect_ratio_bucket == "portrait"


def test_signature_is_hashable():
    grid = np.zeros((4, 4), dtype=int)
    sig = extract_signature(grid)
    assert hash(sig) is not None  # frozen dataclass
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

Expected: `ImportError` — `hpm.store.contextual_store` does not exist yet.

- [ ] **Step 3: Implement `SubstrateSignature` and `extract_signature`**

```python
# hpm/store/contextual_store.py
from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SubstrateSignature:
    grid_size: tuple[int, int]
    unique_color_count: int
    object_count: int
    aspect_ratio_bucket: str  # "square" | "landscape" | "portrait"


def extract_signature(grid: np.ndarray) -> SubstrateSignature:
    """Compute a structural fingerprint for an ARC task grid.

    Background value is 0. Objects are 4-connected non-zero components.
    aspect_ratio_bucket: square (0.8 <= r <= 1.25), landscape (< 0.8), portrait (> 1.25)
    """
    rows, cols = grid.shape
    unique_colors = set(int(v) for v in grid.flat if int(v) != 0)
    unique_color_count = len(unique_colors)

    # 4-connected component labelling for non-zero cells
    visited = np.zeros_like(grid, dtype=bool)
    object_count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                object_count += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc] or grid[cr, cc] == 0:
                        continue
                    visited[cr, cc] = True
                    stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])

    ratio = rows / cols if cols > 0 else 1.0
    if ratio < 0.8:
        bucket = "landscape"
    elif ratio > 1.25:
        bucket = "portrait"
    else:
        bucket = "square"

    return SubstrateSignature(
        grid_size=(rows, cols),
        unique_color_count=unique_color_count,
        object_count=object_count,
        aspect_ratio_bucket=bucket,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

Expected: All 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpm/store/contextual_store.py tests/store/test_contextual_store.py
git commit -m "feat: add SubstrateSignature and extract_signature"
```

---

### Task 2: `ContextualPatternStore` — archive write (`end_context`)

**Files:**
- Modify: `hpm/store/contextual_store.py`
- Modify: `tests/store/test_contextual_store.py`

- [ ] **Step 1: Write the failing tests for archive write**

Add to `tests/store/test_contextual_store.py`:

```python
import json
import tempfile
import pathlib
from hpm.store.contextual_store import ContextualPatternStore, SubstrateSignature
from hpm.store.tiered_store import TieredStore
from hpm.patterns.factory import make_pattern


def _make_store_with_pattern(agent_id="agent"):
    """Helper: TieredStore with one pattern in Tier 2."""
    tiered = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    # Save directly to Tier 2 (no active context)
    tiered.save(p, 0.9, agent_id)
    return tiered, p


def test_end_context_writes_pkl_and_index(tmp_path):
    tiered, p = _make_store_with_pattern()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")

    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=3,
                              object_count=2, aspect_ratio_bucket="square")
    context_id = store.begin_context(sig, first_obs=[])

    store.end_context(context_id, success_metrics={"correct": True})

    pkl_path = tmp_path / "run1" / f"{context_id}.pkl"
    assert pkl_path.exists(), "archive .pkl file must be written"

    index_path = tmp_path / "run1" / "index.json"
    assert index_path.exists(), "index.json must be written"

    index = json.loads(index_path.read_text())
    assert len(index) == 1
    entry = index[0]
    assert entry["context_id"] == context_id
    assert entry["signature"]["grid_size"] == [5, 5]
    assert "timestamp" in entry


def test_end_context_write_is_atomic(tmp_path):
    """No .tmp file should remain after end_context."""
    tiered = TieredStore()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(3, 3), unique_color_count=1,
                              object_count=1, aspect_ratio_bucket="square")
    context_id = store.begin_context(sig, first_obs=[])
    store.end_context(context_id, success_metrics={})
    tmp_files = list((tmp_path / "run1").glob("*.tmp.pkl"))
    assert tmp_files == [], "no .tmp.pkl files should remain"
```

- [ ] **Step 2: Run to verify failure**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_end_context_writes_pkl_and_index tests/store/test_contextual_store.py::test_end_context_write_is_atomic -v
```

Expected: `ImportError` for `ContextualPatternStore`.

- [ ] **Step 3: Implement `ContextualPatternStore` with `begin_context` / `end_context` (Phase 1 archive only)**

Append to `hpm/store/contextual_store.py` after `extract_signature`:

```python
class ContextualPatternStore:
    """Wraps TieredStore to archive Tier 2 across episodes and warm-start from past tasks.

    All TieredStore public methods are delegated unchanged (transparent to Agent.step()).
    Lifecycle hooks (begin_context / end_context) are called explicitly by the benchmark harness.

    Archive uses pickle for Tier 2 GaussianPattern objects (as specified — same objects
    that TieredStore already holds in memory during a run).
    """

    def __init__(self, tiered_store, archive_dir: str, run_id: Optional[str] = None,
                 fingerprint_nll_threshold: float = 50.0,
                 global_weight_threshold: float = 0.6,
                 global_promotion_n: int = 5):
        self._store = tiered_store
        self._archive_dir = archive_dir
        self._run_id = run_id or str(uuid.uuid4())
        self._fingerprint_nll_threshold = fingerprint_nll_threshold
        self._global_weight_threshold = global_weight_threshold
        self._global_promotion_n = global_promotion_n
        self._last_sig: Optional[SubstrateSignature] = None
        self._init_db()

    def _init_db(self) -> None:
        """Create global_patterns table in globals.db if not exists."""
        import sqlite3
        run_dir = os.path.join(self._archive_dir, self._run_id)
        os.makedirs(run_dir, exist_ok=True)
        self._db_path = os.path.join(run_dir, "globals.db")
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS global_patterns (
                id TEXT PRIMARY KEY,
                mu BLOB NOT NULL,
                weight REAL NOT NULL,
                agent_id TEXT NOT NULL,
                is_global INTEGER DEFAULT 0,
                context_ids TEXT DEFAULT '[]'
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Archive lifecycle hooks (called by benchmark harness)
    # ------------------------------------------------------------------

    def begin_context(self, sig: SubstrateSignature, first_obs: list) -> str:
        """Warm-start Tier 2 from best matching past episode; return context_id."""
        context_id = str(uuid.uuid4())
        self._last_sig = sig

        candidates = self._coarse_filter(sig)
        best = self._fine_filter(candidates, first_obs)
        if best is not None:
            self._load_archive(best["archive_path"])

        self._inject_globals()
        return context_id

    def end_context(self, context_id: str, success_metrics: dict) -> None:
        """Serialise Tier 2 to archive, update index, run global pass."""
        run_dir = os.path.join(self._archive_dir, self._run_id)
        os.makedirs(run_dir, exist_ok=True)

        archive_path = os.path.join(run_dir, f"{context_id}.pkl")
        tmp_path = archive_path + ".tmp.pkl"
        tier2_state = self._store.query_tier2_all()
        with open(tmp_path, "wb") as f:
            pickle.dump(tier2_state, f)
        os.replace(tmp_path, archive_path)

        index_path = os.path.join(run_dir, "index.json")
        index = self._load_index(index_path)
        import datetime
        sig = self._last_sig
        entry = {
            "context_id": context_id,
            "signature": {
                "grid_size": list(sig.grid_size) if sig else None,
                "unique_color_count": sig.unique_color_count if sig else None,
                "object_count": sig.object_count if sig else None,
                "aspect_ratio_bucket": sig.aspect_ratio_bucket if sig else None,
            },
            "success_metrics": success_metrics,
            "archive_path": archive_path,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        index.append(entry)
        with open(index_path, "w") as f:
            json.dump(index, f)

        self._run_global_pass(context_id)

    # ------------------------------------------------------------------
    # Internal helpers — Phase 1
    # ------------------------------------------------------------------

    def _coarse_filter(self, sig: SubstrateSignature) -> list:
        """Return index entries whose grid_size matches and color_count within +/-1."""
        candidates = []
        if not os.path.isdir(self._archive_dir):
            return candidates
        for run_dir_name in os.listdir(self._archive_dir):
            index_path = os.path.join(self._archive_dir, run_dir_name, "index.json")
            if not os.path.exists(index_path):
                continue
            for entry in self._load_index(index_path):
                s = entry.get("signature", {})
                stored_size = s.get("grid_size")
                if stored_size is None:
                    continue
                if tuple(stored_size) != sig.grid_size:
                    continue
                stored_cc = s.get("unique_color_count", -999)
                if abs(stored_cc - sig.unique_color_count) > 1:
                    continue
                candidates.append(entry)
        return candidates

    def _fine_filter(self, candidates: list, first_obs: list) -> Optional[dict]:
        """Return the lowest-NLL candidate below threshold, or None."""
        if not first_obs or not candidates:
            return None
        best_nll = self._fingerprint_nll_threshold
        best_candidate = None
        for entry in candidates:
            archive_path = entry.get("archive_path")
            if not archive_path or not os.path.exists(archive_path):
                continue
            try:
                with open(archive_path, "rb") as f:
                    records = pickle.load(f)
            except Exception:
                continue
            patterns = [p for p, _w, _aid in records]
            if not patterns:
                continue
            nll = self._mean_nll(patterns, first_obs)
            if nll < best_nll:
                best_nll = nll
                best_candidate = entry
        return best_candidate

    def _mean_nll(self, patterns, obs_list: list) -> float:
        nlls = []
        for obs in obs_list:
            for p in patterns:
                nlls.append(float(p.log_prob(obs)))
        return float(np.mean(nlls)) if nlls else float("inf")

    def _load_archive(self, archive_path: str) -> None:
        """Deserialise archived Tier 2 records into the wrapped TieredStore's Tier 2."""
        with open(archive_path, "rb") as f:
            records = pickle.load(f)
        for pattern, weight, agent_id in records:
            self._store._tier2.save(pattern, weight, agent_id)

    def _load_index(self, index_path: str) -> list:
        if not os.path.exists(index_path):
            return []
        with open(index_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Internal helpers — Phase 2 (implemented in Task 8 / Task 9)
    # ------------------------------------------------------------------

    def _inject_globals(self) -> None:
        """Load is_global=True patterns from globals.db into Tier 2. No-op until Phase 2."""
        pass

    def _run_global_pass(self, context_id: str) -> None:
        """Promote high-weight Tier 2 patterns to globals.db. No-op until Phase 2."""
        pass

    # ------------------------------------------------------------------
    # TieredStore delegation (transparent to Agent.step())
    # ------------------------------------------------------------------

    def save(self, pattern, weight: float, agent_id: str) -> None:
        self._store.save(pattern, weight, agent_id)

    def load(self, pattern_id: str) -> tuple:
        return self._store.load(pattern_id)

    def query(self, agent_id: str) -> list:
        return self._store.query(agent_id)

    def query_all(self) -> list:
        return self._store.query_all()

    def query_tier2(self, agent_id: str) -> list:
        return self._store.query_tier2(agent_id)

    def query_tier2_all(self) -> list:
        return self._store.query_tier2_all()

    def delete(self, pattern_id: str) -> None:
        self._store.delete(pattern_id)

    def update_weight(self, pattern_id: str, weight: float) -> None:
        self._store.update_weight(pattern_id, weight)

    def similarity_merge(self, context_id: str, **kwargs) -> None:
        self._store.similarity_merge(context_id, **kwargs)

    def promote_to_tier2(self, pattern, weight: float, agent_id: str, **kwargs) -> None:
        self._store.promote_to_tier2(pattern, weight, agent_id, **kwargs)

    def query_negative(self, agent_id: str) -> list:
        return self._store.query_negative(agent_id)

    def query_tier2_negative_all(self) -> list:
        return self._store.query_tier2_negative_all()

    def negative_merge(self, context_id: str, **kwargs) -> None:
        self._store.negative_merge(context_id, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpm/store/contextual_store.py tests/store/test_contextual_store.py
git commit -m "feat: ContextualPatternStore archive write, coarse/fine filter, delegation"
```

---

### Task 3: Warm-start round-trip and integration tests

**Files:**
- Modify: `tests/store/test_contextual_store.py`

- [ ] **Step 1: Write the warm-start round-trip and integration tests**

Add to `tests/store/test_contextual_store.py`:

```python
def test_round_trip_warm_start(tmp_path):
    """end_context archives Tier 2; begin_context on matching sig restores patterns."""
    agent_id = "agent"
    tiered1 = TieredStore()
    p = make_pattern(mu=np.array([1.0, 2.0, 3.0, 4.0]),
                     scale=np.eye(4), pattern_type="gaussian")
    tiered1.save(p, 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig, first_obs=[])
    store1.end_context(ctx1, success_metrics={"correct": True})

    # New store, same run_id so index is found
    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    store2.begin_context(sig, first_obs=[])

    tier2_records = tiered2.query_tier2_all()
    pattern_ids = [rec[0].id for rec in tier2_records]
    assert p.id in pattern_ids, "warm-started Tier 2 must contain archived pattern"


def test_coarse_filter_excludes_mismatched_grid_size(tmp_path):
    """Archive from (5,5) grid must not warm-start a (3,3) episode."""
    agent_id = "agent"
    tiered1 = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tiered1.save(p, 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig_55 = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig_55, first_obs=[])
    store1.end_context(ctx1, success_metrics={})

    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    sig_33 = SubstrateSignature(grid_size=(3, 3), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    store2.begin_context(sig_33, first_obs=[])
    assert tiered2.query_tier2_all() == [], "mismatched grid_size must not warm-start"


def test_coarse_filter_excludes_color_count_outside_range(tmp_path):
    """Archive with color_count=5 must not match sig with color_count=2 (diff=3 > 1)."""
    agent_id = "agent"
    tiered1 = TieredStore()
    tiered1.save(make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian"),
                 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig_c5 = SubstrateSignature(grid_size=(5, 5), unique_color_count=5,
                                 object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig_c5, first_obs=[])
    store1.end_context(ctx1, success_metrics={})

    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    sig_c2 = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    store2.begin_context(sig_c2, first_obs=[])
    assert tiered2.query_tier2_all() == [], "color_count diff > 1 must not warm-start"


def test_fine_filter_rejects_above_threshold(tmp_path):
    """Candidate whose mean NLL > fingerprint_nll_threshold is rejected."""
    agent_id = "agent"
    tiered1 = TieredStore()
    # Pattern far from zero-obs: NLL will be very high
    p = make_pattern(mu=np.array([100.0, 100.0, 100.0, 100.0]),
                     scale=np.eye(4) * 0.001, pattern_type="gaussian")
    tiered1.save(p, 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig, first_obs=[])
    store1.end_context(ctx1, success_metrics={})

    tiered2 = TieredStore()
    # Very low threshold forces rejection
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1",
                                     fingerprint_nll_threshold=0.001)
    obs = [np.zeros(4)]
    store2.begin_context(sig, first_obs=obs)
    assert tiered2.query_tier2_all() == [], "high-NLL candidate must be rejected"


def test_integration_agent_warm_start(tmp_path):
    """Full integration: agent learns task A, archives, task B warm-starts from task A."""
    from hpm.agents.agent import Agent
    from hpm.config import AgentConfig

    config = AgentConfig(agent_id="agent", feature_dim=4, gamma_soc=0.0)
    tiered1 = TieredStore()
    agent1 = Agent(config, store=tiered1)
    obs = np.random.default_rng(42).normal(size=4)
    for _ in range(5):
        agent1.step(obs)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig, first_obs=[])
    # Promote a pattern to Tier 2 so archive is non-empty
    records = tiered1.query("agent")
    if records:
        p, w = records[0]
        tiered1.promote_to_tier2(p, w, "agent")
    store1.end_context(ctx1, success_metrics={"correct": True})

    # Task B: new agent and store, same signature
    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    store2.begin_context(sig, first_obs=[])

    assert len(tiered2.query_tier2_all()) > 0, \
        "after warm-start, Tier 2 must contain patterns from task A"
```

- [ ] **Step 2: Run to verify which tests fail**

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

Expected: The new tests may fail if `_last_sig` is not set before `begin_context` finds no archive (first run). Fix: ensure `self._last_sig = sig` is the first statement in `begin_context` (already done in Task 2 implementation). If tests still fail, check that `_coarse_filter` correctly iterates `self._archive_dir` subdirectories.

- [ ] **Step 3: Fix any issues found**

If `test_round_trip_warm_start` fails: the most likely cause is that `_fine_filter` returns `None` because `first_obs=[]`. Since `first_obs` is empty, `_fine_filter` exits early. The warm-start should still fire if there are coarse-filter candidates. Fix: when `first_obs` is empty, skip NLL filtering and return the first coarse-filter candidate directly (the spec says fine filter uses `first_obs`; when none are provided, skip the fine filter step):

```python
    def _fine_filter(self, candidates: list, first_obs: list) -> Optional[dict]:
        if not candidates:
            return None
        # No observations to filter on: take first coarse candidate
        if not first_obs:
            # Only warm-start if there are candidates; take first
            return candidates[0] if candidates else None
        best_nll = self._fingerprint_nll_threshold
        best_candidate = None
        for entry in candidates:
            ...
```

Re-run after fix:

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

- [ ] **Step 4: Run full project tests to verify no regressions**

```bash
python3 -m pytest tests/ -v
```

Expected: All existing tests continue to pass.

- [ ] **Step 5: Commit**

```bash
git add hpm/store/contextual_store.py tests/store/test_contextual_store.py
git commit -m "feat: Phase 1 warm-start round-trip and integration tests passing"
```

---

### Task 4: Benchmark integration — replace bare `TieredStore` with `ContextualPatternStore`

**Files:**
- Modify: `benchmarks/multi_agent_arc.py`

- [ ] **Step 1: Update imports in `run_persistent`**

In `benchmarks/multi_agent_arc.py`, inside `run_persistent`, add these imports alongside the existing ones:

```python
    from hpm.store.contextual_store import ContextualPatternStore, extract_signature
```

- [ ] **Step 2: Wrap `TieredStore` with `ContextualPatternStore`**

After `tiered = TieredStore()`, add:

```python
    contextual = ContextualPatternStore(tiered, archive_dir="data/archives")
```

- [ ] **Step 3: Update the episode loop**

Replace the start of the episode loop body. Before `tiered.begin_context(context_id)`, collect the signature and first obs:

```python
    for run_idx, (task_idx, task) in enumerate(eligible):
        # Compute structural signature from the first training input grid
        first_input_grid = np.array(task["train"][0]["input"], dtype=int)
        sig = extract_signature(first_input_grid)
        # Collect up to 3 encoded obs for the fine NLL filter
        first_obs_vecs = [
            np.array(task["train"][i]["input"], dtype=int).flatten()[:FEATURE_DIM].astype(float) / 9.0
            for i in range(min(3, len(task["train"])))
        ]

        context_id = contextual.begin_context(sig, first_obs=first_obs_vecs)
        tiered.begin_context(context_id)
```

After `tiered.end_context(context_id, correct=correct)`, add:

```python
        contextual.end_context(context_id, success_metrics={"correct": correct, "rank": rank})
```

- [ ] **Step 4: Verify imports cleanly**

```bash
python3 -c "from benchmarks.multi_agent_arc import run_persistent; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run a 5-task smoke test**

```bash
python3 -c "
import benchmarks.multi_agent_arc as b
tasks = b.load_tasks()
eligible = [(i, t) for i, t in enumerate(tasks) if b.task_fits(t)][:5]
c, r = b.run_persistent(eligible, tasks)
print(f'smoke test: correct={c}, rank_sum={r}')
"
```

Expected: Completes without error, prints results.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/multi_agent_arc.py
git commit -m "feat: integrate ContextualPatternStore into ARC persistent benchmark"
```

---

### Task 5: Phase 1 benchmark baseline

**Files:** No code changes.

- [ ] **Step 1: Run Phase 1 benchmark (first 100 tasks) and record baseline**

```bash
python3 -c "
import benchmarks.multi_agent_arc as b
tasks = b.load_tasks()
eligible = [(i, t) for i, t in enumerate(tasks) if b.task_fits(t)][:100]
c, r = b.run_persistent(eligible, tasks)
n = len(eligible)
print(f'Phase 1 baseline: accuracy={c/n*100:.1f}%, mean_rank={r/n:.2f}')
"
```

Expected: Benchmark completes. Record accuracy and mean rank as the Phase 1 baseline before proceeding to Phase 2.

- [ ] **Step 2: Commit phase marker**

```bash
git commit --allow-empty -m "chore: record Phase 1 baseline before Phase 2"
```

---

## Phase 2: Global Pass and `is_global`

### Task 6: `AgentConfig` fields for global promotion

**Files:**
- Modify: `hpm/config.py`
- Modify: `tests/store/test_contextual_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_contextual_store.py`:

```python
def test_agent_config_global_fields():
    from hpm.config import AgentConfig
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert cfg.global_weight_threshold == 0.6
    assert cfg.global_promotion_n == 5
    assert cfg.fingerprint_nll_threshold == 50.0
```

- [ ] **Step 2: Run to verify failure**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_agent_config_global_fields -v
```

Expected: `AttributeError`.

- [ ] **Step 3: Add fields to `AgentConfig` in `hpm/config.py`**

After the last existing field (`pattern_type`), add:

```python
    # ContextualPatternStore global promotion (Phase 2)
    global_weight_threshold: float = 0.6    # min Tier 2 weight to candidate for global promotion
    global_promotion_n: int = 5             # appearances in context_ids before is_global=True
    fingerprint_nll_threshold: float = 50.0 # max NLL for fine-filter candidate acceptance
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_agent_config_global_fields -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add hpm/config.py tests/store/test_contextual_store.py
git commit -m "feat: add global_weight_threshold, global_promotion_n, fingerprint_nll_threshold to AgentConfig"
```

---

### Task 7: SQLite schema for `global_patterns`

**Files:**
- Modify: `tests/store/test_contextual_store.py`

The `_init_db` method was already written in Task 2. This task verifies it with a dedicated test.

- [ ] **Step 1: Write the failing test for schema migration**

Add to `tests/store/test_contextual_store.py`:

```python
import sqlite3

def test_schema_migration_creates_globals_table(tmp_path):
    tiered = TieredStore()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")
    db_path = tmp_path / "run1" / "globals.db"
    assert db_path.exists(), "globals.db must be created at __init__"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='global_patterns'"
    )
    assert cursor.fetchone() is not None, "global_patterns table must exist"
    cursor = conn.execute("PRAGMA table_info(global_patterns)")
    cols = {row[1] for row in cursor.fetchall()}
    assert {"id", "mu", "weight", "agent_id", "is_global", "context_ids"}.issubset(cols)
    conn.close()
```

- [ ] **Step 2: Run to verify it passes (already implemented in Task 2)**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_schema_migration_creates_globals_table -v
```

Expected: PASS (schema was created in Task 2's `_init_db`).

- [ ] **Step 3: Commit**

```bash
git add tests/store/test_contextual_store.py
git commit -m "test: verify globals.db schema creation"
```

---

### Task 8: Global Pass — upsert Tier 2 patterns into SQLite

**Files:**
- Modify: `hpm/store/contextual_store.py`
- Modify: `tests/store/test_contextual_store.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/store/test_contextual_store.py`:

```python
def test_global_pass_upserts_high_weight_patterns(tmp_path):
    """Patterns with weight > global_weight_threshold are upserted to globals.db."""
    agent_id = "agent"
    tiered = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tiered.save(p, 0.8, agent_id)  # weight 0.8 > default threshold 0.6

    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1",
                                    global_weight_threshold=0.6)
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx = store.begin_context(sig, first_obs=[])
    store.end_context(ctx, success_metrics={})

    conn = sqlite3.connect(str(tmp_path / "run1" / "globals.db"))
    row = conn.execute("SELECT id, context_ids FROM global_patterns WHERE id=?",
                       (p.id,)).fetchone()
    conn.close()
    assert row is not None, "high-weight pattern must be upserted to globals.db"
    context_ids = json.loads(row[1])
    assert ctx in context_ids


def test_global_pass_does_not_upsert_low_weight(tmp_path):
    """Patterns with weight <= global_weight_threshold are not written to globals.db."""
    agent_id = "agent"
    tiered = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tiered.save(p, 0.3, agent_id)  # weight 0.3 < threshold 0.6

    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1",
                                    global_weight_threshold=0.6)
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx = store.begin_context(sig, first_obs=[])
    store.end_context(ctx, success_metrics={})

    conn = sqlite3.connect(str(tmp_path / "run1" / "globals.db"))
    row = conn.execute("SELECT id FROM global_patterns WHERE id=?", (p.id,)).fetchone()
    conn.close()
    assert row is None, "low-weight pattern must not be upserted"


def test_global_promotion_after_n_appearances(tmp_path):
    """Pattern set to is_global=1 after appearing in >= global_promotion_n episodes."""
    agent_id = "agent"
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")

    # Share the same db across 3 separate ContextualPatternStore instances
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    shared_db = str(run_dir / "globals.db")

    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")

    for _ in range(3):
        tiered = TieredStore()
        tiered.save(p, 0.9, agent_id)
        local_store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1",
                                              global_weight_threshold=0.6, global_promotion_n=3)
        local_store._db_path = shared_db  # point all instances at the same db
        ctx = local_store.begin_context(sig, first_obs=[])
        local_store.end_context(ctx, success_metrics={})

    conn = sqlite3.connect(shared_db)
    row = conn.execute("SELECT is_global, context_ids FROM global_patterns WHERE id=?",
                       (p.id,)).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 1, "is_global must be 1 after N appearances"
    assert len(json.loads(row[1])) >= 3
```

- [ ] **Step 2: Run to verify failures**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_global_pass_upserts_high_weight_patterns tests/store/test_contextual_store.py::test_global_pass_does_not_upsert_low_weight tests/store/test_contextual_store.py::test_global_promotion_after_n_appearances -v
```

Expected: All 3 fail (`_run_global_pass` is still a no-op).

- [ ] **Step 3: Implement `_run_global_pass`**

Replace the no-op `_run_global_pass` in `hpm/store/contextual_store.py`:

```python
    def _run_global_pass(self, context_id: str) -> None:
        """Upsert high-weight Tier 2 patterns to globals.db; set is_global after N appearances."""
        import sqlite3 as _sqlite3
        tier2 = self._store.query_tier2_all()
        conn = _sqlite3.connect(self._db_path)
        try:
            for pattern, weight, agent_id in tier2:
                if weight <= self._global_weight_threshold:
                    continue
                mu_blob = pickle.dumps(pattern.mu)
                existing = conn.execute(
                    "SELECT context_ids FROM global_patterns WHERE id=?",
                    (pattern.id,)
                ).fetchone()
                if existing is None:
                    is_global = 1 if 1 >= self._global_promotion_n else 0
                    conn.execute(
                        "INSERT INTO global_patterns "
                        "(id, mu, weight, agent_id, is_global, context_ids) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (pattern.id, mu_blob, weight, agent_id,
                         is_global, json.dumps([context_id]))
                    )
                else:
                    ctx_ids = json.loads(existing[0])
                    if context_id not in ctx_ids:
                        ctx_ids.append(context_id)
                    is_global = 1 if len(ctx_ids) >= self._global_promotion_n else 0
                    conn.execute(
                        "UPDATE global_patterns "
                        "SET mu=?, weight=?, is_global=?, context_ids=? WHERE id=?",
                        (mu_blob, weight, is_global, json.dumps(ctx_ids), pattern.id)
                    )
            conn.commit()
        finally:
            conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: All existing tests continue to pass.

- [ ] **Step 6: Commit**

```bash
git add hpm/store/contextual_store.py tests/store/test_contextual_store.py
git commit -m "feat: Phase 2 global pass — upsert Tier 2 patterns and set is_global after N appearances"
```

---

### Task 9: Inject global patterns in `begin_context`

**Files:**
- Modify: `hpm/store/contextual_store.py`
- Modify: `tests/store/test_contextual_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_contextual_store.py`:

```python
def test_begin_context_injects_global_patterns(tmp_path):
    """is_global=1 patterns from globals.db are injected into Tier 2 at begin_context."""
    from hpm.patterns.factory import make_pattern, pattern_from_dict

    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "globals.db"

    p = make_pattern(mu=np.array([1.0, 2.0, 3.0, 4.0]), scale=np.eye(4),
                     pattern_type="gaussian")
    mu_blob = pickle.dumps(p.mu)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS global_patterns (
            id TEXT PRIMARY KEY, mu BLOB NOT NULL, weight REAL NOT NULL,
            agent_id TEXT NOT NULL, is_global INTEGER DEFAULT 0,
            context_ids TEXT DEFAULT '[]'
        )
    """)
    conn.execute(
        "INSERT INTO global_patterns (id, mu, weight, agent_id, is_global, context_ids) "
        "VALUES (?,?,?,?,1,?)",
        (p.id, mu_blob, 0.9, "agent", json.dumps(["ctx_prev"]))
    )
    conn.commit()
    conn.close()

    tiered = TieredStore()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    store.begin_context(sig, first_obs=[])

    tier2_ids = [rec[0].id for rec in tiered.query_tier2_all()]
    assert p.id in tier2_ids, "is_global pattern must be injected into Tier 2"
```

- [ ] **Step 2: Run to verify failure**

```bash
python3 -m pytest tests/store/test_contextual_store.py::test_begin_context_injects_global_patterns -v
```

Expected: FAIL (`_inject_globals` is still a no-op).

- [ ] **Step 3: Implement `_inject_globals`**

Replace the no-op `_inject_globals` in `hpm/store/contextual_store.py`:

```python
    def _inject_globals(self) -> None:
        """Load all is_global=True patterns from globals.db into Tier 2."""
        import sqlite3 as _sqlite3
        from hpm.patterns.factory import make_pattern, pattern_from_dict
        if not hasattr(self, '_db_path') or not os.path.exists(self._db_path):
            return
        conn = _sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT id, mu, weight, agent_id FROM global_patterns WHERE is_global=1"
            ).fetchall()
        finally:
            conn.close()
        for pid, mu_blob, weight, agent_id in rows:
            mu = pickle.loads(mu_blob)
            # Skip if already in Tier 2
            if self._store._tier2.has(pid):
                continue
            p = make_pattern(mu=mu, scale=np.eye(len(mu)), pattern_type="gaussian")
            p_dict = p.to_dict()
            p_dict['id'] = pid
            p_restored = pattern_from_dict(p_dict)
            self._store._tier2.save(p_restored, weight, agent_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/store/test_contextual_store.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: All existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add hpm/store/contextual_store.py tests/store/test_contextual_store.py
git commit -m "feat: inject is_global patterns from SQLite into Tier 2 at begin_context"
```

---

### Task 10: Phase 2 benchmark validation

**Files:** No code changes.

- [ ] **Step 1: Run Phase 2 benchmark (first 100 tasks)**

```bash
python3 -c "
import benchmarks.multi_agent_arc as b
tasks = b.load_tasks()
eligible = [(i, t) for i, t in enumerate(tasks) if b.task_fits(t)][:100]
c, r = b.run_persistent(eligible, tasks)
n = len(eligible)
print(f'Phase 2 results: accuracy={c/n*100:.1f}%, mean_rank={r/n:.2f}')
"
```

Expected: After episode 5, global patterns are being injected (add a debug print to `_inject_globals` to confirm `global_patterns_loaded > 0`). Accuracy on episodes 6+ should be stable or improving vs. Phase 1 baseline.

- [ ] **Step 2: Commit phase marker**

```bash
git commit --allow-empty -m "chore: Phase 2 benchmarked — proceed to Phase 3"
```

---

## Phase 3: Specialist Roles (Pure Refactor)

### Task 11: `CandidateArchive` dataclass and `Librarian`

**Files:**
- Create: `hpm/agents/librarian.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_librarian.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/test_librarian.py
import json
import os
import pickle
import numpy as np
import pytest
from hpm.agents.librarian import Librarian, CandidateArchive
from hpm.store.contextual_store import SubstrateSignature
from hpm.patterns.factory import make_pattern


def _write_index_and_archive(run_dir, context_id, sig, patterns):
    """Helper: write a fake index.json and .pkl for a run directory."""
    os.makedirs(run_dir, exist_ok=True)
    archive_path = os.path.join(run_dir, f"{context_id}.pkl")
    records = [(p, 0.9, "agent") for p in patterns]
    with open(archive_path, "wb") as f:
        pickle.dump(records, f)
    index_path = os.path.join(run_dir, "index.json")
    entry = {
        "context_id": context_id,
        "signature": {
            "grid_size": list(sig.grid_size),
            "unique_color_count": sig.unique_color_count,
            "object_count": sig.object_count,
            "aspect_ratio_bucket": sig.aspect_ratio_bucket,
        },
        "archive_path": archive_path,
        "success_metrics": {},
        "timestamp": "2026-03-22T00:00:00",
    }
    with open(index_path, "w") as f:
        json.dump([entry], f)
    return archive_path


def test_query_archive_returns_matching_candidate(tmp_path):
    lib = Librarian()
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    run_dir = str(tmp_path / "run1")
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    _write_index_and_archive(run_dir, "ctx1", sig, [p])

    candidates = lib.query_archive(sig, archive_dir=str(tmp_path))
    assert len(candidates) == 1
    assert candidates[0].context_id == "ctx1"
    assert isinstance(candidates[0], CandidateArchive)


def test_query_archive_excludes_mismatched_grid(tmp_path):
    lib = Librarian()
    sig_55 = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    sig_33 = SubstrateSignature(grid_size=(3, 3), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    run_dir = str(tmp_path / "run1")
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    _write_index_and_archive(run_dir, "ctx1", sig_55, [p])

    candidates = lib.query_archive(sig_33, archive_dir=str(tmp_path))
    assert candidates == []


def test_query_archive_excludes_color_count_diff_gt_1(tmp_path):
    lib = Librarian()
    sig_c5 = SubstrateSignature(grid_size=(5, 5), unique_color_count=5,
                                 object_count=1, aspect_ratio_bucket="square")
    sig_c2 = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    run_dir = str(tmp_path / "run1")
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    _write_index_and_archive(run_dir, "ctx1", sig_c5, [p])

    candidates = lib.query_archive(sig_c2, archive_dir=str(tmp_path))
    assert candidates == []
```

- [ ] **Step 2: Run to verify failure**

```bash
python3 -m pytest tests/agents/test_librarian.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Create `tests/agents/__init__.py`**

```bash
touch /path/to/repo/tests/agents/__init__.py
```

Actual path: `tests/agents/__init__.py` (empty file).

- [ ] **Step 4: Implement `CandidateArchive` and `Librarian`**

```python
# hpm/agents/librarian.py
from __future__ import annotations

import json
import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
from hpm.store.contextual_store import SubstrateSignature


@dataclass
class CandidateArchive:
    context_id: str
    signature: SubstrateSignature
    archive_path: str
    success_metrics: dict


class Librarian:
    """Stateless. Owns coarse archive filtering and global pass logic."""

    def query_archive(self, sig: SubstrateSignature,
                      archive_dir: str) -> list[CandidateArchive]:
        """Coarse filter: return candidates with matching grid_size and color_count within +-1."""
        candidates = []
        if not os.path.isdir(archive_dir):
            return candidates
        for run_dir_name in os.listdir(archive_dir):
            index_path = os.path.join(archive_dir, run_dir_name, "index.json")
            if not os.path.exists(index_path):
                continue
            with open(index_path) as f:
                entries = json.load(f)
            for entry in entries:
                s = entry.get("signature", {})
                stored_size = s.get("grid_size")
                if stored_size is None:
                    continue
                if tuple(stored_size) != sig.grid_size:
                    continue
                stored_cc = s.get("unique_color_count", -999)
                if abs(stored_cc - sig.unique_color_count) > 1:
                    continue
                archive_path = entry.get("archive_path", "")
                if not os.path.exists(archive_path):
                    continue
                candidate_sig = SubstrateSignature(
                    grid_size=tuple(stored_size),
                    unique_color_count=s.get("unique_color_count", 0),
                    object_count=s.get("object_count", 0),
                    aspect_ratio_bucket=s.get("aspect_ratio_bucket", "square"),
                )
                candidates.append(CandidateArchive(
                    context_id=entry["context_id"],
                    signature=candidate_sig,
                    archive_path=archive_path,
                    success_metrics=entry.get("success_metrics", {}),
                ))
        return candidates

    def run_global_pass(self, tier2_patterns: list, context_id: str,
                        db_path: str, global_weight_threshold: float,
                        global_promotion_n: int) -> None:
        """Upsert high-weight Tier 2 patterns; promote to is_global after N appearances."""
        conn = sqlite3.connect(db_path)
        try:
            for pattern, weight, agent_id in tier2_patterns:
                if weight <= global_weight_threshold:
                    continue
                mu_blob = pickle.dumps(pattern.mu)
                existing = conn.execute(
                    "SELECT context_ids FROM global_patterns WHERE id=?",
                    (pattern.id,)
                ).fetchone()
                if existing is None:
                    is_global = 1 if 1 >= global_promotion_n else 0
                    conn.execute(
                        "INSERT INTO global_patterns "
                        "(id, mu, weight, agent_id, is_global, context_ids) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (pattern.id, mu_blob, weight, agent_id,
                         is_global, json.dumps([context_id]))
                    )
                else:
                    ctx_ids = json.loads(existing[0])
                    if context_id not in ctx_ids:
                        ctx_ids.append(context_id)
                    is_global = 1 if len(ctx_ids) >= global_promotion_n else 0
                    conn.execute(
                        "UPDATE global_patterns "
                        "SET mu=?, weight=?, is_global=?, context_ids=? WHERE id=?",
                        (mu_blob, weight, is_global, json.dumps(ctx_ids), pattern.id)
                    )
            conn.commit()
        finally:
            conn.close()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/agents/test_librarian.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add hpm/agents/librarian.py tests/agents/__init__.py tests/agents/test_librarian.py
git commit -m "feat: Phase 3 Librarian with CandidateArchive and coarse filter"
```

---

### Task 12: `RankedCandidate` and `Forecaster`

**Files:**
- Create: `hpm/agents/forecaster.py`
- Create: `tests/agents/test_forecaster.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/test_forecaster.py
import os
import pickle
import numpy as np
import pytest
from hpm.agents.forecaster import Forecaster, RankedCandidate
from hpm.agents.librarian import CandidateArchive
from hpm.store.contextual_store import SubstrateSignature
from hpm.patterns.factory import make_pattern


def _make_candidate(tmp_path, context_id, mu, weight=0.9):
    p = make_pattern(mu=np.array(mu, dtype=float), scale=np.eye(len(mu)),
                     pattern_type="gaussian")
    archive_path = str(tmp_path / f"{context_id}.pkl")
    with open(archive_path, "wb") as f:
        pickle.dump([(p, weight, "agent")], f)
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    return CandidateArchive(context_id=context_id, signature=sig,
                             archive_path=archive_path, success_metrics={})


def test_rank_returns_sorted_ascending_by_nll(tmp_path):
    fc = Forecaster(fingerprint_nll_threshold=1e9)
    # Candidate A: pattern near obs zeros -> low NLL
    cand_a = _make_candidate(tmp_path, "a", mu=[0.0, 0.0, 0.0, 0.0])
    # Candidate B: pattern far from obs -> high NLL
    cand_b = _make_candidate(tmp_path, "b", mu=[100.0, 100.0, 100.0, 100.0])
    obs = [np.zeros(4)]
    ranked = fc.rank([cand_a, cand_b], obs)
    assert len(ranked) == 2
    assert ranked[0].candidate.context_id == "a"
    assert ranked[0].nll < ranked[1].nll


def test_rank_excludes_above_threshold(tmp_path):
    fc = Forecaster(fingerprint_nll_threshold=0.001)
    cand = _make_candidate(tmp_path, "a", mu=[100.0, 100.0, 100.0, 100.0])
    obs = [np.zeros(4)]
    ranked = fc.rank([cand], obs)
    assert ranked == [], "candidates above NLL threshold must be excluded"


def test_rank_empty_candidates(tmp_path):
    fc = Forecaster(fingerprint_nll_threshold=50.0)
    ranked = fc.rank([], obs=[np.zeros(4)])
    assert ranked == []


def test_rank_no_obs(tmp_path):
    fc = Forecaster(fingerprint_nll_threshold=50.0)
    cand = _make_candidate(tmp_path, "a", mu=[0.0, 0.0, 0.0, 0.0])
    ranked = fc.rank([cand], obs=[])
    assert ranked == [], "no observations: cannot rank, return empty"
```

- [ ] **Step 2: Run to verify failure**

```bash
python3 -m pytest tests/agents/test_forecaster.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `RankedCandidate` and `Forecaster`**

```python
# hpm/agents/forecaster.py
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from hpm.agents.librarian import CandidateArchive


@dataclass
class RankedCandidate:
    candidate: CandidateArchive
    nll: float


class Forecaster:
    """Stateless. Ranks CandidateArchive entries by mean Pattern Fingerprint NLL."""

    def __init__(self, fingerprint_nll_threshold: float = 50.0):
        self._threshold = fingerprint_nll_threshold

    def rank(self, candidates: list[CandidateArchive],
             obs: list[np.ndarray]) -> list[RankedCandidate]:
        """Return candidates below NLL threshold, sorted ascending by NLL.

        Returns empty list when candidates or obs are empty.
        """
        if not obs or not candidates:
            return []
        ranked = []
        for cand in candidates:
            try:
                with open(cand.archive_path, "rb") as f:
                    records = pickle.load(f)
            except Exception:
                continue
            patterns = [p for p, _w, _aid in records]
            if not patterns:
                continue
            nlls = [float(p.log_prob(o)) for p in patterns for o in obs]
            mean_nll = float(np.mean(nlls)) if nlls else float("inf")
            if mean_nll < self._threshold:
                ranked.append(RankedCandidate(candidate=cand, nll=mean_nll))
        ranked.sort(key=lambda rc: rc.nll)
        return ranked
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/agents/test_forecaster.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpm/agents/forecaster.py tests/agents/test_forecaster.py
git commit -m "feat: Phase 3 Forecaster with NLL ranking and threshold filtering"
```

---

### Task 13: Refactor `ContextualPatternStore` to delegate to `Librarian` and `Forecaster`

**Files:**
- Modify: `hpm/store/contextual_store.py`

- [ ] **Step 1: Confirm all existing tests pass before refactoring**

```bash
python3 -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Refactor `begin_context` to use `Librarian` and `Forecaster`**

Replace `begin_context` in `hpm/store/contextual_store.py`:

```python
    def begin_context(self, sig: SubstrateSignature, first_obs: list) -> str:
        context_id = str(uuid.uuid4())
        self._last_sig = sig

        from hpm.agents.librarian import Librarian
        from hpm.agents.forecaster import Forecaster
        librarian = Librarian()
        forecaster = Forecaster(fingerprint_nll_threshold=self._fingerprint_nll_threshold)

        candidates = librarian.query_archive(sig, self._archive_dir)
        ranked = forecaster.rank(candidates, first_obs)
        if ranked:
            self._load_archive(ranked[0].candidate.archive_path)
        elif candidates and not first_obs:
            # No observations to rank with: fall back to first coarse candidate
            self._load_archive(candidates[0].archive_path)

        self._inject_globals()
        return context_id
```

Replace `_run_global_pass` to delegate to `Librarian`:

```python
    def _run_global_pass(self, context_id: str) -> None:
        from hpm.agents.librarian import Librarian
        librarian = Librarian()
        tier2 = self._store.query_tier2_all()
        librarian.run_global_pass(
            tier2, context_id, self._db_path,
            self._global_weight_threshold, self._global_promotion_n
        )
```

Remove (or comment out) the now-redundant private methods `_coarse_filter`, `_fine_filter`, and `_mean_nll`.

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
python3 -m pytest tests/ -v
```

Expected: All tests pass — no behaviour changes, only delegation.

- [ ] **Step 4: Commit**

```bash
git add hpm/store/contextual_store.py
git commit -m "refactor: Phase 3 — ContextualPatternStore delegates to Librarian and Forecaster"
```

---

### Task 14: Phase 3 benchmark validation and final check

**Files:** No code changes.

- [ ] **Step 1: Run Phase 3 benchmark (first 100 tasks)**

```bash
python3 -c "
import benchmarks.multi_agent_arc as b
tasks = b.load_tasks()
eligible = [(i, t) for i, t in enumerate(tasks) if b.task_fits(t)][:100]
c, r = b.run_persistent(eligible, tasks)
n = len(eligible)
print(f'Phase 3 results: accuracy={c/n*100:.1f}%, mean_rank={r/n:.2f}')
"
```

Expected: Results match Phase 2 within +-0.5% accuracy (pure refactor — no behaviour change).

- [ ] **Step 2: Run final full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: Phase 3 complete — Librarian/Forecaster specialist roles, all tests passing"
```

---

## Summary

| Phase | New/Modified Files | Key Deliverable |
|-------|--------------------|-----------------|
| Phase 1 | `hpm/store/contextual_store.py`, `tests/store/test_contextual_store.py`, `benchmarks/multi_agent_arc.py` | Archive + warm-start with coarse/fine filter |
| Phase 2 | `hpm/config.py`, `hpm/store/contextual_store.py` (global pass + inject) | SQLite global pattern promotion and injection |
| Phase 3 | `hpm/agents/librarian.py`, `hpm/agents/forecaster.py`, `tests/agents/test_librarian.py`, `tests/agents/test_forecaster.py`, `hpm/store/contextual_store.py` (delegation) | Specialist role refactor, no behaviour change |
