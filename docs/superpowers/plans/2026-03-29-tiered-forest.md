# TieredForest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two-level memory tiering to the HFN Forest so the Observer can run indefinitely without OOM, by keeping only recently-used nodes in RAM and paging dormant nodes to disk.

**Architecture:** `TieredForest` subclasses `Forest` with an LRU hot cache (`OrderedDict`, capped at `max_hot`), a `_mu_index` dict of all mus (always in RAM for fast distance screening), and a disk cold store (`{cold_dir}/{id}.npz`). Two Observer lines that directly access `forest._registry[id]` are replaced with `forest.get(id)`. World model builders accept an optional `forest_cls` parameter so experiments can opt in.

**Tech Stack:** Python, numpy (`np.savez_compressed`), psutil (new dependency), collections.OrderedDict

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `hfn/tiered_forest.py` | Create | TieredForest class — all tiering logic |
| `hfn/observer.py` | Modify (2 lines) | Replace `_registry[id]` with `get(id)` |
| `hpm_fractal_node/nlp/nlp_world_model.py` | Modify | Accept `forest_cls` + `**kwargs` |
| `hpm_fractal_node/dsprites/dsprites_world_model.py` | Modify | Accept `forest_cls` + `**kwargs` |
| `hpm_fractal_node/experiments/experiment_nlp.py` | Modify | Use TieredForest |
| `tests/hfn/__init__.py` | Create | Empty — enables test discovery |
| `tests/hfn/test_tiered_forest.py` | Create | Unit tests |
| `requirements.txt` | Modify | Add `psutil>=5.9` |

---

### Task 1: Scaffold + psutil dependency

**Files:**
- Create: `tests/hfn/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add psutil to requirements.txt**

Open `requirements.txt`. Add this line:
```
psutil>=5.9
```

- [ ] **Step 2: Create tests/hfn/__init__.py**

```bash
touch tests/hfn/__init__.py
```

- [ ] **Step 3: Write the failing import test**

Create `tests/hfn/test_tiered_forest.py`:
```python
import psutil
import pytest

def test_psutil_importable():
    mem = psutil.virtual_memory()
    assert mem.available > 0
```

- [ ] **Step 4: Run test — expect FAIL if psutil not installed**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py::test_psutil_importable -v
```
If it fails with `ModuleNotFoundError`, install: `pip install psutil>=5.9`

- [ ] **Step 5: Run test — expect PASS**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py::test_psutil_importable -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/hfn/__init__.py tests/hfn/test_tiered_forest.py
git commit -m "feat: scaffold TieredForest tests and add psutil dependency"
```

---

### Task 2: TieredForest skeleton — register, active_nodes, __contains__, __len__, hot_count, cold_count

**Files:**
- Create: `hfn/tiered_forest.py`
- Modify: `tests/hfn/test_tiered_forest.py`

**Context:** `Forest` is in `hfn/forest.py`. `HFN` has fields: `mu` (np.ndarray), `sigma` (np.ndarray), `id` (str). `Forest.__init__` takes `D` and `forest_id`. The `_registry` dict in `Forest` must NOT be used in `TieredForest` — we replace it with `_hot` + `_mu_index`.

- [ ] **Step 1: Write failing tests**

Add to `tests/hfn/test_tiered_forest.py`:
```python
import tempfile
from typing import Optional
import numpy as np
from pathlib import Path
from hfn import HFN, Forest
from hfn.tiered_forest import TieredForest


def _make_node(d: int = 4, node_id: Optional[str] = None) -> HFN:
    node = HFN(mu=np.zeros(d), sigma=np.eye(d))
    if node_id:
        node.id = node_id
    return node


def test_tiered_forest_is_forest():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        assert isinstance(tf, Forest)


def test_register_makes_node_active():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        node = _make_node()
        tf.register(node)
        assert node in tf.active_nodes()


def test_contains_after_register():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        node = _make_node(node_id="abc")
        tf.register(node)
        assert "abc" in tf


def test_len_after_register():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        tf.register(_make_node())
        tf.register(_make_node())
        assert len(tf) == 2


def test_hot_count_and_cold_count_initial():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        tf.register(_make_node())
        assert tf.hot_count() == 1
        assert tf.cold_count() == 0


def test_deregister_removes_node():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        node = _make_node(node_id="xyz")
        tf.register(node)
        tf.deregister("xyz")
        assert "xyz" not in tf
        assert len(tf) == 0


def test_deregister_protected_is_noop():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td,
                          protected_ids={"prot"})
        node = _make_node(node_id="prot")
        tf.register(node)
        tf.deregister("prot")  # should be no-op
        assert "prot" in tf
        assert len(tf) == 1
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py -k "not psutil" -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'hfn.tiered_forest'`

- [ ] **Step 3: Implement TieredForest skeleton**

Create `hfn/tiered_forest.py`:
```python
"""
TieredForest — two-level memory tiering for the HFN Forest.

Hot cache (RAM): LRU-capped OrderedDict of full HFN nodes.
Cold store (disk): {cold_dir}/{node_id}.npz for evicted nodes.
mu_index (RAM): mu vectors for ALL nodes (hot + cold) — fast distance screening.

See docs/superpowers/specs/2026-03-29-tiered-forest-design.md
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import psutil

from hfn.forest import Forest
from hfn.hfn import HFN


class TieredForest(Forest):

    def __init__(
        self,
        D: int,
        forest_id: str = "forest",
        cold_dir: str | Path = "/tmp/hfn_cold",
        max_hot: int = 500,
        persistence_floor: float = 0.1,
        sweep_every: int = 100,
        min_free_ram_mb: int = 500,
        protected_ids: set[str] | None = None,
    ) -> None:
        super().__init__(D=D, forest_id=forest_id)
        self._cold_dir = Path(cold_dir)
        self._cold_dir.mkdir(parents=True, exist_ok=True)
        # Clear any stale .npz files from a previous run
        for f in self._cold_dir.glob("*.npz"):
            f.unlink()

        self._max_hot = max_hot
        self._persistence_floor = persistence_floor
        self._sweep_every = sweep_every
        self._min_free_ram_mb = min_free_ram_mb
        self._protected_ids: set[str] = set(protected_ids or [])

        # Hot cache: OrderedDict maintains LRU order (most-recently-used at end)
        self._hot: OrderedDict[str, HFN] = OrderedDict()
        # mu_index: all nodes (hot + cold)
        self._mu_index: dict[str, np.ndarray] = {}
        # Observation counter for sweep triggering
        self._obs_count: int = 0

    # ------------------------------------------------------------------
    # Forest interface overrides
    # ------------------------------------------------------------------

    def register(self, node: HFN) -> None:
        if node.id in self._mu_index:
            return  # idempotent
        self._mu_index[node.id] = node.mu.copy()
        self._hot[node.id] = node
        self._evict_lru_if_needed()
        self._sync_gaussian()

    def deregister(self, node_id: str) -> None:
        if node_id in self._protected_ids:
            return  # no-op for protected nodes
        self._mu_index.pop(node_id, None)
        self._hot.pop(node_id, None)
        cold_path = self._cold_path(node_id)
        if cold_path.exists():
            cold_path.unlink()
        self._sync_gaussian()

    def active_nodes(self) -> list[HFN]:
        """Return hot nodes only. See spec Section 3 for HPM rationale."""
        return list(self._hot.values())

    def __contains__(self, node_id: str) -> bool:  # type: ignore[override]
        return node_id in self._mu_index

    def __len__(self) -> int:
        return len(self._mu_index)

    # ------------------------------------------------------------------
    # New methods
    # ------------------------------------------------------------------

    def set_protected(self, ids: set[str]) -> None:
        self._protected_ids = set(ids)

    def hot_count(self) -> int:
        return len(self._hot)

    def cold_count(self) -> int:
        return len(self._mu_index) - len(self._hot)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cold_path(self, node_id: str) -> Path:
        # node_id may be a UUID or a descriptive string with special chars
        safe = node_id.replace("/", "_").replace("\\", "_")
        return self._cold_dir / f"{safe}.npz"

    def _evict_to_cold(self, node_id: str) -> None:
        """Write a hot node to disk and remove from _hot."""
        node = self._hot.pop(node_id, None)
        if node is None:
            return
        path = self._cold_path(node_id)
        np.savez_compressed(
            path,
            mu=node.mu,
            sigma=node.sigma,
            node_id=np.array([node.id]),
        )

    def _load_from_cold(self, node_id: str) -> HFN | None:
        """Load a cold node into _hot. Returns the node or None if not found."""
        path = self._cold_path(node_id)
        if not path.exists():
            return None
        data = np.load(path)
        node = HFN(
            mu=data["mu"].copy(),
            sigma=data["sigma"].copy(),
            id=str(data["node_id"][0]),
        )
        path.unlink()
        self._hot[node_id] = node
        self._hot.move_to_end(node_id)  # mark as most-recently-used
        self._evict_lru_if_needed()
        return node

    def _evict_lru_if_needed(self) -> None:
        """Evict least-recently-used hot nodes until len(_hot) <= max_hot."""
        while len(self._hot) > self._max_hot:
            lru_id, _ = next(iter(self._hot.items()))
            self._evict_to_cold(lru_id)

    def _sync_gaussian(self) -> None:
        """Override: compute from hot nodes only (approximation)."""
        hot = list(self._hot.values())
        if not hot:
            self.mu = np.zeros(self._D)
            self.sigma = np.eye(self._D)
        else:
            mus = np.stack([n.mu for n in hot])
            self.mu = mus.mean(axis=0)
            sigmas = np.stack([n.sigma for n in hot])
            self.sigma = sigmas.mean(axis=0)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py -k "not psutil" -v
```
Expected: all 8 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add hfn/tiered_forest.py tests/hfn/test_tiered_forest.py
git commit -m "feat: TieredForest skeleton with register/deregister/active_nodes"
```

---

### Task 3: LRU eviction to cold and promotion back to hot

**Files:**
- Modify: `tests/hfn/test_tiered_forest.py`
- Modify: `hfn/tiered_forest.py` (already has the helpers — just needs tests)

- [ ] **Step 1: Write failing tests**

Add to `tests/hfn/test_tiered_forest.py`:
```python
def test_lru_eviction_to_cold_when_max_hot_exceeded():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=2)
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        n3 = _make_node(node_id="n3")
        tf.register(n1)
        tf.register(n2)
        tf.register(n3)  # triggers LRU eviction of n1
        assert tf.hot_count() == 2
        assert tf.cold_count() == 1
        assert "n1" in tf        # still known via mu_index
        assert "n1" not in tf._hot  # but evicted from hot


def test_all_nodes_in_contains_after_eviction():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=2)
        for i in range(5):
            tf.register(_make_node(node_id=f"n{i}"))
        assert len(tf) == 5
        assert tf.hot_count() == 2
        assert tf.cold_count() == 3


def test_cold_file_exists_after_eviction():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        tf.register(n1)
        tf.register(n2)
        cold_files = list(Path(td).glob("*.npz"))
        assert len(cold_files) == 1


def test_deregister_cold_node_removes_file():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        tf.register(_make_node(node_id="n1"))
        tf.register(_make_node(node_id="n2"))
        # n1 is now cold
        tf.deregister("n1")
        assert "n1" not in tf
        assert "n2" in tf._hot  # n2 was most-recently-used, stays hot
        cold_files = list(Path(td).glob("*.npz"))
        assert len(cold_files) == 0  # n1 .npz deleted
```

- [ ] **Step 2: Run tests — expect PASS (code already written in Task 2)**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py -k "eviction or cold" -v
```
Expected: all 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/hfn/test_tiered_forest.py
git commit -m "test: LRU eviction and cold store tests for TieredForest"
```

---

### Task 4: retrieve() and get() — mu_index screening and cold promotion

**Files:**
- Modify: `hfn/tiered_forest.py`
- Modify: `tests/hfn/test_tiered_forest.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/hfn/test_tiered_forest.py`:
```python
def test_retrieve_finds_cold_node():
    """retrieve() should find and promote a cold node if it's in top-k."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        # n1: mu=[1,0,0,0], n2: mu=[0,1,0,0]
        n1 = HFN(mu=np.array([1.,0.,0.,0.]), sigma=np.eye(4), id="n1")
        n2 = HFN(mu=np.array([0.,1.,0.,0.]), sigma=np.eye(4), id="n2")
        tf.register(n1)
        tf.register(n2)  # n1 evicted to cold
        # Query close to n1
        results = tf.retrieve(np.array([0.9, 0., 0., 0.]), k=1)
        assert len(results) == 1
        assert results[0].id == "n1"
        # n1 should be promoted back to hot
        assert "n1" in tf._hot


def test_retrieve_returns_up_to_k_nodes():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=10)
        # Use 5 distinct mus (padded eye rows) to avoid distance ties
        mus = [np.array([float(i==j) for j in range(4)]) for i in range(4)]
        mus.append(np.array([0.5, 0.5, 0., 0.]))
        for i, mu in enumerate(mus):
            tf.register(HFN(mu=mu, sigma=np.eye(4), id=f"n{i}"))
        results = tf.retrieve(np.zeros(4), k=3)
        assert len(results) == 3


def test_get_loads_cold_node():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        tf.register(n1)
        tf.register(n2)  # n1 evicted to cold
        loaded = tf.get("n1")
        assert loaded is not None
        assert loaded.id == "n1"
        assert "n1" in tf._hot  # promoted to hot


def test_get_returns_none_for_unknown_node():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        result = tf.get("nonexistent")
        assert result is None
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py -k "retrieve or get" -v
```
Expected: FAIL with `AttributeError: 'TieredForest' object has no attribute 'get'`

- [ ] **Step 3: Implement retrieve() and get()**

Add to `TieredForest` in `hfn/tiered_forest.py`:
```python
def retrieve(self, x: np.ndarray, k: int = 5) -> list[HFN]:
    """
    Screen _mu_index by Euclidean distance, load cold top-k candidates,
    return up to k full HFN nodes (hot or promoted from cold).
    """
    if not self._mu_index:
        return []
    scored = sorted(
        self._mu_index.keys(),
        key=lambda nid: float(np.linalg.norm(self._mu_index[nid] - x)),
    )
    result = []
    for node_id in scored[:k]:
        if node_id in self._hot:
            self._hot.move_to_end(node_id)  # refresh LRU
            result.append(self._hot[node_id])
        else:
            node = self._load_from_cold(node_id)
            if node is not None:
                result.append(node)
    return result

def get(self, node_id: str) -> HFN | None:
    """Return a full HFN by id, loading from cold if needed."""
    if node_id in self._hot:
        self._hot.move_to_end(node_id)
        return self._hot[node_id]
    if node_id in self._mu_index:
        return self._load_from_cold(node_id)
    return None
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py -k "retrieve or get" -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Run full test suite — no regressions**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/ -v
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add hfn/tiered_forest.py tests/hfn/test_tiered_forest.py
git commit -m "feat: TieredForest retrieve() and get() with cold promotion"
```

---

### Task 5: Sweep — psutil RAM check + LRU-based eviction

**Files:**
- Modify: `hfn/tiered_forest.py`
- Modify: `tests/hfn/test_tiered_forest.py`

**Note:** The sweep uses LRU order as a proxy for persistence (least-recently-used nodes evicted first). True `persistence_scores` requires the Observer's weights dict which the Forest doesn't have access to. The `persistence_floor` parameter is kept in the constructor for API compatibility but LRU order is used internally.

- [ ] **Step 1: Write failing tests**

Add to `tests/hfn/test_tiered_forest.py`:
```python
def test_sweep_evicts_lru_hot_nodes_under_ram_pressure(monkeypatch):
    """Simulate low RAM: sweep should evict bottom half of hot nodes."""
    import psutil

    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=10, sweep_every=5, min_free_ram_mb=999_999,  # always triggers
        )
        for i in range(6):
            tf.register(_make_node(node_id=f"n{i}"))

        # Simulate sweep_every observations
        for _ in range(5):
            tf._on_observe()

        # Bottom half (3 nodes) should be evicted to cold
        assert tf.hot_count() == 3
        assert tf.cold_count() == 3
        assert len(tf) == 6  # all still known


def test_sweep_does_not_delete_protected_nodes():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=10, sweep_every=5, min_free_ram_mb=999_999,
            protected_ids={"prot"},
        )
        tf.register(_make_node(node_id="prot"))
        for i in range(5):
            tf.register(_make_node(node_id=f"n{i}"))

        for _ in range(5):
            tf._on_observe()

        # prot should still exist (may be cold but not deleted)
        assert "prot" in tf


def test_sweep_step2_deletes_unprotected_cold_nodes():
    """Step 2: nodes evicted to cold and not re-promoted are deleted on next sweep."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=2, sweep_every=5, min_free_ram_mb=0,  # no RAM pressure
        )
        # Register 4 nodes: n0,n1 in hot, n2,n3 evicted to cold
        for i in range(4):
            tf.register(_make_node(node_id=f"n{i}"))
        assert tf.cold_count() == 2  # n0,n1 cold; n2,n3 hot

        # Trigger sweep (no RAM pressure, just persistence floor)
        for _ in range(5):
            tf._on_observe()

        # Cold unprotected nodes (n0, n1) should be deleted entirely
        assert "n0" not in tf
        assert "n1" not in tf
        assert len(tf) == 2  # only n2, n3 remain


def test_sweep_step2_keeps_protected_cold_nodes():
    """Protected cold nodes are NOT deleted by persistence floor — evicted only."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=2, sweep_every=5, min_free_ram_mb=0,
            protected_ids={"n0"},
        )
        for i in range(4):
            tf.register(_make_node(node_id=f"n{i}"))

        for _ in range(5):
            tf._on_observe()

        # n0 is protected and cold — must still exist
        assert "n0" in tf
        # n1 is unprotected and cold — deleted
        assert "n1" not in tf


def test_on_observe_increments_counter():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, sweep_every=10)
        assert tf._obs_count == 0
        tf._on_observe()
        assert tf._obs_count == 1
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py -k "sweep or on_observe" -v
```
Expected: FAIL with `AttributeError: 'TieredForest' object has no attribute '_on_observe'`

- [ ] **Step 3: Implement _on_observe() and _sweep()**

Add to `TieredForest` in `hfn/tiered_forest.py`:
```python
def _on_observe(self) -> None:
    """Call once per observation. Triggers sweep every sweep_every calls."""
    self._obs_count += 1
    if self._obs_count % self._sweep_every == 0:
        self._sweep()

def _sweep(self) -> None:
    """
    Two-step sweep:

    Step 1 — Emergency RAM check: if free RAM < min_free_ram_mb,
    evict the bottom half of hot nodes (LRU = front of OrderedDict).
    Both protected and unprotected go to cold; neither is deleted.

    Step 2 — Persistence floor (operates on post-step-1 state):
    Any node currently cold (in _mu_index but not in _hot) is treated as
    dormant. Unprotected dormant nodes are deleted entirely. Protected
    dormant nodes stay cold (kept in _mu_index and on disk).
    """
    # Step 1: emergency RAM eviction
    free_mb = psutil.virtual_memory().available / (1024 * 1024)
    if free_mb < self._min_free_ram_mb:
        hot_ids = list(self._hot.keys())
        n_evict = max(1, len(hot_ids) // 2)
        for node_id in hot_ids[:n_evict]:
            self._evict_to_cold(node_id)

    # Step 2: delete unprotected cold nodes (not re-promoted since last sweep)
    cold_ids = [nid for nid in list(self._mu_index.keys()) if nid not in self._hot]
    for node_id in cold_ids:
        if node_id not in self._protected_ids:
            self._mu_index.pop(node_id, None)
            cold_path = self._cold_path(node_id)
            if cold_path.exists():
                cold_path.unlink()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py -k "sweep or on_observe" -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/ -v
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add hfn/tiered_forest.py tests/hfn/test_tiered_forest.py
git commit -m "feat: TieredForest sweep with psutil RAM check and LRU eviction"
```

---

### Task 6: observer.py — 2-line fix

**Files:**
- Modify: `hfn/observer.py` (lines 510, 511, 521, 522)

**Context:** These 4 lines access `self.forest._registry[ids[0]]` and `self.forest._registry[ids[1]]` directly. Replace with `self.forest.get(ids[0])` etc. The `get()` method exists on both `Forest` (as a dict-style lookup via `_registry`) and `TieredForest` (with cold loading). We need to add `get()` to `Forest` as well for API consistency.

- [ ] **Step 1: Add get() to Forest base class**

In `hfn/forest.py`, add after `active_nodes()`:
```python
def get(self, node_id: str) -> HFN | None:
    """Return node by id, or None if not found."""
    return self._registry.get(node_id)
```

- [ ] **Step 2: Write a test verifying Forest.get() works**

Add to `tests/hfn/test_tiered_forest.py`:
```python
def test_forest_base_get():
    from hfn import Forest, HFN
    f = Forest(D=4, forest_id="f")
    node = _make_node(node_id="abc")
    f.register(node)
    assert f.get("abc") is node
    assert f.get("missing") is None
```

- [ ] **Step 3: Run test — expect FAIL**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py::test_forest_base_get -v
```
Expected: FAIL (Forest has no get())

- [ ] **Step 4: Apply forest.py change and observer.py fix**

> **Invariant note:** Both `ids[0]` and `ids[1]` at these call sites come from `self._cooccurrence`, which only records IDs of nodes that were registered at co-occurrence time. Both IDs are guaranteed to be present in `_mu_index`, so `forest.get()` will never return `None` here. No null guard is needed.

In `hfn/observer.py`, find lines 510-511 and 521-522:
```python
# BEFORE (lines 510-511):
                a_mu = self.forest._registry[ids[0]].mu
                b_mu = self.forest._registry[ids[1]].mu
# AFTER:
                a_mu = self.forest.get(ids[0]).mu
                b_mu = self.forest.get(ids[1]).mu

# BEFORE (lines 521-522):
            node_a = self.forest._registry[ids[0]]
            node_b = self.forest._registry[ids[1]]
# AFTER:
            node_a = self.forest.get(ids[0])
            node_b = self.forest.get(ids[1])
```

- [ ] **Step 5: Run all tests — expect PASS**

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
Expected: all tests PASS (including existing observer/hfn tests)

- [ ] **Step 6: Commit**

```bash
git add hfn/forest.py hfn/observer.py tests/hfn/test_tiered_forest.py
git commit -m "feat: add Forest.get() and fix observer.py direct _registry access"
```

---

### Task 7: World model builders accept forest_cls

**Files:**
- Modify: `hpm_fractal_node/nlp/nlp_world_model.py`
- Modify: `hpm_fractal_node/dsprites/dsprites_world_model.py`
- Modify: `tests/hfn/test_tiered_forest.py`

- [ ] **Step 1: Write failing test**

Add to `tests/hfn/test_tiered_forest.py`:
```python
def test_build_nlp_world_model_accepts_tiered_forest():
    import tempfile
    from hfn.tiered_forest import TieredForest
    from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model
    with tempfile.TemporaryDirectory() as td:
        forest, prior_ids = build_nlp_world_model(
            forest_cls=TieredForest,
            cold_dir=td,
            max_hot=100,
        )
        assert isinstance(forest, TieredForest)
        assert len(prior_ids) == 38
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
PYTHONPATH=. python3 -m pytest tests/hfn/test_tiered_forest.py::test_build_nlp_world_model_accepts_tiered_forest -v
```
Expected: FAIL with `TypeError: build_nlp_world_model() got unexpected keyword argument 'forest_cls'`

- [ ] **Step 3: Modify nlp_world_model.py**

> **Note:** The spec shows `forest_cls=Forest` as default. The plan uses `forest_cls=None` with a runtime check to avoid a circular import at module level (importing Forest at the top of nlp_world_model.py is fine, but defaulting to it in the signature requires it to be importable before the function is called — which it is, but using `None` + a local import is safer across refactors).

In `hpm_fractal_node/nlp/nlp_world_model.py`, change the function signature and forest construction:

```python
# BEFORE:
def build_nlp_world_model() -> tuple[Forest, set[str]]:
    forest = Forest(D=D, forest_id="nlp_child")

# AFTER:
def build_nlp_world_model(forest_cls=None, **tiered_kwargs) -> tuple[Forest, set[str]]:
    from hfn.forest import Forest as _Forest
    if forest_cls is None:
        forest_cls = _Forest
    kwargs = tiered_kwargs if forest_cls is not _Forest else {}
    forest = forest_cls(D=D, forest_id="nlp_child", **kwargs)
```

- [ ] **Step 4: Modify dsprites_world_model.py the same way**

In `hpm_fractal_node/dsprites/dsprites_world_model.py`:
```python
# BEFORE:
def build_dsprites_world_model() -> tuple[Forest, set[str]]:
    forest = Forest(D=D, forest_id="dsprites_16x16")

# AFTER:
def build_dsprites_world_model(forest_cls=None, **tiered_kwargs) -> tuple[Forest, set[str]]:
    from hfn.forest import Forest as _Forest
    if forest_cls is None:
        forest_cls = _Forest
    kwargs = tiered_kwargs if forest_cls is not _Forest else {}
    forest = forest_cls(D=D, forest_id="dsprites_16x16", **kwargs)
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add hpm_fractal_node/nlp/nlp_world_model.py hpm_fractal_node/dsprites/dsprites_world_model.py tests/hfn/test_tiered_forest.py
git commit -m "feat: world model builders accept forest_cls for TieredForest"
```

---

### Task 8: Wire TieredForest into experiment_nlp.py + run

**Files:**
- Modify: `hpm_fractal_node/experiments/experiment_nlp.py`

- [ ] **Step 1: Update experiment_nlp.py**

In `hpm_fractal_node/experiments/experiment_nlp.py`, add the import and update `main()`:

```python
# Add import at top (after existing imports):
import tempfile
from hfn.tiered_forest import TieredForest

# In main(), change:
# BEFORE:
    forest, prior_ids = build_nlp_world_model()
# AFTER:
    _cold_dir = Path(__file__).parents[2] / "data" / "hfn_nlp_cold"
    _cold_dir.mkdir(parents=True, exist_ok=True)
    forest, prior_ids = build_nlp_world_model(
        forest_cls=TieredForest,
        cold_dir=_cold_dir,
        max_hot=500,
    )
    forest.set_protected(prior_ids)
```

Also wire `_on_observe()` into the observation loop. `forest` is a local variable in `main()` in scope at this point. After `result = obs.observe(x)`, add:
```python
            forest._on_observe()
```

The surrounding context should look like:
```python
            result = obs.observe(x)
            forest._on_observe()   # ← add this line
            if (n_explained + n_unexplained) % 500 == 0:
```

- [ ] **Step 2: Run experiment (N_PASSES=2)**

```bash
PYTHONPATH=. python3 -c "
import hpm_fractal_node.experiments.experiment_nlp as e
e.N_PASSES = 2
e.main()
" 2>&1
```
Expected: completes without OOM, prints full results including learned node alignment table

- [ ] **Step 3: Commit**

```bash
git add hpm_fractal_node/experiments/experiment_nlp.py
git commit -m "feat: wire TieredForest into NLP experiment"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run full test suite**

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 2: Run experiment with N_PASSES=3 (full run)**

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py 2>&1
```
Expected: completes all 3 passes without OOM, prints full analysis

- [ ] **Step 3: Final commit**

```bash
git add hfn/tiered_forest.py hfn/forest.py hfn/observer.py \
    hpm_fractal_node/nlp/nlp_world_model.py \
    hpm_fractal_node/dsprites/dsprites_world_model.py \
    hpm_fractal_node/experiments/experiment_nlp.py \
    tests/hfn/__init__.py tests/hfn/test_tiered_forest.py \
    requirements.txt
git commit -m "feat: TieredForest complete — memory-tiered HFN Forest with disk cold store"
```
