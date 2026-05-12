# PatternStore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `PatternStore` — shared, cross-agent, cross-corpus pattern storage with cosine-similarity-based merging and running-average weight accumulation — and wire it into `MultiAgentReader`.

**Architecture:** A standalone `PatternStore` class stores pattern embeddings in a NumPy `.npz` file at `<cache_dir>/shared/patterns.npz`. On each `train_sequence()` call, `MultiAgentReader` calls `pattern_store.merge()` for each agent then `pattern_store.save()`. On warm-start, `pattern_store.load()` returns all stored patterns so every agent can be seeded with cross-corpus knowledge.

**Tech Stack:** Python 3.10+, NumPy (already in requirements), Pydantic (`Cell` is a `BaseModel`), pytest.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| CREATE | `hpm_ai_v6/hpm_model/storage/__init__.py` | Empty package marker |
| CREATE | `hpm_ai_v6/hpm_model/storage/pattern_store.py` | `PatternStore` class — all storage logic |
| CREATE | `hpm_ai_v6/tests/test_pattern_store.py` | All unit + integration tests (10 tests) |
| MODIFY | `hpm_ai_v6/agents/multi_agent_reader.py` | Import and wire `PatternStore` into constructor, `warm_start_from_cache`, and `train_sequence` |

---

## Background you need

### `Cell` (read `hpm_ai_v6/hpm_model/core/cell.py`)

`Cell` is a Pydantic `BaseModel` with:
- `name: str`
- `embedding: Any` — usually a `np.ndarray` or `torch.Tensor`
- `weight: float = 1.0`
- `dim: int = 0`
- `as_numpy() -> np.ndarray` — returns embedding as float64 numpy array

To create a test `Cell` with a known embedding:
```python
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell

cell = Cell(name="test", embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32))
```

### `MultiAgentReader` (read `hpm_ai_v6/agents/multi_agent_reader.py`)

- Constructor sets `self.pattern_cache_dir = pattern_cache_dir or self._default_pattern_cache_dir()`
- `_default_pattern_cache_dir()` returns `<corpus_dir>/.hpm_pattern_cache/<corpus_name>/`
- `self.agents` dict maps string names to agent instances: `"char"`, `"word"`, `"contextual"`, `"phrase"`, `"semantic"`, `"causal"`, etc.
- `warm_start_from_cache()` currently calls `agent.hydrate_patterns_from_archive()` per agent
- `train_sequence(sentences)` processes sentences through char/word/contextual/phrase/semantic agents but does **not** currently interact with any shared store
- Agents expose `agent.patterns` (list of `Cell`) and `agent.get_weights()` (returns iterable of floats)

### Cosine similarity formula (used internally by PatternStore)

```python
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
```

For a query vector vs a matrix of stored vectors:
```python
# norms shape (N,), query_norm scalar
norms = np.linalg.norm(stored_vectors, axis=1)  # (N,)
query_norm = np.linalg.norm(query_vec)
sims = stored_vectors @ query_vec / (norms * query_norm + 1e-9)  # (N,)
```

---

## Task 1: Package skeleton and save/load round-trip (empty store)

**Files:**
- Create: `hpm_ai_v6/hpm_model/storage/__init__.py`
- Create: `hpm_ai_v6/hpm_model/storage/pattern_store.py`
- Create: `hpm_ai_v6/tests/test_pattern_store.py`

### Step 1.1: Write the failing tests
- [ ] Write to `hpm_ai_v6/tests/test_pattern_store.py`:

```python
"""Tests for PatternStore."""
import os
import tempfile

import numpy as np
import pytest

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.storage.pattern_store import PatternStore


def _make_cell(name: str, vec: list[float]) -> Cell:
    return Cell(name=name, embedding=np.array(vec, dtype=np.float32))


# ---------------------------------------------------------------------------
# Task 1: skeleton / save / load
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_empty():
    """load() returns ([], []) when no patterns.npz file exists."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        patterns, weights = store.load()
        assert patterns == []
        assert weights == []


def test_save_load_roundtrip():
    """After save, a fresh PatternStore load restores all data correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("hello", [1.0, 0.0, 0.0])
        store.merge([cell], [2.5], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()

        assert len(patterns) == 1
        assert patterns[0].name == "hello"
        assert abs(weights[0] - 2.5) < 1e-5
        np.testing.assert_allclose(
            patterns[0].as_numpy()[:3].astype(np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-5,
        )


def test_clear_resets_in_memory_state():
    """clear() empties in-memory store; a subsequent load() still returns file contents."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("alpha", [0.0, 1.0, 0.0])
        store.merge([cell], [1.0], agent_name="char_agent")
        store.save()

        # clear in-memory
        store.clear()
        # in-memory is empty
        assert store._vectors is None or len(store._vectors) == 0

        # but loading from disk returns the saved pattern
        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 1
        assert patterns[0].name == "alpha"


# ---------------------------------------------------------------------------
# Task 2: merge() -- cosine similarity clustering and running average
# ---------------------------------------------------------------------------

def test_merge_idempotent_running_average():
    """Merging the same pattern twice gives running average weight, count=2."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("word", [1.0, 0.0, 0.0])
        store.merge([cell], [2.0], agent_name="word_agent")
        store.merge([cell], [4.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 1
        assert abs(weights[0] - 3.0) < 1e-5
        assert store2._counts[0] == 2


def test_merge_below_threshold_stays_separate():
    """Two patterns with cosine similarity < 0.85 produce two separate entries."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        # Orthogonal vectors: cosine similarity = 0.0
        a = _make_cell("alpha", [1.0, 0.0, 0.0])
        b = _make_cell("beta",  [0.0, 1.0, 0.0])
        store.merge([a], [1.0], agent_name="word_agent")
        store.merge([b], [1.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 2


def test_merge_above_threshold_merges():
    """Two vectors with cosine similarity >= 0.85 are treated as the same pattern."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        # Nearly identical vectors: sim ~= 0.9999
        a = _make_cell("run",    [1.0, 0.01, 0.0])
        b = _make_cell("running", [1.0, 0.02, 0.0])
        store.merge([a], [1.0], agent_name="word_agent")
        store.merge([b], [3.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 1
        # running average: (1.0*1 + 3.0) / 2 = 2.0
        assert abs(weights[0] - 2.0) < 1e-4


def test_agents_field_accumulates():
    """Two different agents contributing to the same pattern both appear in agents field."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("concept", [1.0, 0.0, 0.0])
        store.merge([cell], [1.0], agent_name="word_agent")
        store.merge([cell], [2.0], agent_name="phrase_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        store2.load()
        agents_field = store2._agents[0]
        assert "word_agent" in agents_field
        assert "phrase_agent" in agents_field


# ---------------------------------------------------------------------------
# Task 3: find_similar()
# ---------------------------------------------------------------------------

def test_find_similar_empty_store_returns_empty():
    """find_similar() returns [] when the store is empty."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.find_similar(query, top_k=5)
        assert results == []


def test_find_similar_top_k_sorted_descending():
    """find_similar() returns top-k results sorted by descending cosine similarity."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        # Three vectors; query = [1,0,0]
        # Similarities: a=1.0, b~=0.707, c=0.0
        a = _make_cell("a", [1.0, 0.0, 0.0])
        b = _make_cell("b", [1.0, 1.0, 0.0])
        c = _make_cell("c", [0.0, 1.0, 0.0])
        store.merge([a], [1.0], agent_name="word_agent")
        store.merge([b], [1.0], agent_name="word_agent")
        store.merge([c], [1.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        store2.load()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store2.find_similar(query, top_k=3)

        assert len(results) == 3
        sims = [r[0] for r in results]
        # Must be sorted descending
        assert sims[0] >= sims[1] >= sims[2]
        # Top result must be "a"
        assert results[0][1].name == "a"
        assert abs(results[0][0] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Task 4: integration -- simulated train_sequence x2
# ---------------------------------------------------------------------------

def test_two_train_sequences_weight_grows():
    """Simulated train_sequence x2 with same pattern -> weight accumulates, count=2."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)

        # First training sequence: one agent, one pattern
        cell = _make_cell("dog", [1.0, 0.0, 0.0])
        store.merge([cell], [1.0], agent_name="word_agent")
        store.save()

        # Second training sequence: same pattern, higher weight
        store2 = PatternStore(cache_dir=tmp)
        store2.load()
        store2.merge([cell], [3.0], agent_name="word_agent")
        store2.save()

        # Reload and check final state
        store3 = PatternStore(cache_dir=tmp)
        patterns, weights = store3.load()

        assert len(patterns) == 1
        # Running average over two observations: (1.0 + 3.0) / 2 = 2.0
        assert abs(weights[0] - 2.0) < 1e-4
        assert store3._counts[0] == 2
```

### Step 1.2: Run the tests to confirm they fail
- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_pattern_store.py -v 2>&1 | head -30`
- Expected: `ImportError` or `ModuleNotFoundError` for `pattern_store`

### Step 1.3: Create the package marker
- [ ] Write to `hpm_ai_v6/hpm_model/storage/__init__.py`:

```python
"""HPM pattern storage package."""
```

### Step 1.4: Implement PatternStore
- [ ] Write to `hpm_ai_v6/hpm_model/storage/pattern_store.py`:

```python
"""Shared cross-agent cross-corpus PatternStore.

Patterns are identified by embedding cosine similarity (threshold 0.85 by
default).  Weights are merged using a running average; observation counts are
tracked per entry.

Storage: <cache_dir>/shared/patterns.npz  (NumPy archive)

Array layout inside the .npz file
----------------------------------
vectors  : (N, D)  float32  -- embedding vectors
weights  : (N,)    float32  -- running-averaged weights
counts   : (N,)    int32    -- merge observation count
names    : (N,)    object   -- canonical pattern name (most recently seen)
agents   : (N,)    object   -- comma-separated agent names that contributed

Note on allow_pickle in np.load / np.savez
------------------------------------------
The `names` and `agents` arrays use numpy object dtype (Python strings).
NumPy requires allow_pickle=True to save/load these.  This is safe here
because we are the sole writer of the file -- no untrusted data is
deserialised.  The file lives under the project's own cache directory and
is never loaded from an external source.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from hpm_ai_v6.hpm_model.core.cell import Cell


class PatternStore:
    """Shared, cross-agent, cross-corpus pattern store."""

    def __init__(
        self,
        cache_dir: str,
        similarity_threshold: float = 0.85,
    ) -> None:
        """
        Args:
            cache_dir: Root cache directory (e.g. '.hpm_pattern_cache').
                       Store writes to <cache_dir>/shared/patterns.npz.
            similarity_threshold: Cosine similarity cutoff for merging.
                                  Patterns with sim >= threshold are treated
                                  as the same conceptual pattern.
        """
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self._store_path = os.path.join(cache_dir, "shared", "patterns.npz")

        # In-memory arrays; None means not yet loaded/initialised.
        self._vectors: Optional[np.ndarray] = None   # (N, D) float32
        self._weights: Optional[np.ndarray] = None   # (N,) float32
        self._counts: Optional[np.ndarray] = None    # (N,) int32
        self._names: Optional[np.ndarray] = None     # (N,) object
        self._agents: Optional[np.ndarray] = None    # (N,) object

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_empty(self) -> bool:
        return self._vectors is None or len(self._vectors) == 0

    def _init_empty(self) -> None:
        """Reset in-memory state to a valid but empty store."""
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._weights = np.empty((0,), dtype=np.float32)
        self._counts = np.empty((0,), dtype=np.int32)
        self._names = np.empty((0,), dtype=object)
        self._agents = np.empty((0,), dtype=object)

    def _ensure_loaded(self) -> None:
        """Ensure in-memory state is initialised (load from disk if needed)."""
        if self._vectors is None:
            self.load()

    def _cosine_similarities(self, query: np.ndarray) -> np.ndarray:
        """Return cosine similarity of query against all stored vectors."""
        norms = np.linalg.norm(self._vectors, axis=1)  # (N,)
        query_norm = float(np.linalg.norm(query))
        return self._vectors @ query / (norms * query_norm + 1e-9)  # (N,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        patterns: List[Cell],
        weights: List[float],
        agent_name: str,
    ) -> None:
        """Merge a batch of (pattern, weight) pairs into the store.

        For each pair:
        - If the store is non-empty and the maximum cosine similarity against
          stored vectors is >= similarity_threshold, update the matching entry
          using a running average and increment its count.
        - Otherwise append a new entry.

        Does NOT call save() -- the caller must call save() after the batch.
        """
        self._ensure_loaded()

        for cell, w in zip(patterns, weights):
            vec = cell.as_numpy().astype(np.float32)
            D = vec.shape[0]

            if self._is_empty():
                self._vectors = vec.reshape(1, D)
                self._weights = np.array([w], dtype=np.float32)
                self._counts = np.array([1], dtype=np.int32)
                self._names = np.array([cell.name], dtype=object)
                self._agents = np.array([agent_name], dtype=object)
                continue

            stored_D = self._vectors.shape[1]
            if D != stored_D:
                raise ValueError(
                    f"Embedding dimension mismatch: incoming vector has dim {D}, "
                    f"store expects dim {stored_D}."
                )

            sims = self._cosine_similarities(vec)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= self.similarity_threshold:
                n = int(self._counts[best_idx])
                self._weights[best_idx] = (
                    float(self._weights[best_idx]) * n + w
                ) / (n + 1)
                self._counts[best_idx] = n + 1
                self._names[best_idx] = cell.name
                existing_agents = self._agents[best_idx].split(",")
                if agent_name not in existing_agents:
                    self._agents[best_idx] = ",".join(existing_agents + [agent_name])
            else:
                self._vectors = np.vstack([self._vectors, vec.reshape(1, D)])
                self._weights = np.append(self._weights, np.float32(w))
                self._counts = np.append(self._counts, np.int32(1))
                self._names = np.append(self._names, cell.name)
                self._agents = np.append(self._agents, agent_name)

    def load(self) -> Tuple[List[Cell], List[float]]:
        """Load all stored patterns and their averaged weights from disk.

        Returns ([], []) if no store file exists yet.
        Also populates the in-memory state so subsequent merge() calls work.
        """
        if not os.path.exists(self._store_path):
            self._init_empty()
            return [], []

        # allow_pickle=True is required for object-dtype string arrays.
        # Safe -- we are the sole writer; see module docstring.
        data = np.load(self._store_path, allow_pickle=True)
        self._vectors = data["vectors"].astype(np.float32)
        self._weights = data["weights"].astype(np.float32)
        self._counts = data["counts"].astype(np.int32)
        self._names = data["names"].astype(object)
        self._agents = data["agents"].astype(object)

        cells = [
            Cell(name=str(name), embedding=vec)
            for name, vec in zip(self._names, self._vectors)
        ]
        weights = [float(w) for w in self._weights]
        return cells, weights

    def find_similar(
        self,
        vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[float, Cell]]:
        """Return the top-k most similar stored patterns (descending similarity).

        Returns [] if the store is empty.
        """
        self._ensure_loaded()
        if self._is_empty():
            return []

        query = np.asarray(vector, dtype=np.float32)
        sims = self._cosine_similarities(query)
        k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:k]

        return [
            (float(sims[i]), Cell(name=str(self._names[i]), embedding=self._vectors[i]))
            for i in top_indices
        ]

    def save(self) -> None:
        """Persist in-memory state to <cache_dir>/shared/patterns.npz.

        Creates the directory if it does not exist. Overwrites any existing file.
        """
        self._ensure_loaded()
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        # allow_pickle=True required for object-dtype string arrays.
        # Safe -- we are the sole writer; see module docstring.
        np.savez(
            self._store_path,
            vectors=self._vectors,
            weights=self._weights,
            counts=self._counts,
            names=self._names,
            agents=self._agents,
        )

    def clear(self) -> None:
        """Reset in-memory store to empty (does not delete the file on disk)."""
        self._init_empty()
```

### Step 1.5: Run all PatternStore tests
- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_pattern_store.py -v`
- Expected: All 10 tests PASS

### Step 1.6: Commit
- [ ] Run:
```bash
git add hpm_ai_v6/hpm_model/storage/__init__.py \
        hpm_ai_v6/hpm_model/storage/pattern_store.py \
        hpm_ai_v6/tests/test_pattern_store.py
git commit -m "feat: add PatternStore with save/load, merge, and find_similar"
```

---

## Task 2: Verify merge() tests pass (RED-GREEN confirmation)

Because all 10 tests are written in Task 1 and the implementation is complete in the same task, this task confirms the merge and find_similar behaviour explicitly.

**Files:** No changes — verification only.

### Step 2.1: Run merge-specific tests
- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_pattern_store.py::test_merge_idempotent_running_average hpm_ai_v6/tests/test_pattern_store.py::test_merge_below_threshold_stays_separate hpm_ai_v6/tests/test_pattern_store.py::test_merge_above_threshold_merges hpm_ai_v6/tests/test_pattern_store.py::test_agents_field_accumulates -v`
- Expected: All 4 PASS

### Step 2.2: Run find_similar tests
- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_pattern_store.py::test_find_similar_empty_store_returns_empty hpm_ai_v6/tests/test_pattern_store.py::test_find_similar_top_k_sorted_descending -v`
- Expected: Both PASS

---

## Task 3: MultiAgentReader wiring

**Files:**
- Modify: `hpm_ai_v6/agents/multi_agent_reader.py`

### Step 3.1: Confirm the import block (read the file first)
- [ ] Read lines 1-30 of `hpm_ai_v6/agents/multi_agent_reader.py` to confirm the final import line before editing.

The file currently ends its imports with:
```python
from hpm_ai_v6.hpm_model.core.cell import Cell
```

### Step 3.2: Add the PatternStore import
- [ ] In `hpm_ai_v6/agents/multi_agent_reader.py`, find the line:
```python
from hpm_ai_v6.hpm_model.core.cell import Cell
```
Replace it with:
```python
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.storage.pattern_store import PatternStore
```

### Step 3.3: Instantiate PatternStore in the constructor
- [ ] In `__init__`, find:
```python
        self.pattern_cache_dir = pattern_cache_dir or self._default_pattern_cache_dir()
        self.max_active_patterns = max_active_patterns
```
Replace with:
```python
        self.pattern_cache_dir = pattern_cache_dir or self._default_pattern_cache_dir()
        self.max_active_patterns = max_active_patterns
        self.pattern_store = PatternStore(cache_dir=self.pattern_cache_dir)
```

### Step 3.4: Update warm_start_from_cache to load from PatternStore
- [ ] Find the beginning of `warm_start_from_cache`:
```python
    def warm_start_from_cache(self, limit: Optional[int] = None) -> int:
        loaded = 0
        for name in ("char", "word", "contextual", "phrase", "semantic", "causal"):
```
Replace with:
```python
    def warm_start_from_cache(self, limit: Optional[int] = None) -> int:
        # Load shared cross-corpus patterns and warm-start every agent.
        shared_patterns, shared_weights = self.pattern_store.load()
        if shared_patterns:
            for agent in self.agents.values():
                if hasattr(agent, "warm_start"):
                    try:
                        agent.warm_start(shared_patterns, shared_weights)
                    except Exception:
                        pass

        loaded = 0
        for name in ("char", "word", "contextual", "phrase", "semantic", "causal"):
```

### Step 3.5: Update train_sequence to merge into PatternStore after training
- [ ] Find the end of `train_sequence`. The method currently ends with:
```python
        syn_agent = self.agents.get("syntactic")
        if syn_agent is not None and hasattr(syn_agent, "learn_from_corpus"):
            syn_agent.learn_from_corpus(sentences)
            if hasattr(syn_agent, "save"):
                cache_path = os.path.join(self.pattern_cache_dir, "syntactic_rules.json")
                try:
                    syn_agent.save(cache_path)
                except Exception:
                    pass
```
Replace with (adding the PatternStore merge block at the end, still inside `train_sequence`):
```python
        syn_agent = self.agents.get("syntactic")
        if syn_agent is not None and hasattr(syn_agent, "learn_from_corpus"):
            syn_agent.learn_from_corpus(sentences)
            if hasattr(syn_agent, "save"):
                cache_path = os.path.join(self.pattern_cache_dir, "syntactic_rules.json")
                try:
                    syn_agent.save(cache_path)
                except Exception:
                    pass

        # Merge all agent patterns into the shared PatternStore.
        for agent_name, agent in self.agents.items():
            patterns = list(getattr(agent, "patterns", None) or [])
            if not patterns:
                continue
            try:
                weights = list(agent.get_weights())
            except Exception:
                continue
            if len(weights) != len(patterns):
                continue
            try:
                self.pattern_store.merge(patterns, weights, agent_name)
            except Exception:
                pass
        self.pattern_store.save()
```

### Step 3.6: Run the full existing test suite to check no regressions
- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/ -v 2>&1 | tail -20`
- Expected: All 76 previously passing tests still pass; total count = 86 (76 + 10 new)

### Step 3.7: Commit
- [ ] Run:
```bash
git add hpm_ai_v6/agents/multi_agent_reader.py
git commit -m "feat: wire PatternStore into MultiAgentReader warm_start and train_sequence"
```

---

## Task 4: Integration test verification

**Files:** No changes.

### Step 4.1: Run the integration test
- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_pattern_store.py::test_two_train_sequences_weight_grows -v`
- Expected: PASS

---

## Task 5: Full suite verification

**Files:** None changed — verification only.

### Step 5.1: Run the complete test suite
- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/ -v 2>&1 | tail -10`
- Expected:
  ```
  ========== 86 passed in X.XXs ==========
  ```

### Step 5.2: If any test fails, diagnose before fixing

Common failure modes:
- **`warm_start` not defined on an agent** — the `warm_start_from_cache` change calls `agent.warm_start()` inside a `try/except Exception: pass`, so this is silently skipped. Confirm the exception is not re-raised.
- **Embedding dimension mismatch** — if agents use different embedding sizes, `merge()` raises `ValueError`. The per-agent `merge` call in Step 3.5 wraps this in `try/except Exception: pass`, so it is skipped. If you see failures, confirm that wrapper is in place.
- **`get_weights()` not on an agent** — also guarded with `try/except` in Step 3.5.

### Step 5.3: Commit only if fixes were needed
- [ ] Run (only if Step 5.1 required edits):
```bash
git add hpm_ai_v6/agents/multi_agent_reader.py
git commit -m "fix: guard PatternStore merge against missing agent interfaces"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Covered by |
|---|---|
| `PatternStore.__init__(cache_dir, similarity_threshold)` | Task 1 Step 1.4 |
| `merge()` with running average and cosine >= 0.85 | Task 1 Step 1.4; Task 2 |
| `load()` returning `([], [])` on missing file | `test_load_missing_file_returns_empty` |
| `save()` creating directory and writing `.npz` | `test_save_load_roundtrip` |
| `find_similar(vector, top_k)` sorted descending | Task 1 Step 1.4; `test_find_similar_top_k_sorted_descending` |
| `clear()` resets in-memory only | `test_clear_resets_in_memory_state` |
| Storage arrays: vectors, weights, counts, names, agents | Task 1 Step 1.4 (all 5 arrays) |
| `allow_pickle=True` with code comment explaining safety | Task 1 Step 1.4 (module docstring + inline comments) |
| agents field accumulates distinct names | `test_agents_field_accumulates` |
| Embedding dimension mismatch raises `ValueError` | Task 1 Step 1.4 (`raise ValueError(...)`) |
| `MultiAgentReader.__init__` creates `PatternStore` | Task 3 Step 3.3 |
| `warm_start_from_cache` calls `pattern_store.load()` | Task 3 Step 3.4 |
| `train_sequence` merges agents + calls `pattern_store.save()` | Task 3 Step 3.5 |
| 76 existing tests must still pass | Task 5 |

No gaps found.

### Placeholder scan

No "TBD", "TODO", "similar to Task N", or vague "add error handling" text. All code steps contain complete, runnable code.

### Type consistency check

- `PatternStore.merge(patterns: List[Cell], weights: List[float], agent_name: str)` — identical signature in skeleton and all test calls.
- `PatternStore.load() -> Tuple[List[Cell], List[float]]` — return type consistent: `patterns[0].name` (str from Cell) and `weights[0]` (float) in every test.
- `PatternStore.find_similar(vector: np.ndarray, top_k: int) -> List[Tuple[float, Cell]]` — `results[0][0]` is float, `results[0][1].name` is str, consistent with skeleton return.
- `store._counts[0]` — `_counts` is `np.ndarray` of int32, defined in `_init_empty()` and `load()`.
- `store._vectors` — after `clear()`, `_init_empty()` sets `_vectors = np.empty((0, 0), ...)`, so `len(store._vectors) == 0` is True. The test assertion `store._vectors is None or len(store._vectors) == 0` passes.
- `store._agents[0]` — object-dtype array; `.split(",")` is valid on the Python str stored there.

All types and field names are consistent across all 5 tasks.
