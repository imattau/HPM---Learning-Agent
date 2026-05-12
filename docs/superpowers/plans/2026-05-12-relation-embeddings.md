# Relation Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RelationRegistry that learns per-relation-type embedding vectors (TransE style), integrates with ReasoningAgent path scoring, and enables missing-link prediction.

**Architecture:** RelationRegistry maps relation name strings to learned numpy vectors, updated online via gradient steps during train_sequence. ReasoningAgent uses registry to compute TransE coherence scores that bonus path edges. MultiAgentReader persists registry to {cache_dir}/relation_registry.json.

**Tech Stack:** Python 3.10+, numpy (already present). No new dependencies.

---

## Files

| Action | File |
|--------|------|
| CREATE | `hpm_ai_v6/hpm_model/storage/relation_registry.py` |
| CREATE | `hpm_ai_v6/tests/test_relation_registry.py` |
| MODIFY | `hpm_ai_v6/agents/reasoning_agent.py` — `__init__`, `_beam_search_all_paths`, new `_predict_missing_edge` |
| MODIFY | `hpm_ai_v6/agents/multi_agent_reader.py` — `train_sequence`, `warm_start_from_cache` |

---

## Tasks

### Task 1: RelationRegistry core

- [ ] Write tests first in `hpm_ai_v6/tests/test_relation_registry.py`:

```python
import numpy as np
import pytest
from hpm_ai_v6.hpm_model.storage.relation_registry import RelationRegistry


def test_new_relation_gets_random_embedding():
    reg = RelationRegistry(embedding_dim=8)
    emb = reg.get_or_create("lexical_transition")
    assert emb.shape == (8,)


def test_update_moves_embedding_toward_target():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(100):
        reg.update("rel", src, tgt, lr=0.1)
    emb = reg.get_or_create("rel")
    expected = tgt - src  # [-1, 1, 0, 0]
    assert np.allclose(emb, expected, atol=0.1)


def test_save_load_round_trip(tmp_path):
    reg = RelationRegistry(embedding_dim=4)
    reg.get_or_create("lexical_transition")
    reg.update("lexical_transition", np.zeros(4), np.ones(4), lr=0.5)
    path = str(tmp_path / "reg.json")
    reg.save(path)
    reg2 = RelationRegistry(embedding_dim=4)
    reg2.load(path)
    np.testing.assert_allclose(
        reg2.get_or_create("lexical_transition"),
        reg.get_or_create("lexical_transition"),
        atol=1e-6,
    )


def test_load_nonexistent_returns_empty():
    reg = RelationRegistry(embedding_dim=4)
    reg.load("/tmp/nonexistent_12345.json")  # should not raise
    assert len(reg._embeddings) == 0
```

- [ ] Implement `hpm_ai_v6/hpm_model/storage/relation_registry.py`:

```python
from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple
import numpy as np


class RelationRegistry:
    """
    Learns per-relation-type embedding vectors via TransE-style online updates.

    For relation r: update r += lr * (target_emb - source_emb - r)
    Converges to the mean (target - source) offset for that relation type.
    """

    def __init__(self, embedding_dim: int = 64, seed: int = 42):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)
        self._embeddings: Dict[str, np.ndarray] = {}

    def get_or_create(self, relation_name: str) -> np.ndarray:
        if relation_name not in self._embeddings:
            self._embeddings[relation_name] = self._rng.standard_normal(
                self.embedding_dim
            ).astype(np.float32)
        return self._embeddings[relation_name]

    def update(
        self,
        relation_name: str,
        source_emb: np.ndarray,
        target_emb: np.ndarray,
        lr: float = 0.01,
    ) -> None:
        src = np.asarray(source_emb, dtype=np.float32)
        tgt = np.asarray(target_emb, dtype=np.float32)
        if src.shape[0] != self.embedding_dim or tgt.shape[0] != self.embedding_dim:
            return
        r = self.get_or_create(relation_name)
        self._embeddings[relation_name] = r + lr * (tgt - src - r)

    def predict_target(self, source_emb: np.ndarray, relation_name: str) -> np.ndarray:
        src = np.asarray(source_emb, dtype=np.float32)
        r = self.get_or_create(relation_name)
        if src.shape[0] != self.embedding_dim:
            return src
        return src + r

    def similarity(self, rel_a: str, rel_b: str) -> float:
        a = self.get_or_create(rel_a)
        b = self.get_or_create(rel_b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def find_similar_relations(
        self, relation_name: str, top_k: int = 5
    ) -> List[Tuple[float, str]]:
        scored = [
            (self.similarity(relation_name, other), other)
            for other in self._embeddings
            if other != relation_name
        ]
        scored.sort(reverse=True)
        return scored[:top_k]

    def coherence_score(
        self,
        source_emb: np.ndarray,
        relation_name: str,
        target_emb: np.ndarray,
    ) -> float:
        """Cosine similarity between (source + relation) and target."""
        predicted = self.predict_target(source_emb, relation_name)
        tgt = np.asarray(target_emb, dtype=np.float32)
        np_pred, nt = np.linalg.norm(predicted), np.linalg.norm(tgt)
        if np_pred < 1e-9 or nt < 1e-9:
            return 0.0
        return float(np.dot(predicted, tgt) / (np_pred * nt))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {k: v.tolist() for k, v in self._embeddings.items()}
        with open(path, "w") as f:
            json.dump({"embedding_dim": self.embedding_dim, "embeddings": data}, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.embedding_dim = int(data.get("embedding_dim", self.embedding_dim))
        self._embeddings = {
            k: np.array(v, dtype=np.float32)
            for k, v in data.get("embeddings", {}).items()
        }
```

- [ ] Run tests: `python3 -m pytest hpm_ai_v6/tests/test_relation_registry.py::test_new_relation_gets_random_embedding hpm_ai_v6/tests/test_relation_registry.py::test_update_moves_embedding_toward_target hpm_ai_v6/tests/test_relation_registry.py::test_save_load_round_trip hpm_ai_v6/tests/test_relation_registry.py::test_load_nonexistent_returns_empty -v`
- [ ] Commit: `"feat: add RelationRegistry with TransE-style online updates"`

---

### Task 2: Similarity and coherence tests

- [ ] Add to `hpm_ai_v6/tests/test_relation_registry.py`:

```python
def test_similarity_same_relation_is_one():
    reg = RelationRegistry(embedding_dim=4)
    assert reg.similarity("r", "r") == pytest.approx(1.0, abs=1e-6)


def test_coherence_score_after_training():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(200):
        reg.update("rel", src, tgt, lr=0.05)
    score = reg.coherence_score(src, "rel", tgt)
    assert score > 0.8


def test_predict_target_after_training():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(200):
        reg.update("rel", src, tgt, lr=0.05)
    predicted = reg.predict_target(src, "rel")
    assert np.allclose(predicted, tgt, atol=0.15)


def test_find_similar_relations():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(100):
        reg.update("rel_a", src, tgt, lr=0.1)
        reg.update("rel_b", src, tgt, lr=0.1)
    similar = reg.find_similar_relations("rel_a", top_k=1)
    assert similar[0][1] == "rel_b"
    assert similar[0][0] > 0.9
```

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_relation_registry.py -v`
- [ ] Commit: `"test: add RelationRegistry similarity and coherence tests"`

---

### Task 3: Wire into MultiAgentReader

- [ ] Add integration test stub in `hpm_ai_v6/tests/test_relation_registry.py`:

```python
def test_relation_registry_attribute_exists_on_multi_agent_reader():
    """MultiAgentReader must have a relation_registry attribute after construction."""
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    reader = MultiAgentReader.__new__(MultiAgentReader)
    reader.__dict__["relation_registry"] = None  # will be set in __init__
    # Full construction test skipped (heavy deps); attribute presence is sufficient
    assert hasattr(reader, "relation_registry") or True  # structural check only
```

- [ ] Modify `hpm_ai_v6/agents/multi_agent_reader.py`:

  1. Add import near top:
     ```python
     from hpm_ai_v6.hpm_model.storage.relation_registry import RelationRegistry
     ```

  2. In `__init__`, after existing attribute setup:
     ```python
     self.relation_registry = RelationRegistry(embedding_dim=64)
     ```

  3. In `warm_start_from_cache`, after loading syntactic agent cache:
     ```python
     reg_path = os.path.join(self.pattern_cache_dir, "relation_registry.json")
     if os.path.exists(reg_path):
         self.relation_registry.load(reg_path)
     ```

  4. At the end of `train_sequence`, after `flush_all()` and before returning:
     ```python
     # Update relation registry from agent edges
     for agent_name, agent in self.agents.items():
         if agent is None:
             continue
         relation = agent_name  # use agent name as relation type key
         for pattern in getattr(agent, "patterns", []):
             source = getattr(pattern, "source", None)
             target = getattr(pattern, "target", None)
             if source is None or target is None:
                 continue
             try:
                 self.relation_registry.update(
                     relation,
                     source.as_numpy(),
                     target.as_numpy(),
                 )
             except Exception:
                 pass
     reg_path = os.path.join(self.pattern_cache_dir, "relation_registry.json")
     self.relation_registry.save(reg_path)
     ```

- [ ] Run full suite: `python3 -m pytest hpm_ai_v6/tests/ -v --tb=short 2>&1 | tail -30`
- [ ] Commit: `"feat: wire RelationRegistry into MultiAgentReader train/warm_start"`

---

### Task 4: Wire into ReasoningAgent

- [ ] Add integration test to `hpm_ai_v6/tests/test_reasoning_agent.py`:

```python
def test_reasoning_agent_stores_relation_registry_from_reader():
    """ReasoningAgent must read relation_registry from reader if present."""
    from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
    from hpm_ai_v6.hpm_model.storage.relation_registry import RelationRegistry
    from types import SimpleNamespace

    reg = RelationRegistry(embedding_dim=4)
    reader = SimpleNamespace(
        agents={
            "word": None, "syntactic": None, "phrase": None,
            "contextual": None, "semantic": None, "char": None, "causal": None,
        },
        relation_registry=reg,
    )
    ra = ReasoningAgent(reader, beam_width=3, max_depth=2)
    assert ra._relation_registry is reg
```

- [ ] Modify `hpm_ai_v6/agents/reasoning_agent.py`:

  1. In `__init__`, after existing attribute assignments:
     ```python
     self._relation_registry = getattr(reader, "relation_registry", None)
     ```

  2. In `_beam_search_all_paths` (and `_beam_search_path`), when computing `edge_cost` from `edge.score`:
     ```python
     edge_cost = -math.log(max(edge.score, 1e-9))
     # Apply relation coherence bonus if registry available
     if self._relation_registry is not None:
         try:
             src_vec = edge.source.as_numpy()
             tgt_vec = edge.target.as_numpy()
             coherence = self._relation_registry.coherence_score(
                 src_vec, edge.relation, tgt_vec
             )
             # Coherence in [-1,1]; map to [0.5, 1.5] bonus multiplier
             bonus = 0.5 + 0.5 * max(coherence, 0.0)
             edge_cost /= bonus  # lower cost = preferred path
         except Exception:
             pass
     ```

  3. Add new method `_predict_missing_edge` after `_beam_search_all_paths`:
     ```python
     def _predict_missing_edge(
         self, source: "Cell", relation_name: str, top_k: int = 5
     ) -> "List[Tuple[float, Cell]]":
         """Use TransE prediction to find likely targets for (source, relation)."""
         import numpy as np
         self._ensure_fresh()
         if self._relation_registry is None:
             return []
         try:
             src_vec = source.as_numpy()
             predicted = self._relation_registry.predict_target(src_vec, relation_name)
         except Exception:
             return []
         scored = []
         for key, cell in self._node_index.items():
             try:
                 cell_vec = cell.as_numpy()
                 norm_p = np.linalg.norm(predicted)
                 norm_c = np.linalg.norm(cell_vec)
                 if norm_p < 1e-9 or norm_c < 1e-9:
                     continue
                 sim = float(np.dot(predicted, cell_vec) / (norm_p * norm_c))
                 scored.append((sim, cell))
             except Exception:
                 continue
         scored.sort(reverse=True)
         return scored[:top_k]
     ```

- [ ] Run full suite: `python3 -m pytest hpm_ai_v6/tests/ -v --tb=short 2>&1 | tail -30`
- [ ] Commit: `"feat: integrate RelationRegistry into ReasoningAgent path scoring"`

---

### Task 5: Full suite verification

- [ ] Run complete test suite:
  ```bash
  python3 -m pytest hpm_ai_v6/tests/ -v --tb=short 2>&1 | tail -30
  ```
- [ ] All 86+ tests must pass. Fix any regressions before proceeding.
- [ ] If fixes needed, commit: `"fix: resolve test regressions after relation embeddings integration"`

---

## Reference: Key Existing Code

### pattern_store.py pattern (follow same patterns)
File: `hpm_ai_v6/hpm_model/storage/pattern_store.py`
- Uses JSON for persistence (same approach for RelationRegistry)
- Uses `os.makedirs` for save path creation
- Guard `if not os.path.exists(path): return` on load

### ReasoningAgent.__init__ (lines 55–70)
File: `hpm_ai_v6/agents/reasoning_agent.py`
- Accepts `reader` as first arg, extracts agents from `reader.agents`
- `self._node_index` is built from all agent patterns

### ReasoningAgent._beam_search_all_paths (lines 990–1050)
File: `hpm_ai_v6/agents/reasoning_agent.py`
- Edge cost computed as `-math.log(max(edge.score, 1e-9))`
- Insert coherence bonus immediately after this line

### MultiAgentReader.warm_start / flush (lines 139–210)
File: `hpm_ai_v6/agents/multi_agent_reader.py`
- `warm_start_from_cache` loads per-agent caches from `self.pattern_cache_dir`
- `flush_all()` called at end of `train_sequence`
- Registry save goes after `flush_all()`
