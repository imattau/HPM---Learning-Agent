# Relation-Polygraph Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate relation embeddings from the standalone RelationRegistry JSON store into the pattern polygraph as dim-2 cells, making relation patterns subject to the same replicator dynamics, pager persistence, and ReasoningAgent indexing as all other patterns.

**Architecture:** A new RelationPatternEmitter accumulates (source, relation, target) observations and emits dim-2 Cells whose embedding = running-mean of (target-source) offsets. MultiAgentReader feeds edges through the emitter during train_sequence; the resulting dim-2 cells enter the word agent's pager for persistence. RelationRegistry becomes a thin in-memory cache populated from these dim-2 cells. ReasoningAgent's existing _build_explicit_analogy_index picks up relation cells automatically.

**Tech Stack:** Python 3.10+, numpy. No new dependencies.

## File Structure

- CREATE: `hpm_ai_v6/hpm_model/storage/relation_pattern_emitter.py`
- CREATE: `hpm_ai_v6/tests/test_relation_pattern_emitter.py`
- MODIFY: `hpm_ai_v6/hpm_model/storage/relation_registry.py` (add populate_from_cells, to_cells; remove save/load JSON)
- MODIFY: `hpm_ai_v6/agents/multi_agent_reader.py` (use emitter, update warm_start, remove JSON persistence)
- MODIFY: `hpm_ai_v6/agents/reasoning_agent.py` (coherence scoring via dim-2 cells, not registry)
- MODIFY: `hpm_ai_v6/tests/test_relation_registry.py` (remove save/load tests, add populate_from_cells tests)

## Tasks

### Task 1: RelationPatternEmitter

- [ ] Write tests first in `hpm_ai_v6/tests/test_relation_pattern_emitter.py`:

```python
from hpm_ai_v6.hpm_model.storage.relation_pattern_emitter import RelationPatternEmitter
from hpm_ai_v6.hpm_model.core.cell import Cell
import numpy as np
import pytest

def make_cell(name, emb):
    return Cell(name=name, dim=0, embedding=emb)

def test_observe_returns_dim2_cell():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_cat", [1,0,0,0])
    tgt = make_cell("word_sat", [0,1,0,0])
    rel_cell = emitter.observe(src, "lexical_transition", tgt)
    assert rel_cell.dim == 2

def test_observe_name_is_rel_prefixed():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_cat", [1,0,0,0])
    tgt = make_cell("word_sat", [0,1,0,0])
    rel_cell = emitter.observe(src, "lexical_transition", tgt)
    assert rel_cell.name == "rel_lexical_transition"

def test_observe_embedding_converges_to_mean_offset():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_cat", [1,0,0,0])
    tgt = make_cell("word_sat", [0,1,0,0])
    # After many observations of same offset, embedding should be close to tgt-src
    for _ in range(50):
        emitter.observe(src, "lexical_transition", tgt)
    rel_cell = emitter.observe(src, "lexical_transition", tgt)
    expected = np.array([0,1,0,0]) - np.array([1,0,0,0])  # [-1,1,0,0]
    np.testing.assert_allclose(rel_cell.as_numpy(), expected, atol=0.1)

def test_same_relation_returns_same_cell_object():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1,0,0,0])
    tgt = make_cell("word_b", [0,1,0,0])
    cell1 = emitter.observe(src, "lexical_transition", tgt)
    cell2 = emitter.observe(src, "lexical_transition", tgt)
    assert cell1.name == cell2.name

def test_different_relations_produce_different_cells():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1,0,0,0])
    tgt = make_cell("word_b", [0,1,0,0])
    cell1 = emitter.observe(src, "lexical_transition", tgt)
    cell2 = emitter.observe(src, "causal_relation", tgt)
    assert cell1.name != cell2.name

def test_get_relation_cells_returns_all_observed():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1,0,0,0])
    tgt = make_cell("word_b", [0,1,0,0])
    emitter.observe(src, "lexical_transition", tgt)
    emitter.observe(src, "causal_relation", tgt)
    cells = emitter.get_relation_cells()
    names = [c.name for c, _ in cells]
    assert "rel_lexical_transition" in names
    assert "rel_causal_relation" in names

def test_get_relation_cells_weight_increases_with_observations():
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = make_cell("word_a", [1,0,0,0])
    tgt = make_cell("word_b", [0,1,0,0])
    for _ in range(5):
        emitter.observe(src, "lexical_transition", tgt)
    cells = emitter.get_relation_cells()
    weight = next(w for c, w in cells if c.name == "rel_lexical_transition")
    assert weight == pytest.approx(5.0)
```

- [ ] Implement `hpm_ai_v6/hpm_model/storage/relation_pattern_emitter.py`:

```python
# hpm_ai_v6/hpm_model/storage/relation_pattern_emitter.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell


class RelationPatternEmitter:
    """
    Accumulates (source, relation_name, target) observations and emits
    dim-2 Cells whose embedding converges to the mean (target-source) offset
    for each relation type.

    These dim-2 cells enter the pattern polygraph like any other analogy cell,
    making relation patterns subject to replicator dynamics and pager persistence.
    """

    def __init__(self, embedding_dim: int = 64, lr: float = 0.05):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self._cells: Dict[str, Cell] = {}
        self._counts: Dict[str, int] = {}

    def observe(self, source: Cell, relation_name: str, target: Cell) -> Cell:
        src = np.asarray(source.as_numpy(), dtype=np.float32)
        tgt = np.asarray(target.as_numpy(), dtype=np.float32)
        if src.shape[0] != self.embedding_dim or tgt.shape[0] != self.embedding_dim:
            if relation_name in self._cells:
                return self._cells[relation_name]
            zero_emb = np.zeros(self.embedding_dim, dtype=np.float32)
            cell = Cell(name=f"rel_{relation_name}", dim=2, embedding=zero_emb.tolist())
            self._cells[relation_name] = cell
            self._counts[relation_name] = 0
            return cell

        offset = tgt - src
        if relation_name not in self._cells:
            cell = Cell(
                name=f"rel_{relation_name}",
                dim=2,
                embedding=offset.tolist(),
            )
            self._cells[relation_name] = cell
            self._counts[relation_name] = 1
        else:
            old_emb = np.asarray(self._cells[relation_name].as_numpy(), dtype=np.float32)
            new_emb = old_emb + self.lr * (offset - old_emb)
            self._cells[relation_name] = Cell(
                name=f"rel_{relation_name}",
                dim=2,
                embedding=new_emb.tolist(),
            )
            self._counts[relation_name] += 1

        return self._cells[relation_name]

    def get_relation_cells(self) -> List[Tuple[Cell, float]]:
        return [
            (cell, float(self._counts[rel_name]))
            for rel_name, cell in self._cells.items()
        ]
```

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_relation_pattern_emitter.py -v`
- [ ] Expected: all 7 tests PASS
- [ ] Commit: `"feat: add RelationPatternEmitter — relation embeddings as dim-2 cells"`

---

### Task 2: Update RelationRegistry (remove JSON I/O, add cell bridge)

- [ ] Update `hpm_ai_v6/tests/test_relation_registry.py` — remove tests for save/load JSON, add:

```python
def test_populate_from_cells():
    from hpm_ai_v6.hpm_model.storage.relation_pattern_emitter import RelationPatternEmitter
    emitter = RelationPatternEmitter(embedding_dim=4)
    src = Cell(name="word_a", dim=0, embedding=[1,0,0,0])
    tgt = Cell(name="word_b", dim=0, embedding=[0,1,0,0])
    for _ in range(20):
        emitter.observe(src, "lexical_transition", tgt)
    cells = [c for c, _ in emitter.get_relation_cells()]

    reg = RelationRegistry(embedding_dim=4)
    reg.populate_from_cells(cells)
    emb = reg.get_or_create("lexical_transition")
    assert emb.shape == (4,)
    # embedding should reflect the observed offset
    expected = np.array([0,1,0,0]) - np.array([1,0,0,0])
    assert np.allclose(emb, expected, atol=0.2)

def test_to_cells_returns_dim2_cells():
    reg = RelationRegistry(embedding_dim=4)
    reg.get_or_create("lexical_transition")
    cells = reg.to_cells()
    assert all(c.dim == 2 for c, _ in cells)
    assert all(c.name.startswith("rel_") for c, _ in cells)
```

- [ ] Update `hpm_ai_v6/hpm_model/storage/relation_registry.py`:
  - Add `populate_from_cells(self, cells: List[Cell]) -> None`: for each cell with name starting `"rel_"`, extract relation_name and set `self._embeddings[relation_name] = cell.as_numpy()`
  - Add `to_cells(self) -> List[Tuple[Cell, float]]`: return `[(Cell(name=f"rel_{name}", dim=2, embedding=emb.tolist()), 1.0) for name, emb in self._embeddings.items()]`
  - Remove `save(path)` and `load(path)` methods entirely
  - Keep all other methods unchanged (get_or_create, update, coherence_score, predict_target, similarity, find_similar_relations)

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_relation_registry.py -v`
- [ ] Commit: `"refactor: RelationRegistry — remove JSON I/O, add populate_from_cells/to_cells"`

---

### Task 3: Wire emitter into MultiAgentReader

- [ ] Add to `hpm_ai_v6/agents/multi_agent_reader.py`:

  1. Import: `from hpm_ai_v6.hpm_model.storage.relation_pattern_emitter import RelationPatternEmitter`

  2. In `__init__` (after relation_registry): `self.relation_emitter = RelationPatternEmitter(embedding_dim=64)`

  3. In `warm_start_from_cache` (remove relation_registry.json load, replace with):

```python
# Hydrate relation registry from dim-2 cells in word agent's pager
rel_cells = [
    p for p in getattr(self.word_agent, "patterns", [])
    if getattr(p, "dim", 0) == 2 and getattr(p, "name", "").startswith("rel_")
]
if rel_cells:
    self.relation_registry.populate_from_cells(rel_cells)
```

  4. In `train_sequence` (replace relation_registry update + save with emitter):

```python
# Feed edges through emitter to produce dim-2 relation cells
agent_relation_map = {
    "word": "lexical_transition", "phrase": "syntactic_transition",
    "semantic": "semantic_transition", "contextual": "contextual_prediction",
    "char": "character_transition", "causal": "causal_relation",
}
for agent_name, relation_name in agent_relation_map.items():
    agent = self.agents.get(agent_name)
    if agent is None:
        continue
    for pattern in getattr(agent, "patterns", []):
        src = getattr(pattern, "source", None)
        tgt = getattr(pattern, "target", None)
        if src is None or tgt is None:
            continue
        try:
            self.relation_emitter.observe(src, relation_name, tgt)
        except Exception:
            pass

# Store relation cells in word_agent's pattern list for pager persistence
for rel_cell, _ in self.relation_emitter.get_relation_cells():
    if rel_cell not in self.word_agent.patterns:
        self.word_agent.patterns.append(rel_cell)

# Update registry from emitter
rel_cells = [c for c, _ in self.relation_emitter.get_relation_cells()]
self.relation_registry.populate_from_cells(rel_cells)
# Remove old JSON save
```

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/ -x -q 2>&1 | tail -10`
- [ ] Commit: `"feat: wire RelationPatternEmitter into MultiAgentReader train/warm_start"`

---

### Task 4: Update ReasoningAgent coherence scoring to use dim-2 cells

- [ ] In `hpm_ai_v6/agents/reasoning_agent.py`:

  - Add `_relation_cell_index: Dict[str, Cell] = {}` to `__init__`

  - In `refresh()`, after `_build_explicit_analogy_index`:

```python
self._relation_cell_index = {
    pattern.name.removeprefix("rel_"): pattern
    for pattern in all_patterns
    if getattr(pattern, "dim", 0) == 2
    and getattr(pattern, "name", "").startswith("rel_")
}
```

  - Update coherence scoring in `_beam_search_path` and `_beam_search_all_paths` (replace registry-based coherence):

```python
rel_cell = self._relation_cell_index.get(edge.relation)
if rel_cell is not None:
    try:
        rel_emb = rel_cell.as_numpy()
        src_emb = edge.source.as_numpy()
        tgt_emb = edge.target.as_numpy()
        predicted = src_emb + rel_emb
        np_p = np.linalg.norm(predicted)
        nt = np.linalg.norm(tgt_emb)
        if np_p > 1e-9 and nt > 1e-9:
            coherence = float(np.dot(predicted, tgt_emb) / (np_p * nt))
            bonus = 0.5 + 0.5 * max(coherence, 0.0)
            edge_cost /= bonus
    except Exception:
        pass
```

  - Update `_predict_missing_edge` to use `_relation_cell_index` instead of `_relation_registry`

- [ ] Add to `hpm_ai_v6/tests/test_reasoning_agent.py`:

```python
def test_relation_cell_index_populated_after_refresh():
    from types import SimpleNamespace
    from hpm_ai_v6.hpm_model.core.cell import Cell
    # Create a dim-2 rel cell and put it in a stub agent
    rel_cell = Cell(name="rel_lexical_transition", dim=2, embedding=[0.1, 0.2, 0.3])
    stub = StubAgent(patterns=[rel_cell], weights=[1.0], lookup={})
    reader = SimpleNamespace(agents={"word": stub, "phrase": None, "contextual": None,
                                      "semantic": None, "char": None, "causal": None})
    ra = ReasoningAgent(reader)
    ra.refresh()
    assert "lexical_transition" in ra._relation_cell_index
```

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/ -x -q 2>&1 | tail -10`
- [ ] Commit: `"feat: ReasoningAgent uses dim-2 relation cells for coherence scoring"`

---

### Task 5: Remove relation_registry.json references

- [ ] Verify no remaining references:

```bash
grep -r "relation_registry.json" hpm_ai_v6/
```

- [ ] Expected: no output. If any found, remove them.
- [ ] Run full suite: `python3 -m pytest hpm_ai_v6/tests/ -v --tb=short 2>&1 | tail -20`
- [ ] All 96+ tests must pass
- [ ] Commit: `"chore: remove relation_registry.json persistence — relation patterns now in polygraph"`

---

### Task 6: Delete stale relation_registry.json cache files

- [ ] Run:

```bash
find . -name "relation_registry.json" -delete && echo "cleaned"
```

- [ ] Final commit: `"chore: clean up stale relation_registry.json cache files"`
