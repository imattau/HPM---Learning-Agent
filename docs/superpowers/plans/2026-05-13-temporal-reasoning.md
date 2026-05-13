# Temporal Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add temporal reasoning to the HPM agent via a new TemporalCell (3-cell), TemporalAgent, and three query methods on ReasoningAgent.

**Architecture:** `TemporalAgent` builds `TemporalCell`s from `CausalAgent.patterns` during `MultiAgentReader.maintenance_cycle()`. `ReasoningAgent` gains `temporal_sequence()`, `temporal_between()`, and `temporal_overlap()` methods, plus two new intents in `reason_with_trace()`.

**Tech Stack:** Python 3.10+, dataclasses, existing `Cell` base class (`hpm_ai_v6/hpm_model/core/cell.py`), pytest.

**Spec:** `docs/superpoors/specs/2026-05-13-temporal-reasoning-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `hpm_ai_v6/hpm_model/core/temporal_cell.py` | `TemporalCell` dataclass |
| Create | `hpm_ai_v6/agents/temporal_agent.py` | Build/update temporal cells from causal patterns |
| Modify | `hpm_ai_v6/agents/multi_agent_reader.py` | Instantiate TemporalAgent, call in maintenance_cycle |
| Modify | `hpm_ai_v6/agents/reasoning_agent.py` | Add temporal index access + 3 query methods + intent parsing |
| Create | `hpm_ai_v6/tests/test_temporal_agent.py` | Unit tests for TemporalAgent |
| Create | `hpm_ai_v6/tests/test_temporal_reasoning.py` | Integration tests for ReasoningAgent temporal queries |

---

## Task 1: TemporalCell dataclass

**Files:**
- Create: `hpm_ai_v6/hpm_model/core/temporal_cell.py`
- Test: `hpm_ai_v6/tests/test_temporal_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# hpm_ai_v6/tests/test_temporal_agent.py
import numpy as np
import pytest
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.core.temporal_cell import TemporalCell

def make_cell(name: str) -> Cell:
    return Cell(name=name, dim=1, weight=1.0, embedding=np.zeros(8))

def test_temporal_cell_fields():
    cause = make_cell("word:rain")
    effect = make_cell("word:flood")
    tc = TemporalCell(
        name="temporal:rain->flood",
        dim=3,
        weight=0.8,
        embedding=np.zeros(8),
        cause=cause,
        effect=effect,
        onset_weight=0.8,
        duration_weight=1.0,
    )
    assert tc.cause is cause
    assert tc.effect is effect
    assert tc.onset_weight == 0.8
    assert tc.duration_weight == 1.0
    assert tc.lapsed is False
    assert tc.concurrent == []

def test_temporal_cell_lapsed_default_false():
    cause = make_cell("word:a")
    effect = make_cell("word:b")
    tc = TemporalCell(
        name="temporal:a->b", dim=3, weight=0.5, embedding=np.zeros(8),
        cause=cause, effect=effect, onset_weight=0.5, duration_weight=1.0,
    )
    assert not tc.lapsed
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py::test_temporal_cell_fields -v
```
Expected: `ModuleNotFoundError: No module named 'hpm_ai_v6.hpm_model.core.temporal_cell'`

- [ ] **Step 3: Implement TemporalCell**

```python
# hpm_ai_v6/hpm_model/core/temporal_cell.py
from __future__ import annotations
from typing import List
from hpm_ai_v6.hpm_model.core.cell import Cell


class TemporalCell(Cell):
    cause: Cell
    effect: Cell
    onset_weight: float
    duration_weight: float
    lapsed: bool = False
    concurrent: List["TemporalCell"] = []

    class Config:
        arbitrary_types_allowed = True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py::test_temporal_cell_fields hpm_ai_v6/tests/test_temporal_agent.py::test_temporal_cell_lapsed_default_false -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/hpm_model/core/temporal_cell.py hpm_ai_v6/tests/test_temporal_agent.py
git commit -m "feat: add TemporalCell (3-cell) for HPM temporal interval layer"
```

---

## Task 2: TemporalAgent — segment and spanning cell construction

**Files:**
- Create: `hpm_ai_v6/agents/temporal_agent.py`
- Test: `hpm_ai_v6/tests/test_temporal_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to hpm_ai_v6/tests/test_temporal_agent.py
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.agents.temporal_agent import TemporalAgent

def make_causal_cell(src_name: str, tgt_name: str, weight: float = 0.7) -> Cell:
    """Simulates a CausalAgent pattern cell with metadata in name."""
    emb = np.zeros(8)
    c = Cell(
        name=f"causal:{src_name}->{tgt_name}",
        dim=1,
        weight=weight,
        embedding=emb,
    )
    # TemporalAgent reads source/target from cell.name split on '->'
    return c

def test_segment_cells_from_single_causal_pattern():
    agent = TemporalAgent()
    patterns = [make_causal_cell("rain", "flood", weight=0.9)]
    cells = agent.build_temporal_cells(patterns)
    assert len(cells) == 1
    assert cells[0].cause.name == "word:rain"
    assert cells[0].effect.name == "word:flood"
    assert cells[0].duration_weight == 1.0
    assert cells[0].onset_weight == pytest.approx(0.9)

def test_spanning_cell_from_chain():
    agent = TemporalAgent()
    patterns = [
        make_causal_cell("rain", "flood", weight=0.9),
        make_causal_cell("flood", "damage", weight=0.7),
    ]
    cells = agent.build_temporal_cells(patterns)
    names = [(c.cause.name, c.effect.name) for c in cells]
    assert ("word:rain", "word:flood") in names
    assert ("word:flood", "word:damage") in names
    assert ("word:rain", "word:damage") in names  # spanning

def test_spanning_cell_duration_weight():
    agent = TemporalAgent()
    patterns = [
        make_causal_cell("a", "b", weight=0.8),
        make_causal_cell("b", "c", weight=0.6),
    ]
    cells = agent.build_temporal_cells(patterns)
    spanning = next(c for c in cells if c.cause.name == "word:a" and c.effect.name == "word:c")
    assert spanning.duration_weight == 2.0
    assert spanning.onset_weight == pytest.approx(min(0.8, 0.6))
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py -k "segment or spanning" -v
```
Expected: `ModuleNotFoundError: No module named 'hpm_ai_v6.agents.temporal_agent'`

- [ ] **Step 3: Implement TemporalAgent build_temporal_cells**

```python
# hpm_ai_v6/agents/temporal_agent.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.core.temporal_cell import TemporalCell


class TemporalAgent:
    def __init__(self, threshold: float = 0.01, max_depth: int = 4):
        self.threshold = threshold
        self.max_depth = max_depth
        self._temporal_index: Dict[str, List[TemporalCell]] = {}

    @property
    def temporal_index(self) -> Dict[str, List[TemporalCell]]:
        return self._temporal_index

    def _parse_causal_pattern(self, pattern: Cell) -> Optional[Tuple[str, str]]:
        """Extract (source_name, target_name) from causal pattern cell name.
        Expects format: 'causal:src->tgt' or 'src->tgt'."""
        name = pattern.name
        if ":" in name:
            name = name.split(":", 1)[1]
        if "->" not in name:
            return None
        parts = name.split("->", 1)
        return f"word:{parts[0].strip()}", f"word:{parts[1].strip()}"

    def _make_stub_cell(self, name: str) -> Cell:
        return Cell(name=name, dim=0, weight=1.0, embedding=np.zeros(8))

    def _make_temporal_cell(
        self,
        cause: Cell,
        effect: Cell,
        onset_weight: float,
        duration_weight: float,
    ) -> TemporalCell:
        return TemporalCell(
            name=f"temporal:{cause.name}->{effect.name}",
            dim=3,
            weight=onset_weight,
            embedding=np.zeros(8),
            cause=cause,
            effect=effect,
            onset_weight=onset_weight,
            duration_weight=duration_weight,
        )

    def build_temporal_cells(self, causal_patterns: List[Cell]) -> List[TemporalCell]:
        """Build segment and spanning TemporalCells from causal pattern cells."""
        parsed: List[Tuple[str, str, float]] = []
        for p in causal_patterns:
            result = self._parse_causal_pattern(p)
            if result:
                src, tgt = result
                parsed.append((src, tgt, p.weight))

        cells: List[TemporalCell] = []
        stub_cache: Dict[str, Cell] = {}

        def get_stub(name: str) -> Cell:
            if name not in stub_cache:
                stub_cache[name] = self._make_stub_cell(name)
            return stub_cache[name]

        # Segment cells
        for src, tgt, weight in parsed:
            tc = self._make_temporal_cell(get_stub(src), get_stub(tgt), weight, 1.0)
            cells.append(tc)

        # Spanning cells: follow chains up to max_depth
        adj: Dict[str, List[Tuple[str, float]]] = {}
        for src, tgt, weight in parsed:
            adj.setdefault(src, []).append((tgt, weight))

        def follow_chain(start: str, path: List[Tuple[str, float]], depth: int):
            if depth >= self.max_depth:
                return
            for nxt, w in adj.get(path[-1][0], []):
                if any(nxt == p[0] for p in path):
                    continue  # cycle guard
                new_path = path + [(nxt, w)]
                if len(new_path) >= 2:
                    chain_weight = min(w for _, w in new_path)
                    tc = self._make_temporal_cell(
                        get_stub(start), get_stub(nxt),
                        chain_weight, float(len(new_path)),
                    )
                    cells.append(tc)
                follow_chain(start, new_path, depth + 1)

        for src, tgt, weight in parsed:
            follow_chain(src, [(src, weight)], 0)

        # Deduplicate by (cause.name, effect.name), keep highest onset_weight
        seen: Dict[Tuple[str, str], TemporalCell] = {}
        for tc in cells:
            key = (tc.cause.name, tc.effect.name)
            if key not in seen or tc.onset_weight > seen[key].onset_weight:
                seen[key] = tc

        result = list(seen.values())
        self._temporal_index = {}
        for tc in result:
            self._temporal_index.setdefault(tc.cause.name, []).append(tc)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/temporal_agent.py hpm_ai_v6/tests/test_temporal_agent.py
git commit -m "feat: implement TemporalAgent with segment and spanning cell construction"
```

---

## Task 3: Concurrent interval detection

**Files:**
- Modify: `hpm_ai_v6/agents/temporal_agent.py`
- Test: `hpm_ai_v6/tests/test_temporal_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# append to hpm_ai_v6/tests/test_temporal_agent.py
def test_concurrent_detection_shared_cause():
    agent = TemporalAgent()
    patterns = [
        make_causal_cell("rain", "flood"),
        make_causal_cell("rain", "mud"),
    ]
    cells = agent.build_temporal_cells(patterns)
    rain_flood = next(c for c in cells if c.cause.name == "word:rain" and c.effect.name == "word:flood")
    rain_mud = next(c for c in cells if c.cause.name == "word:rain" and c.effect.name == "word:mud")
    assert rain_mud in rain_flood.concurrent
    assert rain_flood in rain_mud.concurrent
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py::test_concurrent_detection_shared_cause -v
```
Expected: FAIL — `concurrent` is empty.

- [ ] **Step 3: Add concurrent detection to build_temporal_cells**

Add this block at the end of `build_temporal_cells`, before the `return result` line:

```python
        # Concurrent detection: shared cause boundary
        by_cause: Dict[str, List[TemporalCell]] = {}
        for tc in result:
            by_cause.setdefault(tc.cause.name, []).append(tc)
        for siblings in by_cause.values():
            if len(siblings) > 1:
                for tc in siblings:
                    tc.concurrent = [s for s in siblings if s is not tc]
```

- [ ] **Step 4: Run all temporal agent tests**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/temporal_agent.py hpm_ai_v6/tests/test_temporal_agent.py
git commit -m "feat: add concurrent interval detection to TemporalAgent"
```

---

## Task 4: Lapsed cell tracking (evolution across maintenance passes)

**Files:**
- Modify: `hpm_ai_v6/agents/temporal_agent.py`
- Test: `hpm_ai_v6/tests/test_temporal_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to hpm_ai_v6/tests/test_temporal_agent.py
def test_lapsed_on_weight_decay():
    agent = TemporalAgent(threshold=0.1)
    patterns_pass1 = [make_causal_cell("rain", "flood", weight=0.8)]
    agent.build_temporal_cells(patterns_pass1)

    # Second pass: weight drops below threshold
    patterns_pass2 = [make_causal_cell("rain", "flood", weight=0.05)]
    agent.update_temporal_cells(patterns_pass2)

    cells = agent.temporal_index.get("word:rain", [])
    rain_flood = next(c for c in cells if c.effect.name == "word:flood")
    assert rain_flood.lapsed is True

def test_update_preserves_lapsed_history():
    agent = TemporalAgent(threshold=0.1)
    agent.build_temporal_cells([make_causal_cell("rain", "flood", weight=0.8)])
    agent.update_temporal_cells([make_causal_cell("rain", "flood", weight=0.05)])
    # Lapsed cell still in index
    cells = agent.temporal_index.get("word:rain", [])
    assert any(c.effect.name == "word:flood" for c in cells)

def test_update_adjusts_onset_weight_ema():
    agent = TemporalAgent(threshold=0.01)
    agent.build_temporal_cells([make_causal_cell("rain", "flood", weight=0.8)])
    agent.update_temporal_cells([make_causal_cell("rain", "flood", weight=0.4)])
    cells = agent.temporal_index.get("word:rain", [])
    rain_flood = next(c for c in cells if c.effect.name == "word:flood")
    # EMA alpha=0.3: 0.8 * 0.7 + 0.4 * 0.3 = 0.68
    assert rain_flood.onset_weight == pytest.approx(0.68, abs=1e-3)
    assert not rain_flood.lapsed
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py -k "lapsed or update" -v
```
Expected: `AttributeError: 'TemporalAgent' object has no attribute 'update_temporal_cells'`

- [ ] **Step 3: Implement update_temporal_cells**

Add this method to `TemporalAgent`:

```python
    def update_temporal_cells(self, causal_patterns: List[Cell], alpha: float = 0.3) -> None:
        """Update existing temporal cells with new causal pattern weights (EMA).
        Marks cells as lapsed if onset_weight drops below threshold.
        New patterns not yet in index are added via build_temporal_cells logic."""
        parsed: Dict[Tuple[str, str], float] = {}
        for p in causal_patterns:
            result = self._parse_causal_pattern(p)
            if result:
                src, tgt = result
                parsed[(src, tgt)] = p.weight

        seen_keys = set()
        for cause_name, cells in self._temporal_index.items():
            for tc in cells:
                key = (tc.cause.name, tc.effect.name)
                if key in parsed:
                    seen_keys.add(key)
                    new_weight = tc.onset_weight * (1 - alpha) + parsed[key] * alpha
                    tc.onset_weight = new_weight
                    tc.weight = new_weight
                    tc.lapsed = new_weight < self.threshold

        # Add new patterns not yet in index
        new_patterns = [
            p for p in causal_patterns
            if self._parse_causal_pattern(p) and
            self._parse_causal_pattern(p) not in seen_keys
        ]
        if new_patterns:
            new_cells = self.build_temporal_cells(new_patterns)
            for tc in new_cells:
                self._temporal_index.setdefault(tc.cause.name, []).append(tc)
```

- [ ] **Step 4: Run all temporal agent tests**

```bash
pytest hpm_ai_v6/tests/test_temporal_agent.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/temporal_agent.py hpm_ai_v6/tests/test_temporal_agent.py
git commit -m "feat: add update_temporal_cells with EMA weight decay and lapsed tracking"
```

---

## Task 5: Integrate TemporalAgent into MultiAgentReader

**Files:**
- Modify: `hpm_ai_v6/agents/multi_agent_reader.py`

- [ ] **Step 1: Add TemporalAgent import and instantiation**

In `hpm_ai_v6/agents/multi_agent_reader.py`, add import near the top with other agent imports:

```python
from hpm_ai_v6.agents.temporal_agent import TemporalAgent
```

In `MultiAgentReader.__init__()`, after the `self.causal_agent` line, add:

```python
self.temporal_agent = TemporalAgent()
```

- [ ] **Step 2: Call TemporalAgent in maintenance_cycle**

In `maintenance_cycle()`, after the `train_sequence` / `train_sequence_active` block completes (around where causal agent is used), add:

```python
        causal_patterns = list(self.causal_agent.patterns)
        if not hasattr(self, '_temporal_agent_initialized'):
            self.temporal_agent.build_temporal_cells(causal_patterns)
            self._temporal_agent_initialized = True
        else:
            self.temporal_agent.update_temporal_cells(causal_patterns)
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
pytest hpm_ai_v6/tests/ -v --tb=short -q
```
Expected: no regressions.

- [ ] **Step 4: Commit**

```bash
git add hpm_ai_v6/agents/multi_agent_reader.py
git commit -m "feat: integrate TemporalAgent into MultiAgentReader maintenance_cycle"
```

---

## Task 6: ReasoningAgent — temporal_sequence and temporal_overlap

**Files:**
- Modify: `hpm_ai_v6/agents/reasoning_agent.py`
- Create: `hpm_ai_v6/tests/test_temporal_reasoning.py`

- [ ] **Step 1: Write the failing tests**

```python
# hpm_ai_v6/tests/test_temporal_reasoning.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.core.temporal_cell import TemporalCell
from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent


def make_cell(name: str, dim: int = 0) -> Cell:
    return Cell(name=name, dim=dim, weight=1.0, embedding=np.zeros(8))


def make_temporal_cell(src: str, tgt: str, onset: float = 0.8, duration: float = 1.0) -> TemporalCell:
    cause = make_cell(f"word:{src}")
    effect = make_cell(f"word:{tgt}")
    return TemporalCell(
        name=f"temporal:word:{src}->word:{tgt}",
        dim=3, weight=onset, embedding=np.zeros(8),
        cause=cause, effect=effect,
        onset_weight=onset, duration_weight=duration,
    )


def make_reader_mock(temporal_index: dict) -> MagicMock:
    reader = MagicMock()
    reader.temporal_agent = MagicMock()
    reader.temporal_agent.temporal_index = temporal_index
    reader.char_agent = MagicMock(); reader.char_agent.patterns = []
    reader.word_agent = MagicMock(); reader.word_agent.patterns = []
    reader.contextual_agent = MagicMock(); reader.contextual_agent.patterns = []
    reader.phrase_agent = MagicMock(); reader.phrase_agent.patterns = []
    reader.semantic_agent = MagicMock(); reader.semantic_agent.patterns = []
    reader.causal_agent = MagicMock(); reader.causal_agent.patterns = []
    reader.dependency_relation_agent = MagicMock(); reader.dependency_relation_agent.patterns = []
    return reader


def test_temporal_sequence_returns_intervals_for_concept():
    tc1 = make_temporal_cell("rain", "flood", onset=0.9, duration=1.0)
    tc2 = make_temporal_cell("rain", "mud", onset=0.7, duration=1.0)
    index = {"word:rain": [tc1, tc2]}
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.temporal_sequence("rain")
    assert tc1 in result
    assert tc2 in result


def test_temporal_sequence_excludes_lapsed_by_default():
    tc1 = make_temporal_cell("rain", "flood")
    tc2 = make_temporal_cell("rain", "mud")
    tc2.lapsed = True
    index = {"word:rain": [tc1, tc2]}
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.temporal_sequence("rain")
    assert tc1 in result
    assert tc2 not in result


def test_temporal_sequence_include_lapsed_flag():
    tc = make_temporal_cell("rain", "mud")
    tc.lapsed = True
    index = {"word:rain": [tc]}
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.temporal_sequence("rain", include_lapsed=True)
    assert tc in result


def test_temporal_overlap_returns_concurrent_intervals():
    tc1 = make_temporal_cell("rain", "flood")
    tc2 = make_temporal_cell("rain", "mud")
    tc1.concurrent = [tc2]
    tc2.concurrent = [tc1]
    index = {"word:rain": [tc1, tc2]}
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.temporal_overlap("rain")
    assert tc2 in result
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest hpm_ai_v6/tests/test_temporal_reasoning.py -v
```
Expected: `AttributeError: 'ReasoningAgent' object has no attribute 'temporal_sequence'`

- [ ] **Step 3: Add temporal_sequence and temporal_overlap to ReasoningAgent**

Add these methods to `ReasoningAgent` (near the other public query methods, around line 2000):

```python
    def _temporal_index(self):
        ta = getattr(self.reader, "temporal_agent", None)
        if ta is None:
            return {}
        return ta.temporal_index

    def temporal_sequence(
        self, concept: str, include_lapsed: bool = False
    ) -> List["TemporalCell"]:
        """All temporal intervals where concept is cause or effect, ordered by duration_weight."""
        from hpm_ai_v6.hpm_model.core.temporal_cell import TemporalCell
        index = self._temporal_index()
        key = concept if concept.startswith("word:") else f"word:{concept}"
        cells = index.get(key, [])
        # Also check as effect
        for intervals in index.values():
            for tc in intervals:
                if tc.effect.name == key and tc not in cells:
                    cells = cells + [tc]
        if not include_lapsed:
            cells = [c for c in cells if not c.lapsed]
        return sorted(cells, key=lambda c: c.duration_weight)

    def temporal_overlap(
        self, concept: str, include_lapsed: bool = False
    ) -> List["TemporalCell"]:
        """All intervals concurrent with any interval involving concept."""
        direct = self.temporal_sequence(concept, include_lapsed=True)
        result = []
        for tc in direct:
            for concurrent in tc.concurrent:
                if concurrent not in result:
                    if include_lapsed or not concurrent.lapsed:
                        result.append(concurrent)
        return result
```

- [ ] **Step 4: Run tests**

```bash
pytest hpm_ai_v6/tests/test_temporal_reasoning.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/reasoning_agent.py hpm_ai_v6/tests/test_temporal_reasoning.py
git commit -m "feat: add temporal_sequence and temporal_overlap to ReasoningAgent"
```

---

## Task 7: ReasoningAgent — temporal_between

**Files:**
- Modify: `hpm_ai_v6/agents/reasoning_agent.py`
- Modify: `hpm_ai_v6/tests/test_temporal_reasoning.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to hpm_ai_v6/tests/test_temporal_reasoning.py
def test_temporal_between_follows_chain():
    tc1 = make_temporal_cell("rain", "flood", onset=0.9)
    tc2 = make_temporal_cell("flood", "damage", onset=0.7)
    index = {
        "word:rain": [tc1],
        "word:flood": [tc2],
    }
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.temporal_between("rain", "damage")
    assert tc1 in result
    assert tc2 in result

def test_temporal_between_returns_empty_when_no_path():
    tc1 = make_temporal_cell("rain", "flood")
    index = {"word:rain": [tc1]}
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.temporal_between("rain", "drought")
    assert result == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest hpm_ai_v6/tests/test_temporal_reasoning.py -k "between" -v
```
Expected: `AttributeError: 'ReasoningAgent' object has no attribute 'temporal_between'`

- [ ] **Step 3: Implement temporal_between**

Add to `ReasoningAgent`:

```python
    def temporal_between(
        self, start: str, end: str, include_lapsed: bool = False
    ) -> List["TemporalCell"]:
        """Chain of temporal intervals connecting start to end."""
        index = self._temporal_index()
        start_key = start if start.startswith("word:") else f"word:{start}"
        end_key = end if end.startswith("word:") else f"word:{end}"

        def search(current_key: str, path: list, visited: set) -> list:
            if current_key == end_key:
                return path
            if current_key in visited:
                return []
            visited = visited | {current_key}
            candidates = index.get(current_key, [])
            if not include_lapsed:
                candidates = [c for c in candidates if not c.lapsed]
            candidates = sorted(candidates, key=lambda c: -c.onset_weight)
            for tc in candidates:
                result = search(tc.effect.name, path + [tc], visited)
                if result:
                    return result
            return []

        return search(start_key, [], set())
```

- [ ] **Step 4: Run all temporal reasoning tests**

```bash
pytest hpm_ai_v6/tests/test_temporal_reasoning.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/reasoning_agent.py hpm_ai_v6/tests/test_temporal_reasoning.py
git commit -m "feat: add temporal_between to ReasoningAgent"
```

---

## Task 8: Temporal intent parsing in reason_with_trace

**Files:**
- Modify: `hpm_ai_v6/agents/reasoning_agent.py`
- Modify: `hpm_ai_v6/tests/test_temporal_reasoning.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to hpm_ai_v6/tests/test_temporal_reasoning.py
def test_reason_with_trace_temporal_sequence_intent():
    tc = make_temporal_cell("rain", "flood")
    index = {"word:rain": [tc]}
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.reason_with_trace("what happened after rain?")
    assert result.get("method") == "temporal_sequence"
    assert "temporal_cells" in result

def test_reason_with_trace_temporal_overlap_intent():
    tc1 = make_temporal_cell("rain", "flood")
    tc2 = make_temporal_cell("rain", "mud")
    tc1.concurrent = [tc2]
    index = {"word:rain": [tc1, tc2]}
    reader = make_reader_mock(index)
    agent = ReasoningAgent(reader)
    result = agent.reason_with_trace("what happened during rain?")
    assert result.get("method") == "temporal_overlap"
    assert "temporal_cells" in result
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest hpm_ai_v6/tests/test_temporal_reasoning.py -k "trace_temporal" -v
```
Expected: FAIL — `method` is not `temporal_sequence`.

- [ ] **Step 3: Add temporal intent detection to reason_with_trace**

In `ReasoningAgent.reason_with_trace()` (around line 2049), add temporal intent detection at the top of the method, before the existing intent checks:

```python
        # Temporal intent detection
        q_lower = question.lower()
        _temporal_sequence_triggers = {"when", "after", "before", "sequence", "then", "next"}
        _temporal_overlap_triggers = {"while", "during", "at the same time", "simultaneously"}

        if any(t in q_lower for t in _temporal_overlap_triggers):
            terms = self._extract_query_terms(question)
            if terms:
                cells = self.temporal_overlap(terms[0])
                return {
                    "question": question,
                    "method": "temporal_overlap",
                    "answer": f"While '{terms[0]}' was active: " + ", ".join(
                        f"{c.cause.name.replace('word:', '')}→{c.effect.name.replace('word:', '')}"
                        for c in cells
                    ) if cells else f"No concurrent intervals found for '{terms[0]}'.",
                    "temporal_cells": [
                        {"cause": c.cause.name, "effect": c.effect.name,
                         "duration_weight": c.duration_weight, "lapsed": c.lapsed}
                        for c in cells
                    ],
                }

        if any(t in q_lower for t in _temporal_sequence_triggers):
            terms = self._extract_query_terms(question)
            if terms:
                cells = self.temporal_sequence(terms[0])
                return {
                    "question": question,
                    "method": "temporal_sequence",
                    "answer": f"Temporal sequence for '{terms[0]}': " + " → ".join(
                        c.effect.name.replace("word:", "") for c in cells
                    ) if cells else f"No temporal intervals found for '{terms[0]}'.",
                    "temporal_cells": [
                        {"cause": c.cause.name, "effect": c.effect.name,
                         "duration_weight": c.duration_weight, "lapsed": c.lapsed}
                        for c in cells
                    ],
                }
```

- [ ] **Step 4: Run all temporal reasoning tests**

```bash
pytest hpm_ai_v6/tests/test_temporal_reasoning.py -v
```
Expected: all pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest hpm_ai_v6/tests/ -v --tb=short -q
```
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v6/agents/reasoning_agent.py hpm_ai_v6/tests/test_temporal_reasoning.py
git commit -m "feat: add temporal intent parsing to reason_with_trace"
```

---

## Self-Review Notes

- All method signatures consistent across tasks (`temporal_sequence`, `temporal_between`, `temporal_overlap` with `include_lapsed` param).
- `TemporalCell.concurrent` is a mutable list — Pydantic may require `default_factory`. If `Cell` uses Pydantic v1, change `concurrent: List["TemporalCell"] = []` to `concurrent: List["TemporalCell"] = Field(default_factory=list)` and import `Field`.
- `_temporal_index()` is a method not a property on `ReasoningAgent` to avoid name clash with dict type annotation.
- Task 5 integration uses `hasattr` sentinel — on first call `build_temporal_cells` is called, subsequently `update_temporal_cells`. This is simple but not thread-safe; acceptable for current single-threaded use.
