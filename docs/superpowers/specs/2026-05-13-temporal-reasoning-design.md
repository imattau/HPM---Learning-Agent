# Temporal Reasoning — Design Spec

**Date**: 2026-05-13
**Branch**: hpm-ai-v6
**Status**: Approved, awaiting implementation

---

## 1. Purpose

Introduce temporal reasoning as a genuine new HPM pattern level (level-6, above the existing meta-pattern layer). The system currently treats the learned graph as a timeless snapshot. This spec adds:

1. **Intra-graph temporal ordering** — reasoning about event sequences and causal intervals within the graph
2. **Inter-graph temporal evolution** — tracking how patterns appear, strengthen, and lapse across training passes

The core abstraction is the **TemporalCell** (3-cell), which reifies a causal transition as a named interval with start/end boundaries defined by cause activation and effect stabilisation.

---

## 2. Scope

### In scope
- `TemporalCell` dataclass (new HPM level)
- `TemporalAgent` — constructs temporal cells from causal edges
- `ReasoningAgent` extensions — three new temporal query methods
- Integration into `MultiAgentReader.maintenance_cycle()`
- Temporal intent parsing in `reason_with_trace()`
- Tests: unit (TemporalAgent) + integration (ReasoningAgent temporal queries)

### Out of scope
- Clock/timestamp-based temporal reasoning
- Persistence of temporal cells across sessions (future)
- Temporal reasoning over non-causal edge types (future)
- Visualisation of temporal intervals

---

## 3. TemporalCell (3-cell) — New HPM Level

### Definition

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from hpm_ai_v6.hpm_model.core.cell import Cell

@dataclass
class TemporalCell(Cell):
    cause: Cell
    # The activating causal pattern (1-cell or 2-cell) — interval start boundary

    effect: Cell
    # The stabilised downstream pattern — interval end boundary

    onset_weight: float
    # Learned strength of the cause activation (inherited from causal edge score)

    duration_weight: float
    # Causal chain depth: 1.0 for a single hop, 2.0 for A→B→C spanning interval, etc.

    concurrent: List["TemporalCell"] = field(default_factory=list)
    # Intervals that overlap this one (shared cause or effect boundary)

    lapsed: bool = False
    # True when onset_weight drops below threshold on a subsequent maintenance cycle
```

### Semantics

"From when `cause` fires to when `effect` stabilises, this interval holds."

Duration is not clock time. It is a learned weight reflecting causal chain depth. Two intervals are **concurrent** when they share a cause boundary or their effect boundaries overlap in the causal graph.

### HPM fit

TemporalCell is a level-6 meta-pattern: it encodes *when* a lower-level pattern dynamic was active, not the dynamic itself. It sits above the existing level-5 meta-patterns (monitoring, error correction, strategy) implemented in `ReasoningAgent.reflect()`.

---

## 4. TemporalAgent

**File**: `hpm_ai_v6/agents/temporal_agent.py`

**Responsibility**: Extract causal chains from the learned graph and construct `TemporalCell`s.

### Interface

```python
class TemporalAgent:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self._temporal_index: Dict[str, List[TemporalCell]] = {}
        # keyed by cause cell name

    def build_temporal_cells(
        self,
        causal_edges: List[EdgeRecord],
    ) -> List[TemporalCell]:
        ...

    def update_temporal_cells(
        self,
        causal_edges: List[EdgeRecord],
    ) -> None:
        # Called on subsequent maintenance cycles — adjusts weights, marks lapsed
        ...

    @property
    def temporal_index(self) -> Dict[str, List[TemporalCell]]:
        return self._temporal_index
```

### Construction logic

1. Filter `causal_edges` to those where `relation == "causes"`.
2. **Segment cells**: for each edge `A→B`, create `TemporalCell(cause=A, effect=B, onset_weight=edge.score, duration_weight=1.0)`.
3. **Spanning cells**: follow chains `A→B→C` (where B is both target of A and source of C). Create `TemporalCell(cause=A, effect=C, duration_weight=2.0, onset_weight=min(score_AB, score_BC))`. Continue for longer chains up to `max_depth=4`.
4. **Concurrent detection**: two cells are concurrent if they share a cause boundary or their effects are connected by a non-causal edge in the graph. Populate `concurrent` lists bidirectionally.
5. Store in `_temporal_index` keyed by `cause.name`.

### Evolution (update cycle)

On subsequent `maintenance_cycle()` calls:
- If a matching cell exists (same cause/effect): update `onset_weight` with new score (exponential moving average, α=0.3).
- If `onset_weight` drops below `threshold`: set `lapsed = True`.
- If a new chain is detected with no existing cell: create fresh cell.
- Do **not** delete lapsed cells — they represent evolution history.

### Integration point

Called from `MultiAgentReader.maintenance_cycle()` after `_add_causal_bridge_edges()` completes, following the same pattern as `DependencyRelationAgent`.

```python
# in MultiAgentReader.maintenance_cycle()
# TemporalAgent receives causal patterns from CausalAgent directly
causal_patterns = list(self.causal_agent.patterns.values())
self._temporal_agent.build_temporal_cells(causal_patterns)
# or update_temporal_cells on subsequent passes
```

---

## 5. ReasoningAgent — Temporal Query Methods

The `ReasoningAgent` gains access to `TemporalAgent.temporal_index` and three new public methods.

### 5.1 `temporal_sequence`

```python
def temporal_sequence(
    self,
    concept: str,
    include_lapsed: bool = False,
) -> List[TemporalCell]:
```

Returns all temporal intervals involving `concept` as cause or effect, ordered by `duration_weight` ascending (shortest/most immediate first).

Answers: *"What sequence of states did X move through?"*

### 5.2 `temporal_between`

```python
def temporal_between(
    self,
    start: str,
    end: str,
    include_lapsed: bool = False,
) -> List[TemporalCell]:
```

Returns the ordered chain of intervals connecting `start` to `end` through the temporal index. Follows cause→effect links greedily, prefers highest `onset_weight`.

Falls back to `_beam_search_path()` over causal edges if no direct temporal path exists, wrapping results as synthetic `TemporalCell`s with `duration_weight` derived from path length.

Answers: *"What happened between X and Y?"*

### 5.3 `temporal_overlap`

```python
def temporal_overlap(
    self,
    concept: str,
    include_lapsed: bool = False,
) -> List[TemporalCell]:
```

Returns all intervals in the `concurrent` lists of any interval involving `concept`.

Answers: *"What else was happening when X was active?"*

### 5.4 Integration with `reason_with_trace()`

The query parser in `_parse_question()` gains two new intents:

| Intent | Trigger words | Method called |
|---|---|---|
| `temporal_sequence` | "when", "after", "before", "sequence", "then", "next" | `temporal_sequence()` |
| `temporal_overlap` | "while", "during", "at the same time", "simultaneously" | `temporal_overlap()` |

The `reason_with_trace()` return dict gains a `temporal_cells` key when a temporal intent is detected:

```python
{
    "answer": "...",
    "method": "temporal_sequence",
    "temporal_cells": [
        {"cause": "...", "effect": "...", "duration_weight": 1.0, "lapsed": False},
        ...
    ]
}
```

---

## 6. Testing

### Unit tests — `TemporalAgent`
- `test_segment_cells_from_causal_edges` — single hop produces one cell
- `test_spanning_cells_from_chain` — A→B→C produces segment cells + spanning cell
- `test_concurrent_detection` — shared cause boundary marks cells as concurrent
- `test_lapsed_on_weight_decay` — onset_weight below threshold sets `lapsed=True`
- `test_update_preserves_lapsed_history` — lapsed cells retained across update cycle

### Integration tests — `ReasoningAgent`
- `test_temporal_sequence_returns_ordered_intervals`
- `test_temporal_between_follows_causal_chain`
- `test_temporal_between_fallback_to_beam_search`
- `test_temporal_overlap_returns_concurrent_intervals`
- `test_reason_with_trace_temporal_intent_parsing`
- `test_include_lapsed_flag_exposes_history`

---

## 7. Open questions (future)

- Should spanning cells with `duration_weight > 3` be suppressed by default (too coarse)?
- Should `temporal_between` use A* rather than greedy traversal for longer chains?
- Cross-session persistence of lapsed cells for long-term evolution tracking.
