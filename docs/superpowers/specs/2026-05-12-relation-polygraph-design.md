# Relation-Polygraph Migration Design Spec

**Date**: 2026-05-12
**Branch**: hpm-ai-v6
**Status**: Approved — awaiting implementation

---

## Problem

`RelationRegistry` (implemented in commit 6e6b9cde) is a separate side-structure running parallel to the pattern polygraph. This violates HPM principles: relation embeddings **are** patterns and should live and compete in the polygraph like everything else.

Two specific violations:

1. **Duplication** — a second learning system with its own update rule and persistence (`relation_registry.json`)
2. **No competition** — relation patterns never get pruned, field-reinforced, or interact with other patterns through the polygraph dynamics

---

## HPM Framing

A relation embedding is a **level-2 pattern** — a regularity *over* level-1 transitions. Specifically, it captures "edges labelled X have a consistent source→target offset vector." This is exactly what `dim=2` cells represent: analogies between 1-cells.

`ReasoningAgent._build_explicit_analogy_index` already indexes dim-2 cells. Once relation cells exist in the pattern graph as dim-2 cells, they flow through that machinery automatically.

---

## Architecture

### 1. RelationPatternEmitter (new, small class)

**File**: `hpm_ai_v6/hpm_model/storage/relation_pattern_emitter.py`

Responsibility: given a batch of `(source_cell, relation_name, target_cell)` observations, emit or update dim-2 `Cell` objects encoding the relation-type embedding.

```python
class RelationPatternEmitter:
    def __init__(self, embedding_dim: int = 64)

    def observe(self, source: Cell, relation_name: str, target: Cell) -> Cell
        # Returns a dim-2 Cell representing this relation type.
        # Cell.name        = f"rel_{relation_name}"
        # Cell.embedding   = running-mean of (target_emb - source_emb) offsets
        # Cell.source      = representative source cell (or None)
        # Cell.target      = representative target cell (or None)
        # Cell.weight      = observation count (used by polygraph dynamics)

    def get_relation_cells(self) -> List[Tuple[Cell, float]]
        # Returns all relation cells + weights for pager storage.
```

Each unique `relation_name` maps to one dim-2 Cell. The embedding is the running mean of `(target_emb - source_emb)` across all observed transitions of that type — equivalent to the steady-state of TransE, expressed as a Cell in the polygraph.

### 2. RelationRegistry becomes a thin in-memory cache

Keep `RelationRegistry` but remove all file I/O. It is populated from dim-2 cells on warm start and exported to dim-2 cells after training.

New methods to add:

```python
class RelationRegistry:
    def populate_from_cells(self, relation_cells: List[Cell]) -> None
        # Load relation embeddings from dim-2 cells into the in-memory map.

    def to_cells(self) -> List[Tuple[Cell, float]]
        # Export all relation entries as dim-2 cells for pager storage.
```

Methods to remove: `save()`, `load()`, all `relation_registry.json` path logic.

### 3. Agent integration in MultiAgentReader

`MultiAgentReader` holds one shared `RelationPatternEmitter`. After each agent trains:

- Iterate the agent's newly learned patterns.
- For each dim-1 pattern with `source` and `target`: call `emitter.observe(source, agent_relation_name, target)`.
- After all agents finish: call `registry.populate_from_cells(emitter.get_relation_cells())`.
- Pass relation cells to the word_agent's pager (or a dedicated relation pager) for storage.

In `warm_start_from_cache`:

- Hydrate relation cells via `hydrate_patterns_from_archive` (same path as all other dim-2 patterns).
- Call `registry.populate_from_cells(relation_cells)` to rebuild the in-memory lookup.

### 4. Persistence via existing pager infrastructure

Relation cells are dim-2 cells. They pass through the normal pattern pager. No separate JSON file. `warm_start_from_cache` already calls `hydrate_patterns_from_archive`, which handles dim-2 patterns — relation cells appear there automatically once stored through the pager.

Existing `relation_registry.json` files: delete on next run (no migration needed — relation embeddings will be re-learned from the next training pass).

### 5. ReasoningAgent changes (minimal)

- `_build_explicit_analogy_index` already processes dim-2 cells — relation cells appear here automatically.
- **Coherence scoring**: instead of `registry.coherence_score(src, rel_name, tgt)`, look up the dim-2 cell for `rel_name` from `_explicit_analogy_index` and compute cosine similarity.
- **`_predict_missing_edge`**: use the dim-2 cell's embedding as the relation vector — no registry dependency needed.

---

## Migration Steps

1. Add `RelationPatternEmitter` (new file).
2. Update `RelationRegistry`: add `populate_from_cells`, `to_cells`; remove `save`, `load`, JSON path logic.
3. Update `MultiAgentReader.train_sequence`: use emitter → registry → pager pipeline.
4. Update `MultiAgentReader.warm_start_from_cache`: hydrate registry from dim-2 cells.
5. Update `ReasoningAgent` coherence scoring to use dim-2 cells directly.
6. Remove `relation_registry.json` persistence path everywhere.
7. Keep `RelationRegistry` as fast in-memory lookup cache only.

---

## Files Affected

| Action | File |
|--------|------|
| CREATE | `hpm_ai_v6/hpm_model/storage/relation_pattern_emitter.py` |
| MODIFY | `hpm_ai_v6/hpm_model/storage/relation_registry.py` |
| MODIFY | `hpm_ai_v6/agents/multi_agent_reader.py` |
| MODIFY | `hpm_ai_v6/agents/reasoning_agent.py` |
| CREATE | `hpm_ai_v6/tests/test_relation_pattern_emitter.py` |

---

## Non-Goals

- Changing how existing dim-2 analogy cells work.
- Replacing `RelationRegistry` entirely (keep as fast in-memory cache).
- Migrating existing `relation_registry.json` files (delete them; re-learned automatically).
- Changing the TransE-style update formula (running mean is sufficient for this stage).
