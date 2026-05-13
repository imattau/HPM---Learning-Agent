# Reasoning-Guided Learning Feedback Loop — Design Spec

**Date**: 2026-05-13
**Branch**: hpm-ai-v6
**Status**: Approved, awaiting implementation

---

## 1. Purpose

This spec defines a bidirectional feedback loop between the `ReasoningAgent` and the learning agents (`MultiAgentReader`, `WordAgent`). Currently data flows one way: learning agents build the graph → ReasoningAgent reasons over it. This spec introduces the return path: reasoning outputs guide the next training pass.

This implements HPM's **level-5 meta-pattern layer** — monitoring and strategy over lower-level pattern learning. The ReasoningAgent knows things the learning agents don't:
- Which concepts it cannot connect (sparse graph regions → curiosity signal)
- Which multi-hop paths it found (could be reinforced as direct edges)
- Which analogies it detected (could bootstrap related pattern learning)

---

## 2. Scope

### In scope
- `ReasoningSignal` dataclass
- `ReasoningAgent.reflect()` method
- `MultiAgentReader.maintenance_cycle` update to consume signals
- `MultiAgentReader._reinforce_edge()` helper
- Focus-weighted sentence selection
- Tests: unit + integration

### Non-goals (future)
- Causal intervention selection
- Analogy bootstrapping (logged only in this iteration)
- Cross-session persistence of reasoning signals
- Modifying how agents learn internally (reinforcement is additive, not substitutive)

---

## 3. Core Data Structure: ReasoningSignal

```python
from dataclasses import dataclass, field
from typing import List, Tuple
from hpm_ai_v6.hpm_model.core.cell import Cell

@dataclass
class ReasoningSignal:
    uncertain_concepts: List[str] = field(default_factory=list)
    # Words the reasoning agent resolved to cells but could not connect via paths

    high_value_paths: List[List[dict]] = field(default_factory=list)
    # Paths worth reinforcing — each path is a list of step dicts from reason_with_trace

    suggested_focus_words: List[str] = field(default_factory=list)
    # Words MultiAgentReader should prioritise in the next training pass
    # (initially identical to uncertain_concepts; future: may diverge)

    derived_edges: List[Tuple[Cell, Cell, float]] = field(default_factory=list)
    # (source_cell, target_cell, combined_score) — direct edges to inject
    # into word_agent.patterns from multi-hop paths found during reasoning
```

---

## 4. New Method: ReasoningAgent.reflect()

### Signature
```python
def reflect(self, queries: List[str], top_k_paths: int = 5) -> ReasoningSignal:
```

### Behaviour
1. Calls `self.reason_with_trace(query)` for each query.
2. If no path was found for a query: records the resolved terms as `uncertain_concepts`.
3. If paths were found: inspects `candidate_paths[:top_k_paths]`.
   - For each multi-hop path (≥2 steps): derives a direct edge from first-step source to last-step target.
4. Deduplicates `uncertain_concepts` (order-preserving), caps at 20.
5. Sets `suggested_focus_words = uncertain_concepts` (simple initial policy).
6. Returns a `ReasoningSignal`.

### Pseudocode
```python
def reflect(self, queries: List[str], top_k_paths: int = 5) -> ReasoningSignal:
    self._ensure_fresh()
    uncertain = []
    high_value = []
    derived = []

    for query in queries:
        trace = self.reason_with_trace(query)

        if trace["chosen_path"] is None:
            for term in trace.get("terms", []):
                cell = self._resolve_cell(term)
                if cell is not None:
                    uncertain.append(term)
        else:
            for path in trace["candidate_paths"][:top_k_paths]:
                steps = path.get("steps", [])
                if len(steps) >= 2:
                    first_src_key = steps[0].get("source_key")
                    last_tgt_key = steps[-1].get("target_key")
                    src_cell = self._node_index.get(first_src_key)
                    tgt_cell = self._node_index.get(last_tgt_key)
                    if src_cell and tgt_cell:
                        derived.append((src_cell, tgt_cell, path["combined_score"]))

    uncertain = list(dict.fromkeys(uncertain))[:20]

    return ReasoningSignal(
        uncertain_concepts=uncertain,
        high_value_paths=high_value,
        suggested_focus_words=uncertain,
        derived_edges=derived,
    )
```

---

## 5. MultiAgentReader Changes

### 5.1 maintenance_cycle signature update
```python
def maintenance_cycle(self, sentences, query_batch=None, ...):
    # ... existing training logic ...

    if query_batch and self.reasoning_agent is not None:
        signal = self.reasoning_agent.reflect(query_batch)

        # Mechanism 1: Derived edge reinforcement
        for src, tgt, score in signal.derived_edges:
            self._reinforce_edge(src, tgt, score)

        # Mechanism 2: Focus words for next training pass
        self._focus_words = set(signal.suggested_focus_words)

    return report
```

### 5.2 New helper: _reinforce_edge()
```python
def _reinforce_edge(self, source: Cell, target: Cell, score: float) -> None:
    """
    Inject a derived edge into word_agent.patterns.
    The pattern is additive — normal dynamics govern whether it survives.
    """
    import numpy as np
    src_emb = source.as_numpy()
    tgt_emb = target.as_numpy()
    if src_emb.shape != tgt_emb.shape:
        return  # mismatched dims — skip silently

    pattern = Cell(
        name=f"derived_{source.name}_{target.name}",
        dim=1,
        embedding=(tgt_emb - src_emb).tolist(),
        source=source,
        target=target,
        weight=score,
    )
    if pattern not in self.word_agent.patterns:
        self.word_agent.patterns.append(pattern)
        self.reasoning_agent.invalidate()  # force graph refresh
```

### 5.3 Focus-weighted sentence selection
In `_split_sentences` or `train_sequence`, sentences containing any word in `self._focus_words` should be sorted to the front of the processing queue.

Simplest implementation:
```python
if self._focus_words:
    chunks.sort(
        key=lambda s: any(w in s.lower() for w in self._focus_words),
        reverse=True
    )
```

---

## 6. Files to Modify / Create

| File | Change |
|---|---|
| `hpm_ai_v6/agents/reasoning_agent.py` | Add `ReasoningSignal` dataclass; add `reflect()` method |
| `hpm_ai_v6/agents/multi_agent_reader.py` | Update `maintenance_cycle`; add `_reinforce_edge`; add `_focus_words` attribute |
| `hpm_ai_v6/tests/test_reasoning_feedback.py` | New test file (see §7) |

---

## 7. Testing Strategy

### Unit tests
| Test | Setup | Assert |
|---|---|---|
| `test_reflect_returns_uncertain_when_no_path` | Mock `reason_with_trace` to return `chosen_path=None`, terms resolved to cells | `signal.uncertain_concepts` contains those terms |
| `test_reflect_returns_derived_edges_when_path_found` | Mock `reason_with_trace` to return 2-step path | `signal.derived_edges` contains `(src_cell, tgt_cell, score)` |
| `test_reinforce_edge_adds_pattern` | Real WordAgent with 2 cells | After `_reinforce_edge`, pattern in `word_agent.patterns` |
| `test_reinforce_edge_invalidates_reasoning_agent` | Real ReasoningAgent with `invalidate()` spy | `invalidate()` called once |
| `test_reinforce_edge_skips_dim_mismatch` | Cells with different embedding dims | No exception; no pattern added |

### Integration tests
| Test | Setup | Assert |
|---|---|---|
| `test_reflect_then_reinforce_included_in_refresh` | Real agents, real cells | After `reflect` + `_reinforce_edge`, `reasoning_agent.refresh()` graph includes the new edge |
| `test_second_reasoning_pass_finds_reinforced_edge` | Build minimal graph; reinforce a shortcut edge | Second `reason()` call resolves via the shortcut with higher confidence than before |

---

## 8. HPM Alignment

| HPM concept | Implementation |
|---|---|
| Meta-patterns (level 5) | `ReasoningAgent.reflect()` monitors lower-level pattern gaps and feeds back |
| Pattern evaluators | Reasoning confidence + path score act as evaluator signal |
| Pattern dynamics | `_reinforce_edge` is additive injection; normal gating (weight threshold, coherence) governs survival |
| Curiosity/boredom | `uncertain_concepts` → `suggested_focus_words` implements curiosity-driven attention |
| Hierarchical levels | Word-level edges are reinforced by phrase/semantic reasoning at higher levels |

---

## 9. Open Questions (for implementation)

1. Should `_reinforce_edge` use a weight floor (e.g. min score 0.3) to avoid injecting near-zero edges?
2. Should `_focus_words` decay after one pass or persist until the uncertainty is resolved?
3. Should `reflect()` be called automatically inside `maintenance_cycle` when `query_batch` is provided, or should the caller always pass it explicitly?

These are decisions for the implementer; the spec intentionally leaves them open.
