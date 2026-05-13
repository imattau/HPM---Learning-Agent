# Abductive Reasoning — Design Spec

**Date**: 2026-05-13
**Branch**: hpm-ai-v6
**Status**: Approved, awaiting implementation

---

## 1. Purpose

Add abductive reasoning to the `ReasoningAgent`: given an **observed effect**, find the **minimal explanatory subgraph** — the smallest set of cells and edges that best accounts for it. This is distinct from backward chaining (which finds a valid proof path) and beam search (which finds the highest-scoring path). Abduction ranks candidate explanations by plausibility and selects the most parsimonious one.

HPM alignment: abduction is the inverse of pattern activation — given that a downstream pattern is active, infer which upstream configuration most plausibly produced it. This mirrors HPM's core learning dynamic.

---

## 2. Scope

### In scope
- `ExplanatorySubgraph` dataclass
- `ReasoningAgent.abductive_explain()` method
- `_score_explanation()` internal scoring method
- Intent trigger in `reason_with_trace()` ("why", "explain", "what caused", "how did")
- Tests: unit + integration

### Out of scope
- Multi-effect abduction (explaining two simultaneous observations jointly)
- Abduction over temporal intervals (future — combine with TemporalCell layer)
- Persisting explanations across sessions

---

## 3. ExplanatorySubgraph dataclass

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.agents.reasoning_agent import EdgeRecord

@dataclass
class ExplanatorySubgraph:
    effect: Cell
    # The observed effect being explained

    root_causes: List[Cell]
    # Cells with no incoming causal edges within the subgraph — the hypothesised origins

    edges: List[EdgeRecord]
    # The minimal set of edges connecting root_causes to effect

    plausibility: float
    # Aggregate score: product of edge scores, penalised by subgraph size

    depth: int
    # Number of hops from root_cause to effect (size proxy)
```

### Semantics

The explanatory subgraph is **minimal** in the sense that removing any edge would disconnect a root cause from the effect. **Plausibility** is computed as the noisy-OR of edge scores (consistent with `_noisy_or_score` already in `ReasoningAgent`), divided by `depth` to penalise unnecessarily long explanations (Occam's razor).

---

## 4. ReasoningAgent.abductive_explain()

### Signature

```python
def abductive_explain(
    self,
    effect: str,
    max_depth: int = 4,
    top_k: int = 3,
) -> List[ExplanatorySubgraph]:
```

### Behaviour

1. Resolve `effect` to a `Cell` via `_resolve_cell()`. Return `[]` if not found.
2. **Collect candidate antecedents**: find all `EdgeRecord`s where `target.name` matches the effect cell, across all agents in `_edge_index`. These are the direct causal predecessors.
3. **Recursive expansion**: for each predecessor, recursively collect *their* antecedents up to `max_depth`. This builds a set of candidate subgraphs rooted at different cells.
4. **Identify root causes**: within each candidate subgraph, root causes are cells with no incoming edges in that subgraph.
5. **Score each subgraph** via `_score_explanation()`.
6. Return the top `top_k` subgraphs sorted by `plausibility` descending.

### _score_explanation()

```python
def _score_explanation(self, edges: List[EdgeRecord], depth: int) -> float:
    scores = [e.score for e in edges]
    raw = self._noisy_or_score(scores)   # already implemented
    return raw / max(depth, 1)           # Occam penalty
```

### Integration with reason_with_trace()

Add intent triggers to `reason_with_trace()` before existing intent checks:

| Trigger words | Method called |
|---|---|
| "why", "explain", "what caused", "how did", "reason for" | `abductive_explain()` |

Return dict gains an `explanatory_subgraph` key:

```python
{
    "question": "...",
    "method": "abductive_explain",
    "answer": "The most plausible explanation for 'flood': rain (0.91) → flood via causes edge.",
    "explanatory_subgraph": {
        "effect": "word:flood",
        "root_causes": ["word:rain"],
        "edges": [...],
        "plausibility": 0.87,
        "depth": 1,
    }
}
```

---

## 5. Testing

### Unit tests — ExplanatorySubgraph construction
- `test_abductive_explain_single_cause` — one direct causal edge → subgraph with one root cause, depth=1
- `test_abductive_explain_chain` — A→B→C observed at C → subgraph rooted at A, depth=2
- `test_abductive_explain_multiple_causes` — two independent edges →C → two root causes in subgraph
- `test_abductive_explain_returns_top_k` — multiple subgraphs → only top_k returned
- `test_plausibility_penalises_depth` — longer chain scores lower than shorter chain with same edge scores

### Integration tests — reason_with_trace
- `test_reason_with_trace_why_intent` — "why did flood happen?" → method=`abductive_explain`
- `test_reason_with_trace_explain_intent` — "explain damage" → method=`abductive_explain`
- `test_abductive_explain_unknown_effect` — unknown term → returns `[]`, answer indicates not found

---

## 6. Open questions (future)

- Should abduction draw on `TemporalCell`s to prefer explanations where the cause temporally precedes the effect?
- Should `root_causes` be limited to 0-cells (concept nodes) or can they be 1-cells (relations)?
- Top-k tie-breaking when plausibility scores are equal (currently undefined — first found wins).
