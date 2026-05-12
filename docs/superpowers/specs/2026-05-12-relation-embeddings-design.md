# Relation Embeddings Design Spec

**Date**: 2026-05-12
**Branch**: hpm-ai-v6
**Status**: Approved — awaiting implementation

---

## Summary

Add relation embeddings to the HPM pattern graph. Currently `EdgeRecord` carries a `relation` string (e.g. `"lexical_transition"`, `"pos_NOUN"`) and a scalar `score`. There is no vector representation of *how* two concepts are connected — only *that* they are and with what confidence.

Relation embeddings give each relation type its own learned vector, enabling:
- Structural distinction between `"caused-by"`, `"is-a"`, `"follows-in-sequence"`, `"analogous-to"`
- Better analogy cache (find patterns with similar *relationship structure*, not just similar endpoints)
- Richer reasoning paths (compose relation vectors: A→B via r₁ + B→C via r₂ predicts A→C via r₁∘r₂)

---

## HPM Framing

This is a **level-3 pattern substrate** addition. Currently level-3 cells encode rules (`dim=3`). Relation embeddings add a *vector field* over the level-1 transitions — encoding what kind of transition each edge represents, independent of the specific source/target cells.

This maps to TransE/RotatE from knowledge graph research:
- **TransE**: `h + r ≈ t` (source + relation ≈ target in embedding space)
- **RotatE**: relation = rotation in complex space

We use a simpler TransE variant suited to the HPM architecture: each relation type has a learned vector, trained to minimise `||source_emb + relation_emb - target_emb||` across observed transitions.

---

## Core Components

### 1. RelationRegistry

**File**: `hpm_ai_v6/hpm_model/storage/relation_registry.py`

```python
class RelationRegistry:
    def __init__(self, embedding_dim: int = 64)

    def get_or_create(self, relation_name: str) -> np.ndarray
        # Returns learned embedding for this relation type.
        # New relations get random init; updated via update().

    def update(self, relation_name: str, source_emb: np.ndarray,
               target_emb: np.ndarray, lr: float = 0.01) -> None
        # Online TransE gradient step:
        # r += lr * (target - source - r)

    def similarity(self, rel_a: str, rel_b: str) -> float
        # Cosine similarity between two relation embeddings.

    def predict_target(self, source_emb: np.ndarray, relation_name: str) -> np.ndarray
        # Returns source_emb + relation_emb  (TransE prediction).

    def find_similar_relations(self, relation_name: str, top_k: int = 5) -> List[Tuple[float, str]]
        # Returns relations with similar embedding vectors.

    def save(self, path: str) -> None
    def load(self, path: str) -> None
        # JSON format: {relation_name: [float, ...]}
```

### 2. EdgeRecord enrichment

`EdgeRecord` already has `relation: str`. No dataclass change is needed. The `RelationRegistry` maps that string to a vector on demand.

### 3. ReasoningAgent integration

- `ReasoningAgent.__init__`: accept optional `relation_registry: Optional[RelationRegistry] = None`
- `_beam_search_all_paths`: when scoring edges, add a bonus for edges whose `relation_emb + source_emb` is close to `target_emb` (TransE coherence score):
  ```
  coherence = cosine(source_emb + relation_emb, target_emb)
  adjusted_score = edge.score * (0.5 + 0.5 * coherence)
  ```
- `_build_analogy_cache`: use `relation_registry.similarity()` to group patterns by relation type, not just embedding similarity
- New method `_predict_missing_edge(source, relation_name, top_k)`: use TransE prediction to find likely targets even when no direct edge exists

### 4. Training integration

In `MultiAgentReader.train_sequence`, after each agent learns:
- For each new `EdgeRecord`, call `relation_registry.update(edge.relation, source.as_numpy(), target.as_numpy())`
- This gradually trains relation embeddings from the corpus

### 5. Persistence

`MultiAgentReader` saves relation registry to `{cache_dir}/relation_registry.json` after `train_sequence`, and loads it on `warm_start_from_cache`.

---

## Data Flow

```
train_sequence(sentences)
    │
    ├── agents learn edges (EdgeRecord with relation string)
    │
    ▼
RelationRegistry.update(relation, source_emb, target_emb) for each new edge
    → r += lr * (target - source - r)   [TransE gradient]
    → relation embeddings converge to mean offset for that relation type
    │
    ▼
ReasoningAgent._beam_search_all_paths(start, goal)
    → for each candidate edge:
        coherence = cosine(source_emb + relation_emb, target_emb)
        adjusted_score = edge.score * (0.5 + 0.5 * coherence)
    → coherent transitions score higher
    │
    ▼
ReasoningAgent._predict_missing_edge(source, "lexical_transition")
    → predicted_target = source_emb + relation_emb["lexical_transition"]
    → find_similar(predicted_target) in node_index
    → return top-k candidate targets even without direct edges
```

---

## What This Enables

1. **Lexical bridging**: `"alice" → "rabbit"` via predicted lexical transition even when direct edge weight is low
2. **Relation-aware analogy**: `"cat:sat :: dog:ran"` detected by similar relation embeddings, not just similar endpoints
3. **Missing link prediction**: query asks about X→Y, no edge exists, but TransE predicts Y from X + relation type
4. **Relation composition**: r(A,B) ∘ r(B,C) ≈ r(A,C) — if we know A teaches B and B teaches C, predict A teaches C

---

## Non-Goals

- Full RotatE (complex space rotation) — TransE is sufficient for this graph
- Per-edge learned embeddings (too many params) — per-relation-type is the correct level
- Replacing existing edge scoring entirely — relation score is a bonus/modifier, not replacement

---

## Files to Create / Modify

| Action | File |
|--------|------|
| CREATE | `hpm_ai_v6/hpm_model/storage/relation_registry.py` |
| CREATE | `hpm_ai_v6/tests/test_relation_registry.py` |
| MODIFY | `hpm_ai_v6/agents/reasoning_agent.py` — use registry in beam search + new predict method |
| MODIFY | `hpm_ai_v6/agents/multi_agent_reader.py` — train registry, save/load |

---

## Testing Strategy

### Unit tests (`test_relation_registry.py`)
- `RelationRegistry` learns correct relation vector from repeated (source, relation, target) triples
- `predict_target` returns embedding close to actual target after sufficient training steps
- `find_similar_relations` clusters semantically related relation names
- Save/load round-trip preserves all embeddings exactly

### Integration tests
- `ReasoningAgent` with trained registry scores coherent paths higher than incoherent ones
- `_predict_missing_edge` finds plausible targets for known relation types when no direct edge exists
