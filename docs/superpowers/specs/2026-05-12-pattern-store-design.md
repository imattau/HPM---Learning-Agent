# PatternStore Design Spec

**Date**: 2026-05-12
**Status**: Approved — awaiting implementation
**Author**: Matt Thomson (design); Claude Code (spec document)

---

## Goal

Introduce a `PatternStore` class that provides **shared, cross-agent, cross-corpus pattern storage** with running-average weight merging. The store replaces fragmented per-corpus pattern caches with a single canonical store where patterns are identified by **embedding vector similarity**, not by name or agent origin.

---

## Background

### Current State

Each training run writes patterns to a per-corpus directory (`.hpm_pattern_cache/<corpus_name>/`). When the same concept appears across five different corpora, it accumulates weight five times in five separate files — none of which reflect the true density of that pattern.

`MultiAgentReader.warm_start_from_cache()` loads these per-corpus caches and distributes patterns to agents before training. This means patterns never cross-pollinate across corpora, and a highly-stable concept seen in many contexts never reaches the weight it deserves.

### HPM Theory Motivation

HPM defines pattern stabilisation as the progressive reinforcement of a pattern across **diverse contexts**. A pattern observed in a single context is tentative; the same pattern confirmed across many unrelated contexts is structurally robust. The current per-corpus architecture inverts this: fragmentation keeps patterns artificially weak.

The `PatternStore` corrects this by merging observations across corpora and agents into a single entry per conceptual pattern, accumulating weight and count across every context in which that pattern appears.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Pattern identity | Cosine similarity >= 0.85 | Matches existing `analogy_threshold` in `ReasoningAgent`; avoids string matching brittleness |
| Merge formula | Running average: `new_avg = (w*n + new_w) / (n+1)` | Numerically stable; does not require storing all historical weights |
| Storage location | `.hpm_pattern_cache/shared/patterns.npz` | Single shared file replaces per-corpus dirs |
| Storage format | NumPy `.npz` archive | Fast, compact, zero new dependencies (numpy already in requirements) |
| Architecture | Standalone `PatternStore` class | New agents use it directly; existing agents unchanged for now (additive migration) |
| Similarity threshold | 0.85 (configurable) | Constructor parameter allows per-instance tuning without touching shared default |

---

## Component Design

### `PatternStore` (`hpm_ai_v6/hpm_model/storage/pattern_store.py`)

```python
class PatternStore:
    def __init__(self, cache_dir: str, similarity_threshold: float = 0.85) -> None:
        """
        Initialise the store.

        Args:
            cache_dir: Root cache directory (e.g. '.hpm_pattern_cache').
                       Store writes to <cache_dir>/shared/patterns.npz.
            similarity_threshold: Cosine similarity cutoff for merging.
                                  Patterns with sim >= threshold are treated as the same.
        """

    def merge(
        self,
        patterns: List[Cell],
        weights: List[float],
        agent_name: str,
    ) -> None:
        """
        Merge a batch of patterns (from one agent, one training sequence) into the store.

        For each (pattern, weight) pair:
          - Compute cosine similarity against all stored vectors.
          - If max_sim >= threshold: update the matching entry:
              stored_weight = (stored_weight * count + weight) / (count + 1)
              count += 1
              agents field: add agent_name if not already present
          - Else: append a new entry with count=1.

        Does NOT call save() — caller must call save() explicitly after a batch.

        Args:
            patterns: List of Cell objects. Each must have a `.embedding` (np.ndarray)
                      and `.name` (str) attribute.
            weights: Parallel list of float weights corresponding to each pattern.
            agent_name: Name of the agent submitting these patterns (e.g. 'word_agent').
        """

    def load(self) -> Tuple[List[Cell], List[float]]:
        """
        Load all stored patterns and their averaged weights.

        Returns an empty ([], []) pair if no store file exists yet.
        Used by MultiAgentReader.warm_start_from_cache() to pre-populate agents.

        Returns:
            (patterns, weights) — parallel lists.
        """

    def find_similar(
        self,
        vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[float, Cell]]:
        """
        Return the top-k most similar stored patterns to the given vector.

        Results are sorted by descending cosine similarity.
        Returns an empty list if the store is empty.

        Args:
            vector: Query embedding (1-D float32 array).
            top_k: Maximum number of results to return.

        Returns:
            List of (similarity_score, Cell) tuples.
        """

    def save(self) -> None:
        """
        Persist the current in-memory store to <cache_dir>/shared/patterns.npz.
        Creates the directory if it does not exist.
        Overwrites any existing file.
        """

    def clear(self) -> None:
        """
        Reset the in-memory store to empty (does not delete the file on disk).
        Used in tests to ensure a clean state between cases.
        """
```

---

## Storage Format (`.npz`)

The file at `.hpm_pattern_cache/shared/patterns.npz` contains the following named arrays:

| Array key | Shape | dtype | Description |
|---|---|---|---|
| `vectors` | `(N, D)` | float32 | Pattern embedding vectors. D = embedding dimension. |
| `weights` | `(N,)` | float32 | Running-averaged weight per pattern. |
| `counts` | `(N,)` | int32 | Number of observations merged into this entry. |
| `names` | `(N,)` | object (str) | Canonical name — most recently seen name for this pattern. |
| `agents` | `(N,)` | object (str) | Comma-separated list of agent names that have contributed to this entry. |

All arrays have the same first dimension `N` (number of stored patterns). String arrays use numpy object dtype and are saved with `allow_pickle=True`.

### Empty store

If `patterns.npz` does not exist, `load()` returns `([], [])`. The first call to `save()` creates the file and directory.

---

## Data Flow

```
MultiAgentReader.train_sequence(sentences)
    |
    |-- word_agent.learn(sentences)     --> patterns + weights
    |-- phrase_agent.learn(sentences)   --> patterns + weights
    |-- semantic_agent.learn(sentences) --> patterns + weights
    `-- ...
    |
    v
for each agent:
    PatternStore.merge(agent.patterns, agent.get_weights(), agent_name)

PatternStore.save()   <- called once after all agents in this sequence
    |
    v  (next session / warm start)
MultiAgentReader.warm_start_from_cache()
    |
    v
PatternStore.load() -> (all_patterns, all_weights)
    -> distribute to all agents as initial pattern set
```

### Cosine similarity merge (detail)

```
for each incoming (embedding_v, weight_w):
    sims = cosine_similarity(embedding_v, stored_vectors)   # shape (N,)
    best_idx = argmax(sims)
    if sims[best_idx] >= threshold:
        n = counts[best_idx]
        weights[best_idx] = (weights[best_idx] * n + weight_w) / (n + 1)
        counts[best_idx] += 1
        # update agents field if agent_name not already listed
    else:
        # append new entry
        vectors  = vstack([vectors, embedding_v])
        weights  = append(weights, weight_w)
        counts   = append(counts, 1)
        names    = append(names, pattern.name)
        agents   = append(agents, agent_name)
```

---

## MultiAgentReader Integration

These changes are **additive** — existing agent interface is unchanged.

### Constructor

```python
self.pattern_store = PatternStore(cache_dir=self.cache_dir)
```

### `warm_start_from_cache()`

```python
patterns, weights = self.pattern_store.load()
if patterns:
    for agent in self.agents:
        agent.warm_start(patterns, weights)  # existing interface
```

### After each `train_sequence()`

```python
for agent_name, agent in self.agent_map.items():
    self.pattern_store.merge(
        agent.get_patterns(),   # existing interface
        agent.get_weights(),    # existing interface
        agent_name,
    )
self.pattern_store.save()
```

---

## Agent Interface (New Agents)

New agents that want to use the store directly accept an optional constructor argument:

```python
def __init__(
    self,
    ...,
    pattern_store: Optional[PatternStore] = None,
) -> None:
    self.pattern_store = pattern_store
```

Existing agents are **not modified** as part of this spec. Migration is additive: existing agents gain access to the shared store only when `MultiAgentReader` merges their output after training.

---

## Files

| Action | Path |
|---|---|
| CREATE | `hpm_ai_v6/hpm_model/storage/__init__.py` |
| CREATE | `hpm_ai_v6/hpm_model/storage/pattern_store.py` |
| CREATE | `hpm_ai_v6/tests/test_pattern_store.py` |
| MODIFY | `hpm_ai_v6/agents/multi_agent_reader.py` |
| MODIFY | `requirements.txt` — no new dependencies; numpy already present |

---

## Testing Strategy

All tests live in `hpm_ai_v6/tests/test_pattern_store.py`. No agent instances are needed for unit tests — use synthetic `Cell` objects with hand-crafted embeddings.

### Unit tests

| Test | Description |
|---|---|
| `test_merge_idempotent` | Merging the same pattern twice gives running average of the two weights, count=2 |
| `test_merge_below_threshold_stays_separate` | Two patterns with cosine similarity < 0.85 produce two separate entries |
| `test_merge_above_threshold_merges` | Two patterns with cosine similarity >= 0.85 produce one merged entry |
| `test_save_load_roundtrip` | After save + reload, all arrays (vectors, weights, counts, names, agents) are restored correctly |
| `test_load_missing_file` | `load()` returns `([], [])` when no file exists |
| `test_find_similar_top_k` | Returns correct top-k results sorted by descending similarity |
| `test_find_similar_empty_store` | Returns `[]` when store is empty |
| `test_clear_resets_state` | `clear()` empties in-memory store; subsequent `load()` still returns file contents |
| `test_agents_field_accumulates` | Two different agents contributing to same pattern both appear in agents field |

### Integration test

| Test | Description |
|---|---|
| `test_two_train_sequences_weight_grows` | Simulated train_sequence x2 with same pattern -> weight accumulates correctly, count=2 after second merge |

---

## Non-Goals (Out of Scope)

- **Migrating existing per-corpus caches** to the shared store — separate task, not part of this spec.
- **Replacing `ReasoningAgent._build_analogy_cache`** with `find_similar` — noted as a future enhancement; not in scope here.
- **Per-agent similarity thresholds** — single threshold per `PatternStore` instance is sufficient for now.
- **Domain tagging / namespace separation** — all patterns share one namespace; no domain labels.
- **Distributed / concurrent writes** — single-process, single-writer assumption; no file locking.
- **Compression or quantisation** of stored vectors — raw float32 is sufficient at current corpus scales.

---

## Open Questions

1. **Embedding dimension consistency**: All agents must produce embeddings of the same dimension `D` for cosine similarity to be valid. If agents use different embedding models, the store cannot merge across them. Implementation should assert consistent `D` on first merge and raise clearly if a mismatch is detected.

2. **Store growth bounding**: As corpora accumulate, `N` will grow unboundedly. A future spec should address pruning low-count, low-weight entries below a threshold (e.g. `count == 1` and `weight < min_weight` after K training runs).

3. **Canonical name policy**: The spec sets canonical name to "most recently seen." An alternative is "most frequent name." Implementer should confirm with Matt before committing to either.
