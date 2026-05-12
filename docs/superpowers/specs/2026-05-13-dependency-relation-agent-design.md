# DependencyRelationAgent Design Spec

**Date**: 2026-05-13
**Branch**: hpm-ai-v6
**Status**: Approved — awaiting implementation

---

## Problem

Current reasoning paths must traverse 4-hop sentence-bridge chains:

```
alice → sentence → beginning → sentence → rabbit
```

Each hop carries confidence ~0.45, so the 4-hop path scores 0.45⁴ ≈ **0.04** — too low to be useful. The intermediate nodes (sentence cells) are semantically arbitrary; they encode co-occurrence in a sentence, not structural relationships between concepts.

---

## Solution

Introduce a `DependencyRelationAgent` that uses spaCy's dependency parser to extract **subject-verb-object (SVO) + prepositional triples** from sentences, then creates direct dim-1 edges between concept cells.

For the example sentence "Alice followed the rabbit down the hole.", this yields:

```
alice → followed → rabbit   (2 hops, score 0.85² ≈ 0.72)
```

vs. the current sentence-bridge path scoring ~0.04. Dependency paths will dominate beam search.

---

## HPM Framing

Dependency relations are **level-2 structural patterns** in the HPM hierarchy. Subject-verb-object structure is a relational regularity *above* the level of individual word transitions (level-1). It encodes how concepts participate in events — the foundation of meaningful compositional reasoning.

- **Pattern substrate**: dim-1 Cell objects (edge cells) encoding directional semantic shifts
- **Pattern dynamics**: extracted by spaCy dependency parse; stable once parsed
- **Pattern evaluators**: confidence scores (0.85 SVO, 0.75 prepositional) derived from parse quality
- **Pattern field**: sentence corpus; cross-sentence and coreference patterns are out of scope

---

## Dependency Relations Extracted

| Dep tag | Relation name | Edge direction |
|---------|--------------|----------------|
| `nsubj` | `subject_of` | subject_word → verb_word |
| `dobj`  | `object_of`  | verb_word → object_word |
| `pobj` + prep | `prep_{preposition}` | verb_word → pobj_word |

### Example

Sentence: `"Alice followed the rabbit down the hole."`

Edges produced:
```
word_alice    -[subject_of]→  word_followed   (score: 0.85)
word_followed -[object_of]→   word_rabbit     (score: 0.85)
word_followed -[prep_down]→   word_hole       (score: 0.75)
```

---

## Architecture

### File: `hpm_ai_v6/agents/dependency_relation_agent.py`

Same interface as `SyntacticRuleAgent` (StubAgent-compatible):

```python
class DependencyRelationAgent:
    def __init__(self, nlp=None, base_score: float = 0.85, prep_score: float = 0.75):
        ...

    # Agent interface (required by MultiAgentReader / ReasoningAgent)
    def get_weights(self) -> List[float]: ...
    def _paging_lookup(self) -> Dict[str, Cell]: ...

    # Training
    def learn_from_corpus(self, sentences: List[str]) -> None:
        # 1. Run spaCy dep parse on each sentence
        # 2. Extract SVO + prepositional triples
        # 3. Create/update dim-0 word cells and dim-1 edge cells

    # Persistence
    def save(self, path: str) -> None: ...  # JSON format (see below)
    def load(self, path: str) -> None: ...

    # Utility
    def get_triples(self) -> List[Tuple[str, str, str]]: ...
    # Returns list of (source_word, relation_name, target_word)
```

### Edge Cell Structure

Each relation becomes a dim-1 Cell:

```python
Cell(
    name=f"dep_{relation}_{subject_word}_{object_word}",
    dim=1,
    embedding=target_cell.as_numpy() - source_cell.as_numpy(),
    source=source_cell,   # dim-0 word cell (name="word_{word}")
    target=target_cell,   # dim-0 word cell
    weight=base_score,    # or prep_score for prepositional edges
)
```

Word cells are added to `_paging_lookup()` so `ReasoningAgent` can find them in `_node_index`.

---

## Integration Points

### MultiAgentReader (`hpm_ai_v6/agents/multi_agent_reader.py`)

- Register as `agents["dependency"]`
- `train_sequence`: call `dep_agent.learn_from_corpus(sentences)`
- `warm_start_from_cache`: load from `{cache_dir}/dependency_relations.json`
- After training: save to `{cache_dir}/dependency_relations.json`

### ReasoningAgent (`hpm_ai_v6/agents/reasoning_agent.py`)

- Add `"dependency"` to the `_iter_reasoning_agents` tuple
- `_agent_relation("dependency")` falls through to `"transition"` default (fine — actual relation is encoded in `EdgeRecord.relation`, e.g. `"subject_of"`, `"object_of"`, `"prep_into"`)
- No other ReasoningAgent changes needed — edges flow through `_edge_index` automatically

---

## Scoring

| Edge type | Score | Rationale |
|-----------|-------|-----------|
| `subject_of`, `object_of` | 0.85 | High-confidence SVO parse |
| `prep_{X}` | 0.75 | Slightly less certain; preposition choice can be ambiguous |
| Sentence bridge (existing) | 0.45 | Co-occurrence only |

Dependency paths will be preferred by beam search because 0.85 >> 0.45 for each hop.

---

## Persistence Format (JSON)

`{cache_dir}/dependency_relations.json`:

```json
{
  "triples": [
    {"source": "alice",    "relation": "subject_of", "target": "followed", "score": 0.85},
    {"source": "followed", "relation": "object_of",  "target": "rabbit",   "score": 0.85},
    {"source": "followed", "relation": "prep_down",  "target": "hole",     "score": 0.75}
  ]
}
```

On load: reconstruct Cell objects from triple data, rebuild patterns list.

---

## Files to Create / Modify

| Action | File |
|--------|------|
| CREATE | `hpm_ai_v6/agents/dependency_relation_agent.py` |
| CREATE | `hpm_ai_v6/tests/test_dependency_relation_agent.py` |
| MODIFY | `hpm_ai_v6/agents/multi_agent_reader.py` |
| MODIFY | `hpm_ai_v6/agents/reasoning_agent.py` |
| VERIFY | `requirements.txt` (spacy already expected to be present) |

---

## Testing Strategy

### Unit Tests

1. `learn_from_corpus(["Alice followed the rabbit."])` produces edges `word_alice→word_followed` and `word_followed→word_rabbit`
2. `learn_from_corpus(["She ran into the hole."])` produces `word_ran→word_hole` via relation `prep_into`
3. Save/load round-trip preserves all edges and scores
4. `_paging_lookup()` contains word cells for all subjects, verbs, and objects

### Integration Tests

5. `ReasoningAgent` with dep agent finds an `alice→rabbit` 2-hop path with score > 0.7
6. Dependency edges score higher than sentence bridge edges for the same concept pair

---

## Non-Goals

- Full constituency parsing
- Named entity relation extraction
- Coreference resolution (alice = she)
- Cross-sentence dependency chains
- Semantic role labelling beyond SVO+prep
