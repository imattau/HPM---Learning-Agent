# PatternVariant Consolidation + ATIS Benchmark Design

**Date**: 2026-05-07  
**Branch**: hpm-ai-v5  
**Status**: Approved for implementation

---

## Overview

Two related deliverables:

1. **PatternVariant** — a new architectural node in HPM v5's pattern hierarchy that promotes near-identical patterns into a shared abstraction during `PatternManager.end_episode()`, preventing store saturation without evicting concrete patterns.

2. **ATIS Benchmark** — a benchmark against the ATIS airline travel dataset (5k real utterances, 18 intent classes) that validates the framework at realistic corpus scale, including consolidation effectiveness.

---

## Part 1: PatternVariant

### Motivation

`PatternStore` has a hard `max_patterns` capacity. On small corpora this causes a saturation regime: beyond ~100 training episodes, old patterns are evicted and recognition degrades. More training hurts.

The HPM framework describes stabilisation as the mechanism by which repeated near-identical patterns merge into a more stable, generalised representation. `PatternVariant` implements this directly: when `PatternManager` detects near-duplicate patterns at episode end, it promotes them into a `PatternVariant` node, retaining both as concrete variants beneath it.

### PatternVariant Node

```python
@dataclass
class PatternVariant:
    name: str                          # e.g. "variant_weather_0"
    member_names: list[str]            # names of constituent Pattern objects
    centroid: tuple[float, ...]        # mean of member state values
    hit_count: int                     # sum of member hit counts
    context_signature: dict            # union of member context signatures
    score: float                       # aggregated score
```

`PatternVariant` is a separate type from `Pattern`. It is not a subclass. Concrete `Pattern` objects are always retained — promotion adds a variant node above them, it does not replace them.

### PatternStore Changes

- Add `variants: dict[str, PatternVariant]` to store.
- Add `register_variant(variant)` method.
- Add `consolidation_threshold: float = 0.8` to `CoreConfig` — consolidation is requested when store reaches 80% of `max_patterns`.
- Extend `top_k()` with a variant-level fallback: when concrete matching returns no exact/near hits, query variant centroids using the same distance metric. Return variant hits with status `"variant"`.
- Eviction remains as a last resort at 100% capacity.

### PatternManager Extension

`end_episode()` gains a `_consolidate_variants()` pass after existing sequence promotion:

```python
def _consolidate_variants(self, engine):
    # Only run when store exceeds consolidation_threshold
    if len(engine.store.patterns) < engine.config.max_patterns * engine.config.consolidation_threshold:
        return
    patterns = list(engine.store.patterns.values())
    used = set()
    for i, p1 in enumerate(patterns):
        if p1.name in used:
            continue
        cluster = [p1]
        for p2 in patterns[i+1:]:
            if p2.name in used:
                continue
            dist = engine.store.distance(p1.state_value, p2.state_value)
            if dist < engine.config.near_threshold:
                cluster.append(p2)
                used.add(p2.name)
        if len(cluster) >= 2:
            used.add(p1.name)
            engine.store.register_variant(_make_variant(cluster))
```

Consolidation only fires at episode boundaries, never mid-stream. This keeps the hot path unchanged.

### Engine Match Order

1. Exact match (concrete patterns)
2. Near match (concrete patterns, dist < `near_threshold`)
3. Near match (variant centroids, dist < `near_threshold`) — new
4. No match

`last_match.status` gains a new value `"variant"` for step 3 hits.

### Files to Create/Modify

| File | Change |
|------|--------|
| `hpm_ai_v5/core/variant.py` | New — `PatternVariant` dataclass |
| `hpm_ai_v5/core/store.py` | Add `variants` dict, `register_variant()`, variant fallback in `top_k()` |
| `hpm_ai_v5/core/config.py` | Add `consolidation_threshold: float = 0.8` |
| `hpm_ai_v5/core/pattern_manager.py` | Add `_consolidate_variants()`, call from `end_episode()` |
| `hpm_ai_v5/core/engine.py` | Surface `"variant"` status in `last_match` |
| `hpm_ai_v5/core/__init__.py` | Export `PatternVariant` |
| `tests/test_pattern_variant.py` | New — unit tests for consolidation and variant matching |

---

## Part 2: ATIS Benchmark

### Dataset

ATIS (Airline Travel Information System) via HuggingFace `datasets`:
- ~4,978 training utterances, ~893 test utterances
- 18 intent classes (e.g. `flight`, `airfare`, `airport`, `city`, `ground_transport`)
- Slot labels included (not used in this benchmark)

Install: `uv add datasets`

### Pipeline

Same adapter stack as SNLP plus one new adapter:

```
NLPTokenizer → CanonicalPhraser → SkeletonExtractor → SkeletonNgramAdapter
→ KnowledgeBaseLookup → IntentLabelAdapter
```

**`IntentLabelAdapter`**: during training, injects gold intent label into `packet.context["intent_label"]` so the engine can associate skeleton patterns with intent classes. During evaluation, the label is withheld; the benchmark reads `last_match` to recover the predicted intent.

Intent prediction: look up the `Pattern` or `PatternVariant` that produced `last_match`, retrieve its stored `context["intent_label"]`.

### Tasks

**B1: Intent Recognition**  
Train on 80% of ATIS, test on 20%. For each test utterance, predict intent via `last_match` pattern lookup. Metric: top-1 accuracy across 18 classes. Target: >60%.

**B2: Slot Generalisation**  
Hold out utterances containing novel city/airline names (not seen during training). Test whether skeleton matching still predicts the correct intent. Metric: accuracy on held-out entity substitutions. Target: >70%.

**B3: Consolidation Effectiveness**  
Run training twice: with and without `PatternVariant` consolidation. Compare: (a) final store size, (b) B1 accuracy. Consolidation should reduce unique patterns by >30% at 1000 training utterances without degrading B1 by more than 5 percentage points.

**B4: Variant-Level Match Rate**  
Among B1 correct predictions, what fraction came from `"variant"` status matches vs `"exact"`/`"near"` concrete matches? Reports the practical contribution of the new abstraction layer.

### Files to Create

| File | Purpose |
|------|---------|
| `hpm_ai_v5/adapter/atis.py` | `IntentLabelAdapter`, ATIS dataset loader |
| `hpm_ai_v5/experiments/run_atis_benchmark.py` | Benchmark harness (B1–B4) |
| `tests/test_atis_benchmark.py` | Smoke tests — dataset loads, pipeline runs, metrics compute |

### WordNet-backed KnowledgeBaseLookup

The current `KnowledgeBaseLookup` uses a hand-crafted ~10-entry dictionary with no aviation vocabulary. For ATIS this means semantic candidate views produce no signal on domain words like `flight`, `airfare`, `airport`.

Replace the static dictionary with WordNet synset lookups via NLTK:

```python
from nltk.corpus import wordnet

def _synonyms(token: str) -> list[str]:
    syns = set()
    for syn in wordnet.synsets(token):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != token:
                syns.add(name)
    return list(syns)[:5]  # cap at 5 candidates to limit view explosion
```

`KnowledgeBaseLookup.run()` calls `_synonyms()` for each token and populates `semantic_candidates` as before. The static fallback dictionary is removed.

**Setup**: `uv add nltk`, then `python -m nltk.downloader wordnet omw-1.4` (one-time download, ~10MB).

This makes B2 meaningful: novel city/airline names that share WordNet synsets with training tokens will produce matching semantic views without any hard-coded entries.

### Dependencies

- `datasets` (HuggingFace) — add to `pyproject.toml`
- `nltk` with `wordnet` + `omw-1.4` corpora — add to `pyproject.toml`

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| B1 intent accuracy | >60% |
| B2 entity generalisation | >70% |
| B3 store size reduction | >30% without >5pp accuracy drop |
| B4 variant match contribution | >0% (confirms the layer is active) |
| SNLP benchmark unaffected | All 5 tasks still pass at prior levels |

---

## What Is Out of Scope

- Slot filling (ATIS has slot labels; not used here)
- Multi-intent utterances
- Domain-specific KB tuning beyond WordNet coverage
- `PatternVariant` serialisation/persistence (defer to a later checkpoint)
