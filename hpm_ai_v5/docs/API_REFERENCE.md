# HPM v5 API Reference

This document is a concise reference for the current public API of the HPM v5 framework.

---

## 1. Core (`hpm_ai_v5.core`)

### `PatternEngine`
Central learning and acting loop.

| Method | Signature | Notes |
|--------|-----------|-------|
| `observe` | `(state: State) -> MatchResult \| None` | Processes a state, updates store, sets `last_match` |
| `act` | `(goal=None, horizon=1, top_k=3) -> Action` | Selects pattern/sequence, returns forecast. `action.confidence` decays over time — use `last_match.status` for discrimination. |
| `select` | `(goal=None, top_k=3) -> Pattern \| None` | Best concrete pattern for current state |
| `select_sequence` | `(goal=None) -> PatternSequence \| None` | Best learned sequence |
| `last_match` | `MatchResult \| None` | Result of most recent `observe()` call |

**`MatchResult.status` values:** `"exact"` / `"near"` / `"novel"` / `"variant"` (after PatternVariant consolidation)

### `Action`
Returned by `act()`.
- `action_type`: `"select"` / `"defer"` — check before using forecast
- `forecast`: `State | None` — single-token state, not a full skeleton prediction
- `confidence`: decaying float — **do not use for input discrimination**

### `Pattern`
A structural invariant (delta template).
- `template`: tuple of floats — canonical state representation
- `support`: int — number of times seen
- `utility`: float — decaying reward signal
- `distance(observation, ...) -> float`: weighted distance to an observation

### `PatternStore`
In-memory pattern library.
- `match(observation) -> MatchResult`
- `learn(observation, name=None) -> Pattern`
- `get(name) -> Pattern | None`
- `top_k(observation, k=3) -> list[Pattern]`
- `register_variant(variant)` — adds a `PatternVariant` (pending ATIS plan)
- `variants: dict[str, PatternVariant]` — promoted near-duplicate clusters
- Eviction policy: drops lowest-utility patterns when `max_patterns` is reached

### `PatternSequence`
A named sequence of pattern references.
- `simulate(state, horizon, resolver, start_offset=0) -> list[State]`

### `PatternManager`
Episode-level promotion and consolidation.
- `start_episode(engine, context=None) -> dict`
- `end_episode(engine, context=None) -> dict` — runs sequence promotion + `_consolidate_variants()`
- `_consolidate_variants(engine) -> int` — promotes near-duplicate patterns to `PatternVariant` nodes when store exceeds `config.consolidation_threshold`

### `CoreConfig`
- `max_patterns: int = 32`
- `near_threshold: float = 1.0`
- `exact_threshold: float = 0.0`
- `consolidation_threshold: float = 0.8` — fraction of `max_patterns` that triggers consolidation
- `history_limit: int = 10`
- `density_decay: float = 0.01`
- `utility_decay: float = 0.005`

---

## 2. Pipeline (`hpm_ai_v5.pipeline`)

### `HPMPipeline`
Orchestrates adapter pipeline → engine → polygraph → postprocessor.

- `step(raw: Any) -> PipelineResult` — full cycle; returns action and polygraph scores
- `register_preprocessor(adapter: Adapter)` — appends to preprocessing chain
- `view_engines: dict[str, PatternEngine]` — per-view engines created by polygraph generator
- `polygraph_confidence_skip: float` — skip polygraph if engine confidence exceeds this (set >1.0 to force always)

---

## 3. Adapters (`hpm_ai_v5.adapter`)

### NLP
- **`NLPTokenizer`**: spaCy `en_core_web_sm`; produces `tokens`, `pos_tags`, `lemmas`
- **`CanonicalPhraser`**: maps lemmas to concept vocabulary (`WEATHER`, `FLIGHT`, etc.)
- **`SkeletonExtractor`**: POS tags → skeleton groups (`N`, `V`, `D`, `P`, `R`, `C`, `A`); maps `AUX→V`, `PART→R`
- **`SkeletonNgramAdapter`**: skeleton bigrams stored in `context["skeleton_ngrams"]` only — not appended to `packet.states`
- **`DeltaEncoder`**: computes first-differing-position delta between consecutive skeletons
- **`KnowledgeBaseLookup`**: WordNet synonym lookup (max 5 candidates); populates `context["semantic_candidates"]`
- **`NL2CodeBridgeAdapter`**: maps NL tokens to `U_*` structural IDs (`U_IF`, `U_WHILE`, `U_TRY`, etc.)

### Code
- **`UnifiedVocabulary`**: global string→int mapping; `get_id(token) -> int`

### Physics / Control
- **`CartpoleStateAdapter`**: normalises CartPole obs to HPM state tuples with action history buffer
- **`ChangepointAdapter`**: detects distribution shifts in reward/polygraph signals

---

## 4. Polygraphs (`hpm_ai_v5.polygraphs`)

### View names by generator

| Generator | View names |
|-----------|-----------|
| `NLPPolygraphGenerator` | `token_view`, `canonical_view`, `skeleton_view`, `skeleton_bigram_view`, `delta_view`, `semantic_view_<candidate>` |
| `CodePolygraphGenerator` | `ast_types`, `token_types`, `skeleton` |
| `CLTPolygraphGenerator` | `unified_node`, `control_skeleton`, `functional_skeleton` |

**Key rule:** `last_match.status` on a view engine is the correct discrimination signal. `act().confidence` on a view engine decays monotonically and has no discriminative value.

### `PolygraphEvaluator`
- `agreement(view_actions, scores)` — weighted consensus across views
- `score_engine(engine)` — reliability score based on pattern concentration

---

## 5. Postprocessors (`hpm_ai_v5.postprocessors`)

- **`ValidationOnlyAdapter`**: passes action through unchanged; used in benchmarks
- **`UCodeRenderer`** *(pending SCB plan)*: maps `U_*` sequences to Python function skeletons
