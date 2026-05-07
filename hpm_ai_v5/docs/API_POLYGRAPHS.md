# HPM v5 Polygraph Reference

All generators implement `PolygraphGenerator`: `name`, `generate(raw, context) -> list[PolygraphView]`.

Each `PolygraphView` has a `name` (used as the key in `pipeline.view_engines`) and a `State`.

**Key rule:** use `view_engine.last_match.status` for discrimination — not `view_engine.act().confidence`, which decays monotonically.

---

## View Name Reference

| Generator | View names produced |
|-----------|-------------------|
| `NLPPolygraphGenerator` | `token_view`, `canonical_view`, `skeleton_view`, `skeleton_bigram_view`, `delta_view`, `semantic_view_<candidate>` |
| `CodePolygraphGenerator` | `ast_types`, `token_types`, `skeleton` |
| `CLTPolygraphGenerator` | `unified_node`, `control_skeleton`, `functional_skeleton` |
| `NumericPolygraphGenerator` | `exact_view`, `noisy_view`, `trend_view` |
| `GridPolygraphGenerator` | `flat_grid`, `connected_components`, `object_signatures` |
| `GraphPolygraphGenerator` | `adjacency_view`, `degree_view` |
| `PhysicsPolygraphGenerator` | `state_view`, `delta_view`, `forecast_view` |
| `AcrobotPolygraphGenerator` | `state_view`, `energy_view` |
| `ActionPolygraphGenerator` | `action_history_view` |
| `AudioPolygraphGenerator` | `frequency_view`, `energy_view` |
| `TimeSeriesPolygraphGenerator` | `raw_view`, `trend_view`, `residual_view` |
| `TextPolygraphGenerator` | `token_view`, `ngram_view` |

---

## NLP (`hpm_ai_v5.polygraphs.nlp`)

`NLPPolygraphGenerator` reads from `packet.context`:
- `tokens` → `token_view`
- `canonical_tokens` → `canonical_view`
- `skeleton` → `skeleton_view`
- `skeleton_ngrams` → `skeleton_bigram_view`
- `delta` → `delta_view`
- `semantic_candidates` → one `semantic_view_<cand>` per candidate (max 3)

---

## Code (`hpm_ai_v5.polygraphs.code`)

`CodePolygraphGenerator` accepts raw Python source or a single token string:
- Full source → `ast_types` (flat AST node IDs), `token_types` (tokenize types), `skeleton` (keywords + operators)
- Single string → `ast_types` with single-element tuple

---

## CLT (`hpm_ai_v5.polygraphs.clt`)

`CLTPolygraphGenerator` produces cross-language structural views from `U_*` sequences:
- `unified_node` — raw U_* state
- `control_skeleton` — control-flow tokens only
- `functional_skeleton` — function-call tokens only

---

## Numeric (`hpm_ai_v5.polygraphs.numeric`)

`NumericPolygraphGenerator` produces three competing views from the same signal for polygraph agreement testing:
- `exact_view` — raw signal
- `noisy_view` — signal + Gaussian noise
- `trend_view` — smoothed trend

---

## Physics (`hpm_ai_v5.polygraphs.physics`)

| Generator | Domain | Key views |
|-----------|--------|-----------|
| `PhysicsPolygraphGenerator` | Generic continuous control | `state_view`, `delta_view`, `forecast_view` |
| `AcrobotPolygraphGenerator` | Acrobot | `state_view`, `energy_view` |

---

## Grid / Graph (`hpm_ai_v5.polygraphs.grid`, `graph`)

| Generator | Key views |
|-----------|-----------|
| `GridPolygraphGenerator` | `flat_grid`, `connected_components`, `object_signatures` |
| `GraphPolygraphGenerator` | `adjacency_view`, `degree_view` |

---

## Other Generators

| Generator | Module | Notes |
|-----------|--------|-------|
| `ActionPolygraphGenerator` | `polygraphs.action_policy` | Action history as a view for planning context |
| `AudioPolygraphGenerator` | `polygraphs.audio` | Frequency + energy views from audio features |
| `TimeSeriesPolygraphGenerator` | `polygraphs.timeseries` | Raw/trend/residual decomposition |
| `TextPolygraphGenerator` | `polygraphs.text` | Token and n-gram views for text |

---

## PolygraphEvaluator (`hpm_ai_v5.core.evaluator`)

- `agreement(view_actions, scores) -> PolygraphAgreement` — weighted consensus across view engines
- `score_engine(engine) -> PolygraphScore` — reliability score based on pattern concentration and density
- Used by `HPMPipeline` automatically when `polygraph_generator` is set
