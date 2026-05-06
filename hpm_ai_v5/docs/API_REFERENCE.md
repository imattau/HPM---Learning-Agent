# HPM v5 API Reference

This document provides a comprehensive reference for the core components, adapter system, and pipeline orchestration of the Hierarchical Pattern Modelling (HPM) v5 framework.

---

## 1. Core Learning Engine (`hpm_ai_v5.core`)

The core engine handles the discovery, stabilization, and composition of structural invariants.

### `PatternEngine`
The central orchestrator for the HPM learning and acting loop.
- **`observe(state: State) -> MatchResult | None`**: Processes a new state and updates the `PatternStore`.
- **`act(goal: Mapping[str, float] | None = None, horizon: int = 1) -> Action`**: Selects a pattern/sequence and provides a forecast.
- **`promote_to_meta(pattern_names, name=None) -> Pattern`**: **(V5 Extension)** Compresses a sequence into a hierarchical meta-pattern.

### `Pattern`
A structural invariant (delta template).
- **`template`**: Numerical delta representation.
- **`precision`**: Per-slot inverse variance ( Heteroscedastic matching).
- **`children`**: List of patterns for hierarchical composition.
- **`predict(state) -> State`**: One-step forecast.
- **`distance(observation) -> float`**: Weighted error calculation.

### `PatternStore`
Library of leaf and meta-patterns.
- **`match(observation) -> MatchResult`**: Finds closest pattern (Exact, Near, Novel).
- **`prune()`**: Managed forgetting based on density/utility.

### `PatternManager`
Cross-episode persistence and promotion.
- **`end_episode(engine, context)`**: Archives high-utility patterns.
- **`start_episode(engine, context)`**: Seeds engine with context-relevant knowledge.

---

## 2. Pipeline Orchestration (`hpm_ai_v5.pipeline`)

The high-level bridge between raw data and core HPM logic.

### `HPMPipeline`
Orchestrates preprocessing, engine execution, and postprocessing.
- **`step(raw: Any, goal: dict, context: dict) -> PipelineResult`**: Runs a full cycle.
- **`register_preprocessor(adapter: Adapter)`**: Adds a stage to the preprocessing chain.

### `PipelineResult`
- **`input`**: The preprocessed `State`.
- **`action`**: The HPM-selected `Action`.
- **`polygraph_scores`**: Reliability metrics from multiple views.
- **`polygraph_agreement`**: Consensus data from the polygraph evaluator.

---

## 3. Adapters & Processing (`hpm_ai_v5.adapter`)

Adapters transform raw data into HPM-compatible states and back.

### Preprocessing Adapters
- **`UnifiedASTFlattener`**: Converts code (Python, Java, etc.) into a generic structural sequence.
- **`CartpoleStateAdapter`**: Normalizes physics observations into HPM states.
- **`UnifiedVocabulary`**: Maps structural tokens to numeric IDs.
- **`ChangepointAdapter`**: Detects distribution shifts in any signal (e.g., reward or polygraph score).

### Postprocessing Adapters
- **`CLTRefinementAdapter`**: Maps abstract code deltas back to language-specific syntax.
- **`ActionSequenceUnpacker`**: Unrolls `execute_sequence` actions into step-by-step environment commands.
- **`ValidationOnlyAdapter`**: Simple pass-through for benchmarks requiring external validation.

---

## 4. Polygraphs (`hpm_ai_v5.polygraphs`)

Polygraphs provide multi-view reliability and consensus for pattern selection.

### `PolygraphGenerator`
- **`generate(packet: AdapterPacket) -> list[PolygraphView]`**: Creates multiple internal representations (e.g., AST view, Skeleton view, Token view).

### `PolygraphEvaluator`
- **`agreement(view_actions, scores)`**: Calculates the dispersion and support across views to determine the "consensus action."

### Specialized Polygraphs
- **`CLTPolygraphGenerator`**: Validates structural patterns across language boundaries.
- **`PhysicsPolygraphGenerator`**: Evaluates prediction reliability in continuous control tasks (CartPole).
- **`GridPolygraphGenerator`**: Handles spatial/topological patterns (ARC).
