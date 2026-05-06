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

---

## 2. Pipeline Orchestration (`hpm_ai_v5.pipeline`)

The high-level bridge between raw data and core HPM logic.

### `HPMPipeline`
Orchestrates preprocessing, engine execution, and postprocessing.
- **`step(raw: Any, goal: dict, context: dict) -> PipelineResult`**: Runs a full cycle.
- **`register_preprocessor(adapter: Adapter)`**: Adds a stage to the preprocessing chain.

---

## 3. Adapters & Processing (`hpm_ai_v5.adapter`)

Adapters transform raw data into HPM-compatible states and back.

### NLP & Bridging Adapters
- **`NLPTokenizer`**: spaCy-based tokenizer for natural language queries.
- **`CanonicalPhraser`**: Normalizes synonyms and variable phrases into stable placeholders.
- **`NL2CodeBridgeAdapter`**: Maps functional linguistic tokens to `UnifiedVocabulary` structural IDs.
- **`KnowledgeBaseLookup`**: Simulates external dictionary lookup for synonym expansion and semantic hypothesis testing.

### CLT & Code Adapters
- **`UnifiedASTFlattener`**: Linearizes code (Python, Java) into a universal structural sequence.
- **`UnifiedStateAdapter`**: Maps universal AST nodes to `UnifiedVocabulary` IDs for the core engine.
- **`UnifiedVocabulary`**: Shared mapping for string tokens (keywords, functional concepts) to stable numeric IDs.

### Physics & Control Adapters
- **`CartpoleStateAdapter`**: Normalizes continuous physics observations into HPM-compatible deltas.
- **`ChangepointAdapter`**: Detects distribution shifts in signals (e.g., reward or polygraph score).

---

## 4. Polygraphs (`hpm_ai_v5.polygraphs`)

Polygraphs provide multi-view reliability and consensus for pattern selection.

### `PolygraphGenerator`
Generates multiple internal representations (e.g., Token view, Skeleton view).
- **`NLPPolygraphGenerator`**: Provides Token, Canonical, Skeleton, and **Semantic** views. The Semantic view leverages `KnowledgeBaseLookup` to resolve linguistic ambiguity.
- **`CLTPolygraphGenerator`**: Provides Unified Node, Control Skeleton, and **Functional Skeleton** views for cross-modal transfer.

### `PolygraphEvaluator`
- **`agreement(view_actions, scores)`**: Calculates the dispersion and support across views to determine the "consensus action."
- **`score_engine(engine)`**: Evaluates engine reliability based on pattern concentration and density.
