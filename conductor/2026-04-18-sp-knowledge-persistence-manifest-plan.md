# Plan: HPM Knowledge Persistence & Global Registry

This plan addresses the "Knowledge Silo" problem by standardizing storage paths, ensuring complete state serialization, and creating a global registry for cross-experiment knowledge discovery.

## Objective
- Eliminate ephemeral knowledge storage (no more `tempfile`).
- Implement a **Global Knowledge Registry** to track learned patterns across domains.
- Ensure `save_state` captures all dynamic HPM metadata (weights, timestamps, usage counts).
- Enable **Forest Merging** to unify fragmented knowledge bases.

## Key Files & Context
- `hpm_ai_v2/agents/base_agent.py`: Core persistence logic.
- `hfn/observer.py`: Dynamic state (weights) storage.
- `hfn/tiered_forest.py`: Structural storage and tiering.
- `hpm_ai_v2/registry.py` (New): Global manifest of learned knowledge.

## Implementation Steps

### 1. Robust Persistence (Weights & Metadata)
- [ ] **Observer Stability**: Modify `hfn/observer.py` to accept a `meta_dir`. If not provided, default to a subdirectory of the main forest's `cold_dir` instead of `tempfile.mkdtemp`.
- [ ] **Complete Agent State**: Update `BaseHFNAgent.save_state` to include:
    - `_pattern_timestamps` (Recency)
    - `_pattern_decay_times` (Stabilization)
    - `_pattern_usage_count` (Utility)
    - Call `self.observer.save_state()` explicitly.
- [ ] **Observer State Recovery**: Update `BaseHFNAgent.load_state` to restore these dictionaries and call `self.observer.load_state()`.

### 2. Global Knowledge Registry
- [ ] **Registry Implementation**: Create `hpm_ai_v2/registry.py` with a `KnowledgeRegistry` class.
    - Tracks `domain_id` -> `forest_path`.
    - Stores metadata like `concept_count`, `last_updated`, and `relation_types`.
    - persists to `data/knowledge_base/registry.json`.
- [ ] **Auto-Registration**: Update `BaseHFNAgent.__init__` to register its forest in the global registry upon creation.

### 3. Forest Operations (Merge & Discover)
- [ ] **Forest Merge**: Add `merge(other_forest: TieredForest)` to `TieredForest`.
    - Iterates through the other forest's `_mu_index` and copies nodes (preserving metadata and children).
    - Rebuilds FAISS index after merging.
- [ ] **Cross-Domain Discovery**: Add `orchestrator.discover_knowledge(query: str)` to search across all forests listed in the registry.
### 4. Specialized Agent Hierarchy (Consolidation & Structural Integrity)
- [ ] **LearningAgent(BaseHFNAgent)**: Create `hpm_ai_v2/agents/learning_agent.py`.
    - **Move Logic**: Pull `curiosity_score` and `encode` (if generic) from `ReaderAgent`.
    - **New Logic**: Implement the `Predict -> Observe -> Update` cycle (Prediction Loop).
- [ ] **ReasoningAgent(BaseHFNAgent)**: Create `hpm_ai_v2/agents/reasoning_agent.py`.
    - **Move Logic**: Pull `answer` (base) and `evaluate` logic from `WriterAgent`.
    - **Constraint**: Strict read-only access to `self.forest`.
- [ ] **CoordinatorAgent(BaseHFNAgent)**: Create `hpm_ai_v2/agents/coordinator_agent.py`.
    - **Formalize Logic**: Convert procedural orchestration from `experiment_sp_web8` into reusable `route_query` and `assign_task` methods.
- [ ] **Grounded L4 Transitions**: Modify `StateTransitionModel` to store deltas in HFN node metadata/edges rather than a separate dict.
    - This allows Level 4 rules to participate in forest-wide retrieval and transfer.

### 5. Agent Refactor (Eliminating Duplication)
- [ ] **ReaderAgent**: Inherit from `LearningAgent`. Remove local `curiosity_score`. Ensure `ingest_text` calls `self.predict()` before observation.
- [ ] **WebAgent**: Inherit from `LearningAgent`. Remove any duplicated encoding logic.
- [ ] **WriterAgent**: Inherit from `ReasoningAgent`. Rename `answer_natural` to `answer` (override). Ensure no learning methods are called.
- [ ] **ExecutiveAgent**: Inherit from `CoordinatorAgent`. Use `assign_task` to delegate chapter reading and web search.

### 6. Final Experiment Verification
...

- [ ] **Canonical Paths**: Update `experiment_sp_web8_book_exam_resit.py` and other active experiments to use `data/knowledge_base/hpm_v2_main/` as the default root.
- [ ] **Executive Workflow**: Use `ExecutiveAgent.run_exam_workflow` to replace manual procedural loops in experiments.
- [ ] **Verification Run**: Run SP-Web8 and verify improvement in score via autonomous research and resit.

## Verification & Testing
- Run SP-Web7 and SP-Web8 in sequence and verify that `dictionary` weights and `pattern_usage_count` are preserved.
- Verify that `data/knowledge_base/registry.json` correctly lists all used domains.
- Test `Forest.merge()` by merging a physics forest into the scientific curiosity forest and performing a joint search.
