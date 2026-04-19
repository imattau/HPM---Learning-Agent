# Plan: HFN Compliance & Infrastructure Stability Fixes

This plan addresses several critical and moderate issues identified in the HFN Compliance Report across the `hpm_ai_v2` agents and mixins. These fixes are essential for the stability and learning integrity of the HPM society.

## Objective
- Fix runtime bugs (AttributeError, NameError) due to uninitialized state.
- Ensure the `Observer` is not bypassed during pattern registration to maintain learning dynamics (weights, scores).
- Correct invalid calls to the retriever in `ReaderAgent`.
- Maintain FAISS index synchronization during reindexing.
- Align semantically ungrounded `mu` in specialized mixins.
- Improve the robustness of `id` mutation for new patterns.

## Key Files & Context
- `hpm_ai_v2/agents/base_agent.py`: Base class for all agents.
- `hpm_ai_v2/agents/reader_agent.py`: Primary perception agent.
- `hfn/tiered_forest.py`: Underlying forest implementation (for FAISS).
- `hpm_ai_v2/agents/mixins/syntax.py`, `hpm_ai_v2/agents/mixins/srl.py`: Specialized mixins.
- `hpm_ai_v2/agents/mixins/l3_relational.py`: Relational schema discovery.

## Implementation Steps

### 1. Base Agent State & Registration Fixes
- [ ] **BaseHFNAgent.__init__**: Initialize `self.patterns = {}` to avoid `AttributeError` in `register_pattern`.
- [ ] **BaseHFNAgent.register_pattern**: Replace all calls to `self.forest.register()` with `self.observer.register()`. This ensures that all reusable patterns have the necessary dynamic state (weights, scores, miss_counts) required by the HPM framework.

### 2. Reader Agent & Retrieval Fixes
- [ ] **ReaderAgent.curiosity_score**: Wrap the raw `mu` vector in an HFN node before calling `self.retriever.retrieve()`.
- [ ] **ReaderAgent.reindex_knowledge_base**: After updating `self.forest._mu_index`, call `self.forest._faiss.rebuild(self.forest._mu_index)` to ensure the high-speed search index is synchronized with the new dimensionality.

### 3. Semantic Grounding in Mixins
- [ ] **SyntaxMixin.learn_pos_tagger**: Update `mu` initialization to use grounded concepts from `self.config` instead of hardcoded indices (e.g., `mu[0] = 1.0`).
- [ ] **SRLMixin.learn_role_mapping**: Update `mu` initialization to use grounded concepts from `self.config` instead of hardcoded indices (e.g., `mu[1] = 1.0`).

### 4. Robust ID Mutation
- [ ] **L3RelationalMixin.discover_meta_schema**: Ensure that `composed.id` is set *before* calling `self.observer.register` to prevent potential inconsistencies.

### 5. Chess Agents Integration (Architectural Alignment)
- [ ] **MentalChessAgent / MacroChessAgent**: Update to inherit from `BaseHFNAgent` to ensure they participate in persistent HPM learning dynamics (TieredForest, Observer). (If time permits and if they are intended to be persistent).

## Verification & Testing
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp_web8_book_exam_resit.py`.
- Verify that `self.patterns` is correctly populated and utilized.
- Verify that FAISS results remain accurate after reindexing.
- Inspect the metadata and `mu` of POS and SRL nodes to ensure semantic grounding.
