# ReaderAgent Fix Implementation Plan

## Objective
Address structural synchronization bugs, missing imports, fragile reindexing logic, and missing state validations in `hpm_ai_v2/agents/reader_agent.py` to restore full functionality and passing tests.

## Key Files & Context
- `hpm_ai_v2/agents/reader_agent.py`: The primary file requiring updates.
- `hfn/forest.py` and `hfn/tiered_forest.py`: Interfaces to ensure proper usage of the `forest` registry over duplicate states.

## Implementation Steps

### 1. Fix Imports and Initialization
- [ ] Add `from hpm_ai_v2.utils.text_chunker import chunk_passages` for proper text ingestion.
- [ ] Handle `WebAgent` type hint by using a string forward reference (`'WebAgent'`) to avoid circular imports.
- [ ] Remove the invalid `self.counting_oracle.wrapped = self.oracle` assignment in `__init__`.

### 2. Eliminate `self.patterns` State Duplication
- [ ] Remove `self.patterns` dictionary from `__init__` and all methods.
- [ ] Refactor methods relying on `self.patterns` (e.g., `query_hierarchical`, `predict_next_topic`, `get_most_curious_topic`) to use `self.forest.active_nodes()` and `self.forest.get()`.
- [ ] This resolves the synchronization desync risk when `reindex_knowledge_base` modifies vectors but the duplicate cache is not updated.

### 3. Batch Vocabulary Expansion
- [ ] Refactor `ingest_text` and `observe_document` to batch all new words, expand the vocabulary *once*, and then call `reindex_knowledge_base` *before* constructing any new passage, paragraph, or sentence nodes. This prevents the forest dimension mismatch occurring mid-construction.

### 4. Robust Reindexing and Agent Persistence
- [ ] Refactor `reindex_knowledge_base` to use the native `self.forest.reindex(new_dim)` and `self.config.reindex()` methods instead of manually mutating `_mu_index` and `_D`, ensuring FAISS and caches remain synchronized.
- [ ] Refactor `save_agent` and `load_agent` to leverage the `TieredForest` native persistence layer (`save_to_cold` / `load_from_cold`) instead of manually reconstructing the forest from `reader_meta.json`.

### 5. Guards and Defensiveness
- [ ] Add a `self._last_topic_mu is not None` guard in `predictive_curiosity_score` to prevent `AttributeError` when no topics have been ingested yet.
- [ ] Add a `hasattr(self.config, '_passage_vecs')` guard in `learn_thematic_transitions` to handle empty states smoothly.
- [ ] Clean up unused parameters like `node_type` in `query` if they are inconsistent, or implement them properly.

## Verification & Testing
- [ ] Run `pytest tests/test_reader_hpm.py` to ensure all structural instantiation and querying logic passes.
- [ ] Run an end-to-end integration test (e.g., SP-Web8 resit) to confirm that the `ReaderAgent` processes chapters without crashing or hanging due to invalid state.