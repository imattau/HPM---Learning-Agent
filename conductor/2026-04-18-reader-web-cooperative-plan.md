# Plan: ReaderAgent + WebAgent Cooperative Knowledge Loop

**Goal:** Refactor `ReaderAgent` and `WebAgent` to work in a cooperative loop where `ReaderAgent` uses `WebAgent` for all external retrieval, sharing a common `TieredForest`.

**Strategic Intent:** We will remove Wikipedia-specific code from `ReaderAgent`, delegating all discovery to `WebAgent`. This creates a modular system where `ReaderAgent` handles comprehension and `WebAgent` handles retrieval, linked via HFN structural edges.

## Changes

### 1. Web Domain & Mixins Refinement
- [x] **`hpm_ai_v2/domains/web_domain.py`**: Update concepts to `HTTP_GET`, `PARSE_HTML`, `EXTRACT_LINKS`, `SEARCH_ENGINE`, `WIKIPEDIA_FETCH`.
- [x] **`hpm_ai_v2/agents/mixins/web_fetch.py`**:
    - [x] Add `fetch_page(webpage_node) -> str`: Fetches content, updates metadata, returns text.
- [x] **`hpm_ai_v2/agents/mixins/web_search.py`**:
    - [x] Add `search(topic, num_results=1) -> List[HFN]`: Returns a list of result `webpage` nodes.

### 2. ReaderAgent Refactoring (`hpm_ai_v2/agents/reader_agent.py`)
- [x] **Remove Wikipedia methods:** `ingest_wikipedia_page`, `explore_wikipedia`, `mock_wikipedia`.
- [x] **Add Cooperative methods:**
    - [x] `ingest_text(text, title) -> HFN`: High-level ingestion pipeline (sentence split -> passages -> paragraphs -> document).
    - [x] `get_most_curious_topic() -> str`: Selects topic with lowest coverage or highest entropy.
- [x] Update `__init__` to strictly require `web_agent` for external tasks.

### 3. Unified Knowledge Graph Integration
- [x] Ensure both agents share the same `forest` and `observer`.
- [x] Implement `derived_from` edge creation in `ingest_text`.
- [x] Update `TextRenderer` to handle the new concepts if necessary.

### 4. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_reader_web_coop.py`)
- [x] Implement the cooperative loop:
    1. Seed with initial passage.
    2. Build topics.
    3. Identify curious topic.
    4. `web_agent.search(topic)`.
    5. `web_agent.fetch_page(webpage_node)`.
    6. `reader_agent.ingest_text(text)`.
    7. Link `document --derived_from--> webpage`.
- [x] **Logging:** Display forest size and node counts after each step to verify knowledge growth.

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_sp_reader_web_coop.py`.
- [x] Ensure no `wikipedia` library dependency remains in `ReaderAgent`.
- [x] Verify `derived_from` edges in the shared forest.
- [x] Confirm forest growth (node count increases across iterations).
