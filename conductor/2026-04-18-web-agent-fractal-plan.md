# Plan: HPM-Native WebAgent with Fractal Uniformity

**Goal:** Implement a `WebAgent` that follows HPM fractal principles, representing web resources as HFN nodes and learning web interaction strategies as patterns.

**Strategic Intent:** We will deliver a `WebAgent` that autonomously manages web fetching and searching, integrated with the `ReaderAgent` to build a unified cross-document and cross-resource knowledge graph.

## Changes

### 1. Web Domain Definition (`hpm_ai_v2/domains/web_domain.py`)
- [x] Implement `WebDomainConfig(DomainConfig)`:
    - [x] Define L1 primitives: `WEB_HTTP_GET`, `WEB_HTML_PARSE`, `WEB_LINK_EXTRACT`, `WEB_SEARCH`.
    - [x] Provide encoding for `webpage` (URL-based), `search_query` (text-based), and `hyperlink` nodes.

### 2. Web Mixins (`hpm_ai_v2/agents/mixins/web_fetch.py`, `hpm_ai_v2/agents/mixins/web_search.py`)
- [x] Implement `WebFetchMixin`:
    - [x] Add `fetch_webpage(url)`: Returns a `webpage` HFN node.
    - [x] Encapsulate strategies like retries and error handling.
- [x] Implement `WebSearchMixin`:
    - [x] Add `search_web(query)`: Returns a `search_query` HFN node with `webpage` children.

### 3. Web Agent Implementation (`hpm_ai_v2/agents/web_agent.py`)
- [x] Implement `WebAgent(BaseHFNAgent, WebFetchMixin, WebSearchMixin)`:
    - [x] Handle registration of web nodes in the shared forest.
    - [x] Implement `build_webpage_node`, `build_search_query_node`, and `build_hyperlink_node`.

### 4. Integration & Fractal Alignment
- [x] **ReaderAgent Integration:**
    - [x] ReaderAgent can take an optional `WebAgent` reference.
    - [x] Update `ReaderAgent.ingest_wikipedia_page` to use `WebAgent` if available.
    - [x] Add `derived_from` edge from `document` node (Reader) to `webpage` node (Web).
- [x] **Renderer:** Update `TextRenderer` to handle basic rendering of web nodes (URLs).

### 5. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_web1_autonomous.py`)
- [x] Demonstrate autonomous web fetching and search.
- [x] Verify creation of `webpage` and `search_query` nodes in the forest.
- [x] Show integration with `ReaderAgent` (finding information on the web and building fractal document nodes linked to source webpages).

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_sp_web1_autonomous.py`.
- [x] Ensure knowledge graph persistence (all nodes saved to knowledge base).
- [x] Verify fractal uniformity (all web resources are HFN nodes).
