# Plan: SP-Reader 11 - Autonomous Wikipedia Exploration & Knowledge Graph Building

**Goal:** Enable the `ReaderAgent` to autonomously explore Wikipedia, selecting links based on curiosity and building a cross-document knowledge graph of HFN nodes.

**Strategic Intent:** We will upgrade the `ReaderAgent` to extract hyperlinks, evaluate them using predictive curiosity, and link document-level HFN nodes in a global `Corpus` graph. This demonstrates the agent's ability to perform open-world knowledge acquisition and relational reasoning.

## Changes

### 1. Reader Agent Upgrades (`hpm_ai_v2/agents/reader_agent.py`)
- [ ] **Initialization:** Add `self.corpus_node: Optional[HFN] = None` to `__init__`.
- [ ] **Document Node IDs:** Update `build_document_node` to use `f"document_{title.replace(' ', '_')}"` as the node ID for deterministic lookup.
- [ ] **Ingestion with Links:** Modify `ingest_wikipedia_page` to:
    - Extract `page.links` from the Wikipedia API.
    - Return `(doc_node, links_titles)`.
- [ ] **Corpus Management:** Add `_update_corpus(doc_node, source_doc=None)`:
    - Create a root `Corpus` HFN node if it doesn't exist.
    - Add the `doc_node` as a child of the `Corpus` node.
    - Add a `links_to` edge from `source_doc` to `doc_node` within the `Corpus` node.
- [ ] **Autonomous Exploration:** Add `explore_wikipedia(seed_title, max_iterations=5)`:
    - Loop for `max_iterations`.
    - Ingest current page.
    - Subsample outgoing links (e.g., 20 links).
    - For each link, fetch a 1-sentence summary and calculate `predictive_curiosity_score`.
    - Select and follow the link with the highest curiosity.
- [ ] **Graph Queries:** Add `find_path_between_docs(start_title, end_title)`:
    - Perform Breadth-First Search (BFS) over the `Corpus` node's `_edges` to find a document path.

### 2. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_reader11_exploration.py`)
- [ ] **Setup:** Seed the exploration with "Artificial intelligence".
- [ ] **Execution:** Run `explore_wikipedia` for 5 iterations.
- [ ] **Verification:**
    - List all document nodes in the forest.
    - Query outgoing links from the AI node.
    - Find a path between the seed and the most recently discovered page.
    - Print the fractal hierarchy of one retrieved document to confirm "fractal uniformity" is maintained.

## Verification
- [ ] Run `hpm_ai_v2/experiments/experiment_sp_reader11_exploration.py`.
- [ ] Ensure 5 unique pages are added to the knowledge graph.
- [ ] Ensure `find_path_between_docs` returns a valid sequence of titles.
- [ ] Ensure new words from each page are added to the dynamic vocabulary.
