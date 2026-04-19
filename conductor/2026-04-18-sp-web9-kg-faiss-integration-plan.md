# Plan: SP-Web9 - Knowledge Graph & FAISS Integration

This plan focuses on enhancing the HPM society with structured fact-based reasoning via a `KnowledgeGraphAgent` and high-speed semantic retrieval using FAISS. It also addresses POS tagging inaccuracies in the `DictionaryAgent`.

## Objective
- Fix POS tagging inaccuracies in `DictionaryAgent` (e.g., 'high' and 'led' being tagged as nouns).
- Implement `KnowledgeGraphAgent` for structured reasoning using Wikidata/SPARQL.
- Integrate FAISS into `TieredForest` for lightning-fast semantic retrieval.
- Demonstrate hybrid retrieval: combining precise graph queries with fuzzy vector search.

## Key Files & Context
- `hpm_ai_v2/agents/dictionary_agent.py`: Lexical grounding.
- `hpm_ai_v2/agents/knowledge_graph_agent.py`: New agent for KG reasoning.
- `hfn/tiered_forest.py`: Storage and retrieval backbone.
- `hpm_ai_v2/agents/librarian_agent.py`: Thematic and conceptual mapping.

## Implementation Steps

### 1. Dictionary Grounding & Corrective Dynamics
- [ ] Update `DictionaryAgent.lookup` to accept an optional `pos_hint`.
- [ ] Refine `DictionaryAgent._fetch_wordnet_entry` with a heuristic to prefer common POS (Verb/Adjective) over rare noun senses.
- [ ] **Native Correction**: If a new definition lookup contradicts an existing one (e.g., different POS), use `Observer.penalize_id()` on the old definition to drop its weight.
- [ ] **Structural Pruning**: Periodically call `Observer.prune()` to clear out the 'losers' with low weight.
- [ ] Ensure `ReaderAgent` calls `observe()` on new definitions to integrate them into the competitive manifold.

### 2. KnowledgeGraphAgent Development
- [ ] **Domain Config:** Create `KnowledgeGraphDomainConfig`.
- [ ] **SPARQL Integration:** Implement `rdflib` and `SPARQLWrapper` for querying Wikidata.
- [ ] **Mapping Logic:** Develop methods to translate Wikidata entities and relationships into HFN nodes and edges.
- [ ] **Relational Primitives:** Expose `query_kg` and `traverse_kg` as HPM-native strategies.

### 3. FAISS Integration
- [ ] **FAISS Wrapper:** Implement a `FaissVectorIndex` to manage a FAISS-backed index.
- [ ] **TieredForest Update:** Add a parallel FAISS index to `TieredForest`.
- [ ] **Synchronization:** Ensure all `register` and `reindex` operations update the FAISS index.
- [ ] **Retrieval Acceleration:** Update `retriever.retrieve` to use FAISS for approximate nearest neighbor search.

### 4. Hybrid Retrieval & Orchestration
- [ ] Update the `Orchestrator` (or create a Routing Agent) to decide between KG queries (precise) and Forest retrieval (semantic).
- [ ] Implement a two-stage retrieval process: KG for entities, then Forest for related abstract patterns.

## Verification & Testing
- Create `hpm_ai_v2/experiments/experiment_sp_web9_hybrid_retrieval.py`.
- Verify the `DictionaryAgent` correctly identifies the POS for 'high' and 'led'.
- Demonstrate a Wikidata query for a complex fact (e.g., "Inventions by Nikola Tesla").
- Measure retrieval speed with FAISS on a 10,000+ node forest.
