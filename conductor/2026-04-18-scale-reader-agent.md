# Plan: Scaling ReaderAgent (Phase 4)

**Goal:** Transform `ReaderAgent` from a collection of clusters into a deep structural hierarchy (HFN-native) with recursive summarization and predictive curiosity.

**Strategic Intent:** Building on the robust SP-Reader3 foundation, we will now implement deep structural wiring, multi-level summarization, and predictive curiosity.

## Changes

### 1. Structural Wiring (L2 -> L3 -> L5)
- [ ] Update `build_topic_clusters` to link `passage` nodes as children of `topic` nodes using `topic_node.add_child(passage_node)`.
- [ ] Update `stabilize_universal_concepts` to link `topic` nodes as children of `concept` nodes.
- [ ] This enables the `Retriever` to use `StructuralRetriever` dynamics.

### 2. Predictive Curiosity (L4-Guided)
- [ ] Implement `predictive_curiosity_score(text)`:
    - Predict the next topic mu using L4 thematic nodes.
    - Measure the Euclidean distance between the actual encoded text and the prediction.
    - Higher distance = higher "structural surprise" = higher curiosity.
- [ ] Update `observe_if_curious` to use this score.

### 3. Hierarchical Summarization
- [ ] Implement `summarize_node(node_id)`:
    - For an L3/L5 centroid, find the top 5 words from the vocabulary that have the highest weights in the centroid mu vector.
    - Return a string like "Topic [word1, word2, word3]".

### 4. Hierarchical Retrieval
- [ ] Implement `query_hierarchical(question)`:
    - Step 1: Find the best L5 Concept.
    - Step 2: Traverse children to find the best L3 Topic under that concept.
    - Step 3: Traverse children to find the best L2 Passage.
    - This is more efficient and "agentic" than flat k-NN.

## Verification
- [ ] New unit tests in `tests/test_reader_scaling.py`:
    - `test_structural_wiring_depth`: Verify that concept nodes have topic children.
    - `test_predictive_curiosity`: Verify that unpredictable sequences get higher scores.
    - `test_summarization`: Verify keyword extraction from centroids.
    - `test_hierarchical_retrieval`: Verify it finds the same (or better) results than flat query.
