# Plan: Ensure Fractal Uniformity in ReaderAgent

**Goal:** Represent sentences, paragraphs, and documents as HFN nodes to ensure fractal uniformity across all levels of text abstraction.

**Strategic Intent:** We will extend `ReaderAgent` to build a complete hierarchy of HFN nodes for every observed text. Instead of ephemeral Python lists, sentences will be nodes with word inputs, paragraphs will be nodes with sentence inputs, and so on. This enables structural reasoning, traversal, and composition at all levels.

## Changes

### 1. Reader Agent Extensions (`hpm_ai_v2/agents/reader_agent.py`)
- [x] Add `build_sentence_node(tokens)`: Creates an HFN node for a sentence, linking to word macros.
- [x] Add `build_paragraph_node(sentence_nodes)`: Creates an HFN node for a paragraph.
- [x] Add `build_document_node(para_nodes)`: Creates an HFN node for a document.
- [x] Update `observe_passage(text)`: 
    - [x] Split into sentences.
    - [x] Build sentence nodes.
    - [x] Group into paragraph node(s).
    - [x] Store hierarchy in forest.
- [x] Update `ingest_wikipedia_page`: Build document-level nodes from chunks.

### 2. Mixin Adaptations
- [x] Update `SyntaxMixin._tag_sentence`: Support both token lists and sentence nodes.
- [x] Update `SemanticRoleMixin.extract_roles`: Support sentence nodes.

### 3. Topic Clustering Refinement
- [x] Update `build_topic_clusters`: Cluster sentence/paragraph nodes instead of raw passage vectors.
- [x] Link topic nodes to their constituent structural nodes as children.

### 4. Verification Experiment (`hpm_ai_v2/experiments/experiment_fractal_uniformity.py`)
- [x] Ingest a document.
- [x] Verify that sentence nodes exist and have correct word children.
- [x] Verify that paragraph nodes exist and have correct sentence children.
- [x] Demonstrate structural retrieval (e.g., finding the parent sentence of a specific word node).

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_fractal_uniformity.py`.
- [x] Check forest for correctly typed nodes (`sentence`, `paragraph`, `document`).
- [x] Ensure no regressions in existing retrieval and query logic.
