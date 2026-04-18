# Plan: WriterAgent – Closing the Cooperative Loop

**Goal:** Implement a `WriterAgent` that generates natural language text (summaries, answers, explanations) using the shared fractal forest and the understanding provided by the `ReaderAgent`.

**Strategic Intent:** We will complete the **Reader-Writer-Web triad**, enabling circular cooperative intelligence where the `WriterAgent` drives knowledge acquisition by identifying gaps during generation.

## Changes

### 1. Writer Mixin (`hpm_ai_v2/agents/mixins/writer.py`)
- [x] Implement `WriterMixin`:
    - [x] `generate_sentence(predicate, agent, patient) -> str`: Basic template-based or POS-macro-based generation.
    - [x] `answer_natural(question) -> str`: Uses `reader_agent.answer_question_hierarchical` and wraps the result in a full sentence.
    - [x] `generate_summary(doc_node, max_sentences=3) -> str`: Selects and renders key sentences from a document hierarchy.
    - [x] `_sentence_concept_score(sent_node) -> float`: Heuristic for sentence importance based on concept weights.

### 2. Writer Agent Implementation (`hpm_ai_v2/agents/writer_agent.py`)
- [x] Implement `WriterAgent(BaseHFNAgent, WriterMixin)`:
    - [x] `__init__`: Accepts a reference to a `ReaderAgent`.
    - [x] `request_knowledge(topic)`: If knowledge is missing, calls `web_agent` (via `reader_agent`) to fetch and ingest content.
    - [x] Handle registration of generated text as new HFN nodes.

### 3. Cooperative Loop Refinement
- [x] Ensure all three agents (`Reader`, `Writer`, `Web`) share the same `TieredForest`.
- [x] Ensure `WriterAgent` uses the same `TextDomainConfig` as `ReaderAgent`.

### 4. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_writer1_coop_loop.py`)
- [x] Implement the full circular loop:
    1. `ReaderAgent` observes a seed passage.
    2. `ReaderAgent` extracts semantic roles.
    3. `WriterAgent` generates a new sentence from those roles.
    4. `WriterAgent` answers a question in natural language.
    5. `WriterAgent` identifies a gap (e.g., "What is deep learning?") and triggers the Web-Reader acquisition pipeline.
    6. `WriterAgent` then answers the previously unknown question.
- [x] Log forest growth and node counts to verify cumulative learning.

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_sp_writer1_coop_loop.py`.
- [x] Verify that generated text is stored as HFN nodes with `derived_from` edges.
- [x] Confirm successful "gap-filling" via `WebAgent`.

