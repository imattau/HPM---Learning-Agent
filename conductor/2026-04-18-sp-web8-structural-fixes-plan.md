# Plan: SP-Web8 Book Exam Structural Fixes

This plan addresses the root causes of the "Book Reading & Exam Resit" experiment failure. The core issue is the lack of a unified representational space between questions (encoded via TF-IDF) and structural nodes (sentences, paragraphs, documents) built from word macros.

## Objective
- Unify representational space by adding one-hot semantic bits to word-level structural nodes.
- Ensure FAISS is reliably rebuilt during reindexing.
- Trigger observer dynamics (surprise, weight updates) during ingestion.
- Fix the remedial loop logic to correctly resolve topics into documents.
- Provide a standalone bootstrap path for the experiment.

## Key Files & Context
- `hpm_ai_v2/agents/mixins/spelling.py`: Character-level word macro creation.
- `hpm_ai_v2/agents/reader_agent.py`: Ingestion, reindexing, and structural hierarchy.
- `hpm_ai_v2/agents/librarian_agent.py`: Knowledge search and topic discovery.
- `hpm_ai_v2/experiments/experiment_sp_web8_book_exam_resit.py`: The test runner.

## Implementation Steps

### 1. Unified Representational Space
- [ ] Update `SpellingMixin.learn_word_spelling` to include the one-hot concept bit from `self.config.concepts` for the word, alongside its character-level mean.
- [ ] This ensures that a sentence node (mean of word nodes) has non-zero values in the word-level (TF-IDF) subspace, matching query vectors.

### 2. Reliable Search Infrastructure
- [ ] Update `ReaderAgent.reindex_knowledge_base` to rebuild FAISS if any retriever is active, even if `_faiss` was lazily initialized.
- [ ] Modify `LibrarianAgent.search_knowledge` to return both topic and document nodes to avoid blocking the remedial loop.
- [ ] Update `LibrarianAgent.answer_question_hierarchical` to use normalized mu vectors or a consistent distance metric that handles combined character/word representations.

### 3. Active Learning Dynamics
- [ ] Add `self.observer.observe(node)` in `ReaderAgent.observe_passage`. This ensures that new nodes generate surprise and start gaining weight through active learning, rather than being "dead" registrations.

### 4. Experiment Orchestration
- [ ] Add a bootstrap fallback in `experiment_sp_web8_book_exam_resit.py` if the cumulative knowledge base is missing.
- [ ] Increase `max_passages` from 1 to 10 (or `None`) during the initial ingestion phase to ensure key facts (McCarthy, 1956, etc.) are actually stored.
- [ ] Fix the remedial research loop to correctly handle the "Topic-to-Document" resolution when a topic node is returned.

## Verification & Testing
- Run the experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp_web8_book_exam_resit.py`.
- Verify the following:
  - Phase 1 initializes correctly even if `data/scientific_curiosity_v2` is absent.
  - Phase 2 ingestion captures key facts (check logs for keyword nodes).
  - Phase 3/5 retrieval finds relevant sentences (check `[DEBUG] Librarian: Selected best sentence` for distances < 1.0).
  - Exam score improves from 0/5 to > 0/5 after remedial research.
