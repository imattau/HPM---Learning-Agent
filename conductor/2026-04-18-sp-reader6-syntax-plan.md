# Plan: SP-Reader 6 (Syntactic Structure Learning)

**Goal:** Extend `ReaderAgent` to learn parts-of-speech (POS) induction and use it for syntactic parsing (noun phrase extraction) and improved retrieval.

**Strategic Intent:** We will introduce a `SyntaxMixin` that adds POS induction capabilities. The agent will learn macros that map token patterns to POS tags using a small set of examples. These tags will then be used to extract higher-level structures (Noun Phrases) and disambiguate queries. Crucially, the agent will persist its knowledge in `data/knowledge_base/reader_lifelong`.

## Changes

### 1. Domain Extension (`hpm_ai_v2/domains/text_domain.py`)
- [x] Add `get_pos_primitives()` to `TextDomainConfig`:
    - `POS_NOUN`, `POS_VERB`, `POS_ADVERB`, `POS_ADJECTIVE`, `POS_DET`, `POS_PREP`, `POS_CONJ`, `POS_PUNCT`.

### 2. Syntax Mixin (`hpm_ai_v2/agents/mixins/syntax.py`)
- [x] Implement `SyntaxMixin` class:
    - [x] `learn_pos_tagger(inputs, outputs)`: Induces a macro (sequence of HFN nodes) for POS tagging.
    - [x] `_tag_sentence(sentence)`: Applies the learned macro.
    - [x] `extract_noun_phrases(sentence)`: Extracts NPs based on `DET ADJ* NOUN` patterns.
    - [x] `_pos_beam_search()`: Internal helper to search for string-to-tag rules (e.g., suffix-based, dictionary-based).

### 3. Reader Agent Integration (`hpm_ai_v2/agents/reader_agent.py`)
- [x] Inherit from `SyntaxMixin`.
- [x] Add `pos_macro` state to store the induced tagger.
- [x] Update `query_hierarchical` to optionally use syntactic features for disambiguation.

### 4. Experiment (`hpm_ai_v2/experiments/experiment_sp_reader6_syntax.py`)
- [x] **Phase 1: Persistent Load**: Load existing knowledge from `data/knowledge_base/reader_lifelong`.
- [x] **Phase 2: POS Induction**: Train on 3 annotated sentences.
- [x] **Phase 3: Generalization**: Test on new vocabulary with same structure.
- [x] **Phase 4: NP Extraction**: Extract noun phrases from complex sentences.
- [x] **Phase 5: Syntax-Aware Retrieval**: Compare precision with/without syntax for subject-verb queries.
- [x] **Phase 6: Persistence**: Save the expanded knowledge base.

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_sp_reader6_syntax.py` and ensure all phases pass.
- [x] Verify that POS tagging accuracy is 100% on generalization test.
- [x] Verify that Noun Phrases are correctly extracted.
