# Plan: SP-Reader 8 (Spelling Induction & Misspelling Detection)

**Goal:** Extend `ReaderAgent` with character-level primitives (L1) to learn word spellings as macros, detect misspellings, and handle case sensitivity.

**Strategic Intent:** We will introduce a `SpellingMixin` that adds character-level orthographic capabilities. The agent will learn macros representing the character sequence of words. This enables it to detect typos using edit distance and reason about spelling at a level below word semantics.

## Changes

### 1. Domain Extension (`hpm_ai_v2/domains/text_domain.py`)
- [x] Update `TextDomainConfig.__init__` to optionally include character primitives:
    - [x] `CHAR_a` to `CHAR_z` (lowercase)
    - [x] `CHAR_A` to `CHAR_Z` (uppercase)
    - [x] `CHAR_DIGIT_0` to `CHAR_DIGIT_9` (digits)
    - [x] Utility primitives: `TO_UPPER`, `TO_LOWER`, `CHAR_EQ`, `STRING_LEN`, `CHAR_AT`, `EDIT_DISTANCE`, `FIND_CLOSEST`.
- [x] Add `include_char_primitives` flag to constructor.

### 2. Spelling Mixin (`hpm_ai_v2/agents/mixins/spelling.py`)
- [x] Implement `SpellingMixin` class:
    - [x] `learn_word_spelling(word, case_sensitive)`: Creates a macro pattern representing the sequence of characters.
    - [x] `detect_misspelling(word, known_words)`: Calculates edit distance and suggests the closest match.
    - [x] `_get_char_node(char, case_sensitive)`: Retrieves or creates an HFN node for a character.
    - [x] `_edit_distance(s1, s2)`: Levenshtein distance implementation.

### 3. Reader Agent Integration (`hpm_ai_v2/agents/reader_agent.py`)
- [x] Inherit from `SpellingMixin`.
- [x] Update documentation to reflect SP-Reader 8 capabilities.
- [x] Update `save_agent` and `load_agent` to persist `word_spellings` mapping.

### 4. Experiment (`hpm_ai_v2/experiments/experiment_sp_reader8_spelling.py`)
- [x] **Phase 1: Learning correct spellings**: Learn a vocabulary of words as macros.
- [x] **Phase 2: Detecting misspellings**: Test with common typos and missing letters.
- [x] **Phase 3: Case sensitivity**: Verify different behavior for "Apple" vs "apple" when configured.
- [x] **Phase 4: Regression Test**: Ensure POS tagging and SRL still work correctly.

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_sp_reader8_spelling.py` and ensure all phases pass.
- [x] Verify that misspelled words correctly suggest the closest known word.
- [x] Verify case sensitivity behavior.
