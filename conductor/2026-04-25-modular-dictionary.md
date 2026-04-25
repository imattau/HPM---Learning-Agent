# Modular Dictionary Implementation Plan

**Date:** 2026-04-25
**Task:** Implement a pluggable dictionary system to guide reasoning and provide institutional evaluation.

---

## Objective

Integrate external linguistic knowledge into the HPM v4 architecture via a modular dictionary. This will improve planning quality (avoiding non-word sequences) and provide an "institutional" signal for pattern selection.

## Components

### 1. Dictionary Tool (`hpm_ai_v4/tools/dictionary.py`)
- Define `DictionaryValidator` (ABC).
- Implement `NLTKWordList` using the `nltk.corpus.words` dataset (approx. 235k words).
- Automate `nltk.download('words')` within the constructor.
- Use a `trie` for fast prefix search over the NLTK dataset.

### 2. Reasoner Integration (`hpm_ai_v4/agents/reasoning.py`)
- Update `Reasoner` to accept an optional `dictionary`.
- Enhance `plan()` to use `is_prefix()` for pruning invalid word completions.
- Implement helper methods: `_is_word_boundary`, `_partial_word`, `_word_completed`, `_last_word`.

### 3. Institutional Field (`hpm_ai_v4/field.py`)
- Implement `InstitutionalField` that uses the dictionary to evaluate patterns.
- Patterns that predict valid words or prefixes receive a `lexical_bonus`.

### 4. Agent & System Integration
- Update `HPMAgent` to take `dictionary` and pass it to its `Reasoner`.
- Update `HPMMetaLayer` and `TotalHPMSystem` to allow passing a global dictionary instance.

## Implementation Steps

### Task 1: Dictionary Module
- Create `hpm_ai_v4/tools/dictionary.py`.
- Implement `NLTKWordList` with `trie` optimization.
- Ensure efficient memory usage for the 235k word set.
- Add unit tests in `hpm_ai_v4/tests/test_dictionary.py`.

### Task 2: Reasoner Enhancement
- Modify `hpm_ai_v4/agents/reasoning.py`.
- Add dictionary-guided logic to `plan`.
- Implement character decoding logic (using `TextAdapter`) inside the reasoner to reconstruct words from tokens.

### Task 3: Institutional Field
- Modify `hpm_ai_v4/field.py`.
- Add `InstitutionalField` class.
- Integrate it into the agent's `perceive_and_learn` loop or meta-layer.

## Verification

### Unit Test
```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py
```

### Integration Run
```bash
PYTHONPATH=. python3 hpm_ai_v4/simulations/experiment_generative_output.py --use-dictionary
```
(I will update this experiment script to include a dictionary flag).

## Impact
- **Pruned Planning:** The agent will stop hallucinating impossible character sequences.
- **Linguistic Selection:** Replicator dynamics will favor patterns that naturally generate human-readable language.
- **Alignment:** Directly implements the "Institutional Pattern Field" concept from the HPM working paper.
