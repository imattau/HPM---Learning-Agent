# Plan: Grammar Library Implementation

Implement a pluggable Grammar Library using NLTK to provide syntactic guidance to the HPM agent.

## Objective
Enhance the HPM agent's linguistic awareness by adding a "Grammar Library" that validates Part-of-Speech (POS) transitions. This library will be used by the `InstitutionalField` to reward patterns that produce grammatical sequences and by the `Reasoner` to prune ungrammatical planning paths.

## Key Files & Context
- `hpm_ai_v4/tools/grammar.py`: (NEW) Core grammar logic using NLTK `treebank`.
- `hpm_ai_v4/field.py`: Update `InstitutionalField` to include `grammar_bonus`.
- `hpm_ai_v4/agents/reasoning.py`: Update `Reasoner` to support grammar-guided planning.
- `hpm_ai_v4/agents/agent.py` & `hpm_ai_v4/meta.py`: Plumbing to pass the grammar library through the architecture.

## Implementation Steps

### Task 1: NLTK Grammar Library
- [ ] Create `hpm_ai_v4/tools/grammar.py` with `GrammarValidator` (ABC) and `NLTKGrammarLibrary`.
- [ ] Implement `NLTKGrammarLibrary`:
    - Download `treebank` and `averaged_perceptron_tagger_eng`.
    - Build a POS transition matrix (bigrams) from the `treebank` corpus.
    - Implement `score_sequence(words: List[str]) -> float`.
    - Implement `is_valid_transition(prev_word: str, current_word: str) -> bool`.

### Task 2: Institutional Field Integration
- [ ] Update `InstitutionalField.__init__` to accept `grammar_library`.
- [ ] Modify `evaluate()` to include a `grammar_bonus` based on the predicted word sequence's POS transitions.

### Task 3: Reasoner Integration
- [ ] Update `Reasoner.__init__` and `plan()` to support grammar-guided pruning.
- [ ] Implement `_get_pos(word: str)` helper (using NLTK tagger).
- [ ] Use `grammar_library.is_valid_transition` during planning rollouts.

### Task 4: Plumbing & Meta-Layer
- [ ] Update `HPMAgent`, `AgentPool`, `HPMMetaLayer`, and `TotalHPMSystem` constructors to accept and propagate the `grammar_library`.

## Verification & Testing
- [ ] Create `hpm_ai_v4/tests/test_grammar.py` to verify POS transition scoring (e.g., "the cat" should score higher than "the sat").
- [ ] Update `hpm_ai_v4/simulations/experiment_generative_output.py` with a `--use-grammar` flag to demonstrate the effect on generated text.
- [ ] Run all tests to ensure no regressions in existing dictionary or pattern logic.
