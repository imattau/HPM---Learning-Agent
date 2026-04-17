# Plan: SP-Reader 7 (Semantic Role Induction)

**Goal:** Extend `ReaderAgent` to learn semantic roles (agent, patient, instrument) from examples and use them for question answering and event understanding.

**Strategic Intent:** We will introduce a `SemanticRoleMixin` that adds SRL (Semantic Role Labelling) capabilities. The agent will learn macros that map syntactic structures (from `SyntaxMixin`) to semantic roles. This will enable complex query answering (e.g., "Who did what?") and cross-domain transfer of role schemas.

## Changes

### 1. Domain Extension (`hpm_ai_v2/domains/text_domain.py`)
- [x] Add `get_srl_primitives()` to `TextDomainConfig`:
    - `SRL_AGENT`, `SRL_PATIENT`, `SRL_INSTRUMENT`, `SRL_PREDICATE`, `SRL_GET_ROLE`.

### 2. Semantic Role Mixin (`hpm_ai_v2/agents/mixins/srl.py`)
- [x] Implement `SemanticRoleMixin` class:
    - [x] `learn_role_mapping(examples)`: Induces a macro that maps sentence structure to roles.
    - [x] `extract_roles(sentence)`: Extracts roles from a sentence.
    - [x] `answer_role_query(question, target_role)`: Answers questions based on roles.
    - [x] Handle auxiliary verbs (e.g., "does", "did") in interrogation.

### 3. Reader Agent Integration (`hpm_ai_v2/agents/reader_agent.py`)
- [x] Inherit from `SemanticRoleMixin`.
- [x] Add `srl_macro` and `role_knowledge` state for persistence.
- [x] Integrate roles into `query_hierarchical` (simulated via `answer_role_query`).

### 4. Experiment (`hpm_ai_v2/experiments/experiment_sp_reader7_srl.py`)
- [x] **Phase 1: Agent & Patient Induction**: Train on 3 basic sentences (e.g., "cat chases mouse").
- [x] **Phase 2: Generalization**: Test on new verbs and nouns.
- [x] **Phase 3: Instrument Induction**: Train on "with [instrument]" patterns.
- [x] **Phase 4: Role-Based QA**: Answer "What does the cat chase?".
- [x] **Phase 5: Cross-Domain Transfer**: Use SP-Reader 5 analogy to answer roles in a software domain.
- [x] **Phase 6: Thematic Integration**: Measure surprise on role-reversed events.

## Verification
- [x] Run `hpm_ai_v2/experiments/experiment_sp_reader7_srl.py` and ensure all phases pass.
- [x] Verify 100% accuracy on new role mappings.
- [x] Verify zero-shot transfer via structural analogy.
