# Plan: SP-Reader 10 - Question Answering with Hierarchical Evidence Retrieval

**Goal:** Implement a robust question-answering pipeline in `ReaderAgent` that utilizes hierarchical structural nodes (sentences, paragraphs) and semantic roles (SRL) to find and extract factual answers.

## Changes

### 1. Semantic Role Mixin Enhancements (`hpm_ai_v2/agents/mixins/srl.py`)
- [ ] Add `extract_roles_from_node(node: HFN)`: Extract semantic roles from a sentence HFN node.
- [ ] Add `score_role_match(q_roles, candidate_roles)`: Calculate a similarity score between a question's roles and a candidate sentence's roles, accounting for wildcards and concept-level matches.

### 2. Reader Agent QA Pipeline (`hpm_ai_v2/agents/reader_agent.py`)
- [ ] Add `retrieve_by_predicate(predicate_word)`: Find sentence nodes in the forest that contain a specific verb/predicate.
- [ ] Add `answer_question_hierarchical(question)`:
    - [ ] Parse question into roles (AGENT, PREDICATE, PATIENT).
    - [ ] Determine the target role (who -> AGENT, what -> PATIENT).
    - [ ] Retrieve candidate sentences using hierarchical topic/concept search + predicate matching.
    - [ ] Score and select the best evidence node.
    - [ ] Extract the answer phrase from the target role of the winning node.
- [ ] Integrate concept-level generalization: If no exact match is found, use the concept-level `mu` vectors to find semantically similar evidence (e.g., matching "animal" to "cat").

### 3. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_reader10_qa.py`)
- [ ] Phase 1: Simple factual QA on a small corpus ("The cat chased the mouse").
- [ ] Phase 2: Concept-level generalization ("Which animal chased the mouse?").
- [ ] Phase 3: Multi-passage Wikipedia QA (Ingest Python and AI pages, ask specific domain questions).
- [ ] Verify accuracy and hierarchical traversal.

## Verification
- [ ] Run `hpm_ai_v2/experiments/experiment_sp_reader10_qa.py`.
- [ ] Ensure 100% accuracy on explicit role matches.
- [ ] Ensure >70% accuracy on generalized concept queries.
