# Plan: NLP Quality Audit & Aesthetic Enhancement Loop

**Objective:** Implement a systematic way to measure and improve the "readability" and linguistic quality of the NLP produced by HPM agents. This moves beyond simple templates to a structural validation of the generated text.

**Strategic Intent:** Ensure that the fractal knowledge stored in HFNs is correctly and elegantly decoded back into human-readable language.

## Changes

### 1. NLP Quality Auditor (`hpm_ai_v2/utils/nlp_auditor.py`)
- [ ] Create `NLPAuditor` utility:
    - [ ] **Heuristic Scores:** Measure casing consistency, punctuation density, and repetition (n-gram overlap).
    - [ ] **Diversity Score:** Calculate Type-Token Ratio (TTR) to ensure the agent isn't stuck in "loops."
    - [ ] **Coherence Check:** Use a "Reader-as-Critic" pattern where a separate `ReaderAgent` ingests the generated text and we measure the "Surprise" (Affective utility) it triggers. High surprise = low readability/coherence.

### 2. Enhanced Linguistic Heuristics (`hpm_ai_v2/agents/mixins/writer.py`)
- [ ] Improve `generate_sentence`:
    - [ ] Implement proper subject-verb agreement (singular/plural) beyond simple "s" suffixing.
    - [ ] Add smart-joining: Ensure list items are joined with commas and "and" (Oxford comma support).
    - [ ] Capitalization: Ensure every generated block starts with a capital and ends with appropriate terminal punctuation.

### 3. "Critic" Integration in WriterAgent (`hpm_ai_v2/agents/writer_agent.py`)
- [ ] Update `answer_natural` and `generate_summary`:
    - [ ] Implement a **Self-Correction Loop**: The writer generates 3 candidate sentences.
    - [ ] It uses the `NLPAuditor` to score them.
    - [ ] It selects the candidate with the best "Utility" (Accuracy + Coherence - Complexity).

### 4. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_web3_nlp_quality.py`)
- [ ] Run a standard research task (e.g., Topic: "Transformer models").
- [ ] Generate 5 different summaries and answers.
- [ ] **Evaluation:**
    - [ ] Print the `NLPAuditor` report for each output.
    - [ ] Compare "Stage 1" (naive templates) vs "Stage 2" (Critic-selected outputs).
    - [ ] Verify that the average "Surprise" in the Critic Reader decreases as the Writer improves.

## Verification
- [ ] Run `hpm_ai_v2/experiments/experiment_sp_web3_nlp_quality.py`.
- [ ] Confirm that generated text follows standard English casing/punctuation rules.
- [ ] Confirm that the "Self-Correction" logic results in higher linguistic scores.
