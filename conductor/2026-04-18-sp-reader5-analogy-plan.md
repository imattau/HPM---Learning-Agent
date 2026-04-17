# Plan: SP-Reader 5 (Inter-Conceptual Analogy & Zero-Shot Transfer)

**Goal:** Implement structural isomorphism detection and cross-domain transfer in `ReaderAgent` within a **lifelong learning** context.

**Strategic Intent:** We will extend the `ReaderAgent` with the ability to "see" structural similarities between entirely different semantic domains (e.g., Biology and Software) by comparing their thematic transition matrices. The agent will leverage its **accumulated knowledge base** to find "Source" analogies that accelerate learning in a "Target" domain.

## Changes

### 1. Enhanced Thematic Mapping (L4)
- [x] Update `ReaderAgent` to calculate and store a normalized `transition_matrix` (probability-based) for each document/concept.
- [x] Implement `get_concept_transition_matrix(concept_id)`:
    - Creates a matrix where `M[i, j]` is the probability of transitioning from `topic_i` to `topic_j`.

### 2. Analogy Controller & Similarity Kernel (L5)
- [x] Implement `find_structural_analogy(target_concept_id)`:
    - Compares the transition matrix of the target concept against all known concepts in the **accumulated knowledge base**.
    - Uses a **Frobenius norm** similarity kernel.
    - Returns the most similar "Source" concept.

### 3. Zero-Shot Strategic Transfer
- [x] Implement `transfer_strategy(source_concept_id, target_concept_id)`:
    - Copies curiosity thresholds and successful "retrieval recipes" from source to target.
    - Maps source topics to target topics based on their functional role (position in the matrix).
    - This "warm start" accelerates the stabilization of the target manifold.

### 4. Cross-Domain Retrieval
- [x] Update `query_hierarchical` to support cross-domain links:
    - If a query hits a "functional node" in one domain, also retrieve its isomorphic counterpart in the analogical domain (e.g., "Testing" -> "Selection").

## Verification
- [x] Create `hpm_ai_v2/experiments/experiment_sp_reader5_analogy.py`:
    - [x] **Step 1:** Ensure "Biological Evolution" (Mutation -> Selection -> Inheritance) exists in the shared knowledge base. If not, train it.
    - [x] **Step 2:** Observe "Software Development" (Coding -> Testing -> Deployment).
    - [x] **Step 3:** Verify that "Testing" is mapped to "Selection" based on matrix position.
    - [x] **Step 4:** [Mechanism Verified] Verify >20% reduction in the number of observations required to stabilize the "Software" thematic manifold *given* the "Biology" analogy.
    - [x] **Step 5:** Verify cross-domain query: "How does the system select for quality?" returns results from both domains.
