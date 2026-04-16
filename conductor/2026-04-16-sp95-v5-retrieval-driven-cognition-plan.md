# Plan: SP95 v5 — Retrieval-Driven Cognition

## Objective
Provide definitive proof of HPM-style cognition by demonstrating that the agent can solve complex physical laws using a **greedy, retrieval-driven chain** without a branching search. This proves that the hierarchical structure and contextual activation are "doing the thinking."

## Key Changes from v4
- **New Strategy**: Implement `_try_greedy_chain` in `BaseHFNAgent`.
    - Zero branching (beam_width=1).
    - At each step, it selects the single best node from the `ContextualRetriever`.
- **Cognition Validation**: Use `_try_greedy_chain` *instead* of BFS for Tasks 2 and 3.
- **Definitive Verdict**: If Task 2/3 solve greedily, search is no longer the assembly mechanism; hierarchy is.

## Key Files & Context
- **Agent**: `hpm_ai_v2/agents/base_agent.py`
- **Experiment Script**: `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v5.py`

## Implementation Steps

### 1. Upgrade BaseHFNAgent
- [ ] Implement `_try_greedy_chain(inputs, outputs, max_depth=6)`.
    - Loop until `max_depth`.
    - At each step, update query based on current residuals.
    - Fetch `k=1` from `retriever`.
    - If success, return path.
- [ ] Add `greedy` to the strategy registry.

### 2. Implementation of Experiment v5
- [ ] Create `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v5.py`.
- [ ] **Step 1: Task 1 (Discover Square)**
    - Learn $Q \to Q^2$ using standard BFS.
- [ ] **Step 2: Contextual Activation**
    - Observe Task 2/3 inputs to activate patterns.
- [ ] **Step 3: Task 2 (Inverse Sprinkler)**
    - Strategy: `greedy` ONLY.
    - Verification: Success confirms hierarchy-driven execution.
- [ ] **Step 4: Task 3 (Kinetic Energy)**
    - Strategy: `greedy` ONLY.
    - Verification: Confirms cross-domain cognition.

## Verification & Testing
- [ ] Run the experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v5.py`.
- [ ] Confirm Task 2 success via `greedy`.
- [ ] Confirm Task 3 success via `greedy`.
- [ ] Structure Audit: Verify macro reuse is preserved in the greedy solution.
