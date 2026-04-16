# Plan: SP95 v4 — Contextual Transfer & Cognitive Drive

## Objective
Provide definitive proof of HPM's hierarchical abstraction by demonstrating that learned invariants are not just "found" by search, but are **contextually activated** and **structurally required** to solve complex tasks.

## Key Changes from v3
- **Contextual Retrieval**: Integrate `ContextualRetriever` into `BaseHFNAgent`.
- **Background Observation**: Add a phase where the agent "observes" Task 2 data to pre-activate relevant patterns.
- **Strict Depth Constraint**: Limit BFS to Depth 5 for Task 2, making the flat solution (Depth 6) impossible.
- **Cross-Task Transfer**: Add Task 3 (Kinetic Energy) to prove the invariant is reused across different physical contexts.

## Key Files & Context
- **Domain Config**: `hpm_ai_v2/domains/fluid_domain.py`
- **Renderer**: `hpm_ai_v2/domains/fluid_renderer.py`
- **Agent**: `hpm_ai_v2/agents/base_agent.py`
- **Experiment Script**: `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v4.py`

## Implementation Steps

### 1. Upgrade BaseHFNAgent
- [ ] Add `contextual` as a `retriever_type` option in `__init__`.
- [ ] Ensure `BaseHFNAgent` calls `retriever.notify_active` during `observe()` and `solve()`.

### 2. Implementation of Experiment v4
- [ ] Create `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v4.py`.
- [ ] **Step 1: Task 1 (Discover Square)**
    - Learn $Q \to Q^2$.
    - Verify macro registration.
- [ ] **Step 2: Background Observation (Contextual Activation)**
    - Present unlabeled Task 2 inputs to the agent via `agent.observe()`.
    - This should populate the `ContextualRetriever`'s recency buffer with nodes related to $Q$.
- [ ] **Step 3: Task 2 (Inverse Sprinkler with Depth Constraint)**
    - Set `max_depth=5`.
    - Verification: The agent *must* use the macro to succeed.
- [ ] **Step 4: Task 3 (Kinetic Energy Transfer)**
    - Goal: Compute $\rho Q^2$.
    - Verification: Confirm reuse of `macro_discover_square`.

## Verification & Testing
- [ ] Run the experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v4.py`.
- [ ] Confirm Task 2 success at Depth 5.
- [ ] Confirm Task 3 success.
- [ ] Verify `ContextualRetriever` logs show the macro was prioritized.
