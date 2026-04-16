# Plan: SP95 v3 — Hierarchical Transfer (Inverse Sprinkler)

## Objective
Provide empirical proof of HPM's hierarchical abstraction by demonstrating that the agent can learn a physical invariant ($Q^2$) in one task and autonomously reuse it as a primitive in a subsequent, more complex task (Inverse Sprinkler).

## Key Files & Context
- **Domain Config**: `hpm_ai_v2/domains/fluid_domain.py`
- **Renderer**: `hpm_ai_v2/domains/fluid_renderer.py`
- **Oracle**: `hpm_ai_v2/utils/oracle/fluid_oracle.py`
- **Experiment Script**: `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v3.py`

## Implementation Steps

### 1. Refine Fluid Domain
- [ ] Keep basic primitives from v2 (`VAR_Q`, `VAR_THETA`, `OP_MUL_Q`, `OP_SIN`, `OP_SIGN`).

### 2. Implement Experiment v3 (Two-Task Transfer)
- [ ] Create `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v3.py`.
- [ ] **Task 1: `discover_square`**
    - Goal: Compute $Q^2$ from $Q$.
    - Inputs: $Q \in \{-2, -1, 0, 1, 2\}$.
    - Primitives: `VAR_Q`, `OP_MUL_Q`.
    - Verification: Agent solves Task 1, macro `macro_discover_square` is created.
- [ ] **Macro Promotion**:
    - Manually "lift" the successful Task 1 macro into the agent's `_candidate_ops` for the next task.
- [ ] **Task 2: `inverse_sprinkler`**
    - Goal: Compute $\text{sign}(\sin(\theta) \cdot Q^2)$.
    - Inputs: Same as v2 (contradictory signs, $Q=0$).
    - Primitives: `VAR_THETA`, `OP_SIN`, `OP_SIGN` + `macro_discover_square`.
    - **Note**: Do *not* include `OP_MUL_Q` in Task 2's primitives initially, or verify that the agent *prefers* the macro due to depth/utility.
- [ ] **Phase 3: Structure Audit**
    - Confirm the Task 2 solution code contains the Task 1 macro ID.
    - Confirm the search depth is reduced compared to a flat search.

## Verification & Testing
- [ ] Run the experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v3.py`.
- [ ] Verify Task 1 success.
- [ ] Verify Task 2 success using Task 1 macro.
- [ ] Verify Phase 3 (Zero-shot) on novel geometry still works via the hierarchical macro.
