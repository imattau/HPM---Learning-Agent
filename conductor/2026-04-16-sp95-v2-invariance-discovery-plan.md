# Plan: SP95 v2 — True Invariance Discovery (Inverse Sprinkler)

## Objective
Upgrade the SP95 experiment to rigorously test HPM's ability to discover physical invariants. The agent must learn that torque sign is independent of flow direction ($Q$) by discovering the $Q \to Q^2$ transformation using basic mathematical primitives.

## Key Changes from v1
- **Data Encoding**: Replace `Q_sign` and `Q_mag` with a single signed scalar `Q`.
- **Primitives**: Replace high-level concepts (`MOMENTUM_FLUX`) with base math ops (`OP_SQUARE`, `OP_SIN`, `OP_MUL`, `OP_SIGN`).
- **Zero-Shot**: Enforce reuse of the discovered macro code in Phases 2 and 3 using `executor.run_batch` instead of `agent.solve`.
- **Invariance Proof**: Force the agent to reconcile contradictory flow signs ($Q > 0$ and $Q < 0$) using the same output target ($+1$).

## Key Files & Context
- **Domain Config**: `hpm_ai_v2/domains/fluid_domain.py`
- **Renderer**: `hpm_ai_v2/domains/fluid_renderer.py`
- **Oracle**: `hpm_ai_v2/utils/oracle/fluid_oracle.py`
- **Experiment Script**: `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v2.py`

## Implementation Steps

### 1. Refactor Fluid Domain Components
- [ ] Update `FluidDomainConfig`:
    - Concepts: `["OP_MUL", "OP_SQUARE", "OP_SIN", "OP_SIGN", "VAR_Q", "VAR_RHO", "VAR_L", "VAR_THETA"]`.
- [ ] Update `FluidRenderer`:
    - Implement `OP_SQUARE`, `OP_SIN`, etc.
    - Support variable loading primitives if needed, or simple sequential execution on `res`.
- [ ] Update `FluidOracle`:
    - Minimize code structural flags to prevent leakage. Focus on the output sign.

### 2. Implementation of Experiment v2
- [ ] Create `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v2.py`.
- [ ] **Data Generation**:
    - Training: 1 Inflow example ($Q = -1.0$, Output = $+1$).
    - Validation: 1 Outflow example ($Q = +1.0$, Output = $+1$).
    - Test: Novel geometry + Novel flow magnitude ($Q = -1.5$, Output = $+1$).
- [ ] **Phase 1: Discovery**: Solve for the Inflow case using BFS.
- [ ] **Phase 2: Strict Validation**: Run the Phase 1 macro on the Outflow case. Success only if it correctly predicts $+1$ without re-solving.
- [ ] **Phase 3: True Zero-Shot**: Run the macro on the novel geometry.
- [ ] **Phase 4: Structure Audit**: Inspect the HFN macro to confirm it contains `OP_SQUARE` (the invariance mechanism).

## Verification & Testing
- [ ] Run the experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler_v2.py`.
- [ ] Confirm Phase 2 success (Invariance validated).
- [ ] Confirm Phase 3 success (Generalization validated).
- [ ] Verify macro code correctly implements `Q**2` or equivalent.
