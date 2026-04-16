# Plan: SP95 Inverse Sprinkler Experiment

## Objective
Implement and validate the SP95 experiment, where an HPM agent learns the counter-intuitive rotation direction of an inverse sprinkler (sucking water in). The goal is to demonstrate that HPM can discover hidden physical laws from structured input-output pairs and generalize to new geometries.

## Key Files & Context
- **Domain Config**: `hpm_ai_v2/domains/fluid_domain.py`
- **Renderer**: `hpm_ai_v2/domains/fluid_renderer.py`
- **Oracle**: `hpm_ai_v2/utils/oracle/fluid_oracle.py`
- **Experiment Script**: `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler.py`

## Implementation Steps

### 1. Fluid Domain Components
- [ ] Create `hpm_ai_v2/domains/fluid_domain.py` with `FluidDomainConfig`.
    - Concepts: `["MOMENTUM_FLUX", "TORQUE", "SIGN", "MULTIPLY", "SINE", "COSINE"]`.
    - `S_DIM = 20`.
- [ ] Create `hpm_ai_v2/domains/fluid_renderer.py` with `FluidRenderer`.
    - `render(node)`: Converts HFN nodes into Python code using `numpy`.
    - Map concepts to physical operations:
        - `MOMENTUM_FLUX`: `rho * Q**2 / A`
        - `TORQUE`: `F * L`
        - `SINE`: `np.sin(theta)`
        - `MULTIPLY`: `a * b`
        - `SIGN`: `np.sign(x)`
- [ ] Create `hpm_ai_v2/utils/oracle/fluid_oracle.py` with `FluidOracle`.
    - `compute_state(outputs, errors, code)`: Encodes execution results and code structure into a 20-D vector.

### 2. Experiment Script
- [ ] Create `hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler.py`.
- [ ] Implement data generation:
    - 3 training examples (Inflow, different N, L, theta).
    - 1 counter-example (Outflow).
    - 1 novel geometry test case.
- [ ] Set up HPM agent with `FluidDomainConfig`, `FluidRenderer`, and `FluidOracle`.
- [ ] Implement Phase 1: One-shot learning from Example 1.
- [ ] Implement Phase 2: Generalization to Examples 2 and 3.
- [ ] Implement Phase 3: Zero-shot test on novel geometry.
- [ ] Implement Phase 4: Output physical explanation (macro code).

## Verification & Testing
- [ ] Run the experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler.py`.
- [ ] Verify that Phase 1 finds a valid macro.
- [ ] Verify that Phase 2 confirms generalization.
- [ ] Verify that Phase 3 successfully predicts `+1` (clockwise) for the novel case.
- [ ] Check the explanation for physical correctness (inclusion of `Q^2` and `sin(theta)`).
