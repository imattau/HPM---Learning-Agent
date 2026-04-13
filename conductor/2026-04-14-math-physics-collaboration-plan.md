# Collaborative Math‑Physics Problem Solving Experiment Plan

## Objective
Implement and run the Collaborative Math-Physics Problem Solving experiment to demonstrate that two specialized agents (Math and Physics) can collaborate via social exchange to solve compound problems that neither can solve alone. This will validate HPM §9.5 (social pattern field convergence), HPM §9.7 (institutional scaffolding), and the newly added `SequentialCompositionMixin`.

## Implementation Roadmap

### 1. Enhance Perceptual Ops and Renderer
**Files:** `hpm_ai_v2/utils/state.py`, `hpm_ai_v2/utils/renderer.py`, `hpm_ai_v2/agents/base_agent.py`
- Add new primitive operations to the environment: `square` ($x^2$), `sqrt` ($\sqrt{x}$), `div2` ($x/2$).
- Update `ASTRenderer` to include `import math` in the generated code so that `math.sqrt` can be used.
- Update `extract_perceptual_ops` or `_inject_blank_priors` to make these operations available to the agents.

### 2. Create the Experiment Script
**File:** `hpm_ai_v2/experiments/experiment_math_physics.py`
- **Phase 0 (Setup):** Create a shared `SocialForest`. Instantiate `MathAgent` and `PhysicsAgent` using `SocialAnalogicalAgent`. Seed them with the new primitive ops.
- **Phase 1 (Individual Specialisation):**
  - Train `MathAgent` on: `solve_linear`, `quadratic_roots`, `derivative_poly`.
  - Train `PhysicsAgent` on: `kinetic_energy`, `force_to_accel`, `distance_from_accel`.
  - *Note: To ensure reliable and fast learning via BFS, we will seed these macros directly as if they were learned (or provide highly constrained scaffolds).*
- **Phase 2 (Social Exchange):** Agents broadcast their macros to the `SocialForest` and import peers' macros.
- **Phase 3 (Collaborative Problem Solving):** 
  - Test both agents on a compound task `distance_from_force_mass_time` (requires `force_to_accel` then `distance_from_accel`).
  - Verify that they use `SequentialCompositionMixin` to generate a novel composite macro successfully.
- **Phase 4 (Optional):** Parameter learning validation on the `quadratic_roots` task using a `GaussianMixtureModel`.

### 3. Verification & Metrics
- Run `PYTHONPATH=. python3 hpm_ai_v2/experiments/experiment_math_physics.py`.
- Verify Phase 1: 100% accuracy on held-out examples.
- Verify Phase 2: Each agent has macros from the other domain.
- Verify Phase 3: 100% success on the compound task with 0 oracle calls during composition (depth $\le$ 3).

## Next Steps
Upon approval, I will begin implementing the new primitive operations and the experiment script according to this roadmap.