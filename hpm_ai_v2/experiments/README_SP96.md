# [SP96] Noisy Projectile Motion — Structure vs Surface

## Objective
Demonstrate HPM's ability to discover robust physical invariants from noisy, continuous data and successfully transfer those structures across varying environmental parameters while remaining sensitive to underlying structural changes.

## Physical Domain
- **Task**: Modeling the vertical displacement ($y$) of a projectile.
- **True Invariant**: $y = v_0 \sin(\theta) t - 0.5 g t^2$.
- **Simplified Target**: $y = 0.5 g t^2$ (Vertical Drop).
- **Complexity**: Inputs include Gaussian noise ($\sigma=0.05$) and missing data points, representing realistic sensor observations.

## Experimental Phases

### 1. Discovery from Noisy Data (Vertical Drop)
- **Goal**: Extract the quadratic relationship $0.5 g t^2$ from 20 noisy trajectories.
- **Process**: The agent uses an upgraded BFS search guided by a **Scientific Oracle** that notices correlations with $t^2$.
- **Outcome**: Successfully discovered and registered a reusable **`macro_drop`**.

### 2. Robustness to Surface Change (Scaling)
- **Goal**: Use the learned macro in a new environment with doubled gravity ($g=19.6$).
- **Process**: The agent attempts to solve the scaled task.
- **Outcome**: Successfully reused the `macro_drop` at Depth 1, proving that HPM learns the **algebraic structure** (independent of specific parameter values) rather than just fitting numbers.

### 3. Structural Sensitivity (Linear Motion)
- **Goal**: Model a system where the physics have changed to linear motion ($y = vt$).
- **Process**: The agent attempts to apply the `macro_drop`.
- **Outcome**: The agent **rejected the old macro** due to collapsed utility and successfully discovered the new linear structure, proving it is sensitive to the underlying physical "topology."

## Key HPM Mechanisms

### Hybrid Noise Tolerance
The agent uses a dual-matching logic: `abs_diff < tolerance * (1.0 + abs(e))`. This allows for high absolute accuracy at small values while maintaining relative tolerance at large scales, essential for continuous physical domains.

### Oracle-Driven "Cognitive Drive"
The beam search is guided by the **ProjectileOracle**, which computes the empirical correlation between agent outputs and key physical variables ($t, v_0, \theta$). This provides a "Scientific Insight" that directs the search toward programs with correct physical dependencies even before a perfect match is found.

### Robust Structural Matching
Discovery success is determined by **percentage-based matching** (e.g., 80% point match), ensuring that statistical outliers in noisy data do not derail the learning of robust invariants.

## Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp96_projectile_motion.py
```
