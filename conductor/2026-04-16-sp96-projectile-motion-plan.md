# Plan: SP96 — Noisy Projectile Motion

## Objective
Test HPM’s ability to discover a quadratic structure ($t^2$) from noisy, partially observed projectile motion data and distinguish it from linear motion (Structural Sensitivity).

## Key Components
- **Domain**: `ProjectileDomainConfig` (theta, v, t, g) and `ProjectileRenderer`.
- **Data Generator**: Gaussian noise ($\sigma=0.05$), 20% missing inputs.
- **Goal**: $y = v \sin(\theta) t - 0.5 g t^2$.

## Implementation Steps

### 1. Domain & Infrastructure
- [ ] Create `hpm_ai_v2/domains/projectile_domain.py` with `PROJECTILE_CONCEPTS`.
- [ ] Create `hpm_ai_v2/domains/projectile_renderer.py` for stack-based physics rendering.
- [ ] Create `hpm_ai_v2/utils/oracle/projectile_oracle.py` to extract structural flags (e.g., presence of `t**2`).

### 2. Experiment Script
- [ ] Implement `hpm_ai_v2/experiments/experiment_sp96_projectile_motion.py`.
- [ ] **Phase 1: Discovery**
    - [ ] Generate noisy, incomplete data.
    - [ ] Solve for $y = f(v, \theta, t, g)$ using **BFS** (max_depth=6).
    - [ ] Extract macro(s) for $t^2$.
- [ ] **Phase 2: Compression & Transfer**
    - [ ] Ensure the agent reuses the learned $t^2$ macro for new trajectories (different $v, \theta$).
    - [ ] **Greedy** strategy preferred; **BFS** allowed as fallback to verify macro inclusion in search space.
- [ ] **Phase 3: Structural Sensitivity**
    - [ ] **Test A (New Surface)**: Change units/scale. Verify macro robustness.
    - [ ] **Test B (New Structure)**: Linear motion $y = vt$. Verify old macro failure and re-discovery via **BFS**.

## Verification & Testing
- [ ] Run the experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp96_projectile_motion.py`.
- [ ] **Success Metric**: Phase 1 RMSE < 0.1, Macro contains `t**2`.
- [ ] **Success Metric**: Phase 2 demonstrates transfer (exact/greedy/bfs-with-macro).
- [ ] **Success Metric**: Phase 3B correctly rejects the quadratic macro and learns the linear model.
