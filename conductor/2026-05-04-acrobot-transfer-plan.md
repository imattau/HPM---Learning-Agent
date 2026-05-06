# Acrobot Transfer (ACT) Benchmark Plan

## Objective
Implement the Acrobot Transfer benchmark to test representation scaling and transfer learning across task families (from balancing to swing-up) using the HPM core.

## Scope
1.  **Environment**: Implement a lightweight `AcrobotEnv` class based on Gym's `Acrobot-v1` dynamics.
2.  **Adapters**: Create `AcrobotStateAdapter` (handling 6D state) and `AcrobotRewardAdapter` (providing shaped reward for swing-up).
3.  **Benchmark Harness**: Implement `AcrobotBenchmark` to handle training from scratch, zero-shot transfer, fine-tuning, and catastrophic forgetting evaluation.
4.  **Experiment Script**: Create a runner script `run_acrobot_benchmark.py`.

## Implementation Steps

### 1. `AcrobotEnv` Implementation
- Create `hpm_ai_v5/planning/acrobot.py`.
- Implement `AcrobotEnv` with properties:
  - State: `[theta1, theta2, theta1_dot, theta2_dot]`
  - Actions: `[-1, 0, 1]` (torque on joint 2)
  - Runge-Kutta integration for dynamics.
- Create `AcrobotEnvConfig` for standard variants (if needed, or just default).

### 2. Adapters
- Edit `hpm_ai_v5/adapter/physics.py`.
- Add `AcrobotStateAdapter`:
  - Input: dictionary with `theta1`, `theta2`, `theta1_dot`, `theta2_dot`.
  - Output: flattened tuple `[actions..., cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot, derived_error]`.
  - The `derived_error` can be the distance from the goal state `theta1 = pi`, `theta2 = 0`.
- Add `AcrobotRewardAdapter` (or update `RewardToGoalAdapter`):
  - Provide shaped reward based on height or distance to goal. e.g. `- (theta1 - pi)^2 - 0.1 * theta1_dot^2`.

### 3. `AcrobotBenchmark` Harness
- In `hpm_ai_v5/planning/acrobot.py`.
- Similar structure to `CartpoleBenchmark`.
- Import the state and reward adapters.
- Use `ActionPolygraphGenerator` and `CartpoleForecastPostprocessor` (renamed or reused, might need to ensure it handles 3 discrete actions - wait, `CartpoleForecastPostprocessor` outputs continuous or discrete? Let's check).
- Implement `run_cross_physics_transfer` for Acrobot that takes a saved Cartpole state, fine-tunes on Acrobot, and then evaluates forgetting on Cartpole.

### 4. Runner Script
- Create `hpm_ai_v5/experiments/run_acrobot_benchmark.py` to execute the ACT benchmark and print results.

## Verification & Testing
- Add `hpm_ai_v5/tests/test_acrobot.py`.
- Test environment step and reset.
- Test adapters output correct shapes.
- Run a short benchmark to ensure no crashes.
