# Cartpole Performance Enhancements Plan

## Objective
Implement missing adapters and agent-side configurations to improve the Cartpole benchmark performance from ~45 steps to the target of >500 steps. This involves adding dense error signals, action history, temporal difference error, exploration, and multi-step planning horizons.

## Key Components

### 1. Action History & Derived Error (Preprocessor)
Modify the existing `CartpoleStateAdapter` in `hpm_ai_v5/adapter/physics.py`:
- Add a buffer for the last `N` (e.g., 3) actions.
- Compute the derived stability error: `error = angle^2 + 0.1 * angular_velocity^2`.
- Append both the action history and the derived error to the flattened state tuple. This converts the problem into a short-term memory task and provides a smooth gradient towards stability.

### 2. Temporal Difference (TD) Error Adapter (Preprocessor)
Add `TDErrorAdapter` to `hpm_ai_v5/adapter/physics.py`:
- Reads the previous forecasted state and the current actual state.
- Computes the squared error per dimension.
- Injects a negative utility signal (`-alpha * error`) into the goal, punishing the core for poor predictions.

### 3. Exploration Adapter (Postprocessor)
Add `ExplorationPostprocessor` to `hpm_ai_v5/postprocessors/numeric.py`:
- Maintains an epsilon ($\epsilon$) value that decays over episodes.
- With probability $\epsilon$, adds small Gaussian noise to the numeric action before clipping.
- Helps the core break out of suboptimal limit cycles (e.g., 45-step oscillations).

### 4. Horizon Configurator (Agent)
Update the `CartpoleBenchmark` loop in `hpm_ai_v5/planning/cartpole.py`:
- Pass the previous forecast back into the context for the `TDErrorAdapter`.
- Change the goal configuration to request a longer planning horizon (e.g., `plan_horizon=3`).
- Manage the $\epsilon$ decay for the `ExplorationPostprocessor`.

## Constraints
- **Core Isolation**: The `hpm_ai_v5/core` module MUST NOT be modified.
- **KISS Principles**: Implement these enhancements directly within the existing adapter framework using minimal math.

## Implementation Steps
1. Update `CartpoleStateAdapter` to include derived error and a multi-step action buffer.
2. Implement `TDErrorAdapter`.
3. Implement `ExplorationPostprocessor`.
4. Update `CartpoleBenchmark` to wire the new adapters, maintain state (previous forecast, action buffer), set horizon=3, and manage $\epsilon$ decay.
5. Run tests and the benchmark to verify >500 average steps.
