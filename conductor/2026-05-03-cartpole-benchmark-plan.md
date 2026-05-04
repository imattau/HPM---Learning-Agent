# Physics Benchmark Proposal: Cartpole Plan

## Objective
Implement a Cartpole-style physics benchmark to test the HPM AI core's ability to learn continuous control from low-dimensional states, handle delayed consequences, and demonstrate sample efficiency and transfer, all without modifying the v5 core.

## Key Components

### 1. New Adapters (`hpm_ai_v5/adapter/physics.py` and `hpm_ai_v5/postprocessors/numeric.py`)
- **CartpoleStateAdapter**: Preprocessor that converts an observation dictionary (`position`, `velocity`, `angle`, `angular_velocity`) into a flattened tuple of 5 floats `(pos, vel, sin(angle), cos(angle), ang_vel)`. Uses `math.sin` and `math.cos`.
- **RunningNormaliserAdapter**: Preprocessor that normalises the tuple so each dimension has zero mean and unit variance, utilizing `sklearn.preprocessing.StandardScaler` (incremental fitting via `partial_fit`) to avoid custom running mean/variance logic.
- **RewardToGoalAdapter**: Preprocessor that reads episodic reward from the packet context, computes a discounted running total, and injects it as `goal["utility"]`.
- **MultiNumericPostprocessor**: Postprocessor (in `hpm_ai_v5/postprocessors/numeric.py`) that uses `numpy.clip` to validate and clamp a scalar action to `[-1, 1]` (or a specified range) before returning.

### 2. RecentBufferAdapter Updates
- Existing `RecentBufferAdapter` may need minor tweaks or can be used as-is, provided it can handle a tuple state and flatten it, or we can ensure the pipeline feeds it correctly.

### 3. Benchmark Harness (`hpm_ai_v5/planning/cartpole.py`)
- A standalone mock environment simulating the `dm_control` cartpole physics or a direct `gymnasium`-style step function (to ensure the benchmark runs without heavy external dependencies like MuJoCo).
- **TerminalChecker Logic**: Agent-side control loop that detects episode termination, resets history/accumulator adapters, but preserves the `PatternManager` store.
- Evaluates if the agent can reach an average episode length of > 500 steps over 50 episodes.

## Constraints
- **Core Isolation**: The `hpm_ai_v5/core` module MUST NOT be modified.
- **KISS & Minimal Changes**: Use standard math/numpy for the mock environment. Do not over-engineer the physics engine; a simple Euler integration cartpole is sufficient to test the reasoning.

## Implementation Steps
1. Add `RunningNormaliserAdapter` and `RewardToGoalAdapter` in `hpm_ai_v5/adapter/physics.py`.
2. Add `CartpoleStateAdapter` in `hpm_ai_v5/adapter/physics.py`.
3. Add `MultiNumericPostprocessor` in `hpm_ai_v5/postprocessors/numeric.py`.
4. Create the `CartpoleBenchmark` in `hpm_ai_v5/planning/cartpole.py` with a lightweight internal simulation of the cartpole mechanics to avoid external dependencies.
5. Create tests in `hpm_ai_v5/tests/test_cartpole.py` to verify the adapters and the benchmark.