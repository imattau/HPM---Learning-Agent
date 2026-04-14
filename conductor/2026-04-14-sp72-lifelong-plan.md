# SP72: Lifelong Learning with Auto-Observation & Auto-Save Plan

## Objective
Implement and validate Experiment SP72 to demonstrate lifelong learning capabilities using the newly integrated auto-observation and auto-save features. The experiment will verify that an agent can retain learned patterns across simulated crashes and solve compositional tasks efficiently.

## Key Files & Context
- `hpm_ai_v2/experiments/experiment_sp72_lifelong.py` (New): The experiment script that will test the agent's lifelong learning capabilities.

## Implementation Steps

### 1. Script Setup
**Target:** `hpm_ai_v2/experiments/experiment_sp72_lifelong.py`
- Initialize `ListDomainConfig` and `BaseHFNAgent`.
- Configure agent with `auto_observe_frequency = 5`, `auto_save_frequency = 3`, `use_density_tracker = True`, and `replay_buffer_size = 100`.
- Create a helper `encode_input(inp)` to format list inputs for the agent's observation loop.

### 2. Phase 1: Training Tasks
- Present three training tasks (`MAP_add1`, `MAP_mul2`, `FILTER_pos`) sequentially.
- For each task, call `agent.observe_example(encoded_input)` followed by `agent.solve(inputs, outputs)`.
- Assert that each task is solved successfully.

### 3. Phase 2: Simulated Crash & Reload
- Delete the current agent instance but retain the `cold_dir`.
- Create a new `BaseHFNAgent` pointing to the same `cold_dir`.
- Call `agent.load_state()` to restore prior learning.
- Re-run the three training tasks and assert they are solved successfully without further training.

### 4. Phase 3: Composition Task
- Present the compositional task `MAP_add1_then_mul2` with inputs `[[1, 2, 3]]` and expected outputs `[[4, 6, 8]]`.
- Call `agent.solve()` and verify that the task succeeds.
- Check the agent's solve history (`agent.meta.history`) to ensure the task was solved with `depth <= 2`, indicating the use of compressed macros rather than deep search.

### 5. Phase 4: Control & Size Comparison
- Run the identical training sequence with a control agent initialized with `auto_observe_frequency = 0`.
- Compare the size of the reloaded agent's forest (`len(agent.forest)`) against the control agent's forest.
- Assert or verify that the auto-observing agent's forest size is reduced by at least 20% compared to the control agent due to background compression and absorption.

## Verification & Testing
- Execute the script using `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp72_lifelong.py`.
- Confirm all 4 phases pass successfully and all assertions hold.
