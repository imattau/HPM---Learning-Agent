# Plan: Autonomous Graph Pruning (Simulation Dreams)

## Objective
Implement Experiment 22 to validate the HFN's ability to autonomously prune its structural search space through sandboxed "dreams" and causal feedback.

## Implementation Steps

### Step 1: Initialize Experiment File
- **Task**: Create `hpm_fractal_node/experiments/experiment_autonomous_pruning.py` based on `experiment_recursive_scaling.py`.
- **Task**: Implement enhanced `PythonExecutor` that returns (result, error_type).
- **Conformity Check**: Ensure imports and core classes are properly set up.

### Step 2: Implement Autonomous Dreaming
- **Task**: Add `dream(self, n_dreams)` to `DevelopmentalAgent`.
- **Task**: Implement generative sampling of structural concepts and sandbox execution.
- **Task**: Apply penalties to nodes/edges that result in `TypeError` or `SyntaxError`.
- **Conformity Check**: Verify that "bad dreams" result in weight penalties for the offending nodes.

### Step 3: Implement Topological Pruning in Planner
- **Task**: Update `plan()` to check weights before following a branch. If a composite node exists with a very low weight, discourage that path.
- **Conformity Check**: Verify the planner bypasses known "invalid" structural combinations discovered during dreams.

### Step 4: Execution & Baseline Comparison
- **Task**: Run a "Baseline" agent (0 dreams) on a composition task (e.g., `map_add_one`).
- **Task**: Run a "Dreaming" agent (e.g., 50 dreams) on the same task.
- **Task**: Compare `nodes_explored` and `planning_time`.
- **Conformity Check**: Verify all metrics (Pruning Rate, Efficiency gain) are captured.

### Step 5: Final Analysis & Documentation
- **Task**: Create `hpm_fractal_node/experiments/README_autonomous_pruning.md`.
- **Task**: Update project-wide READMEs.
- **Conformity Check**: Address Success/Failure criteria from the design.

## Verification & Testing
- Automated run of the comparison loop.
- Assert that dreaming agent explores fewer nodes than baseline.
