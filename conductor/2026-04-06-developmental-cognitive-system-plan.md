# Implementation Plan: SP44 — Developmental Cognitive System (HFN)

## Objective
Implement Experiment 20 to test whether the HFN system builds a stable, reusable, expanding knowledge structure over a prolonged curriculum.

## Implementation Steps

### Step 1: Standardized Task Loader & Persistence
- **Task**: Implement `TaskLoader` to parse YAML curriculum files into `Task` objects.
- **Task**: Create the initial `developmental_curriculum.yaml` with the standardized format.
- **Task**: Implement `PersistenceManager` using HFN's existing snapshot/export capabilities to ensure the Forest and Observer state are maintained across runs.
- **Conformity Check**: Verify that the system can stop and resume without losing learned nodes or weights.

### Step 2: Implement Task Selection & Execution Scaffolding
- **Task**: Implement the `TaskSelector` class with `select_task(task_pool, state)` focusing on near-misses, novelty, and overlap.
- **Task**: Implement `TaskRunner` to handle `run_task(task)` logic: generating a plan, executing it, evaluating the result, and calling `observer.update(eval)`.
- **Conformity Check**: Verify the `TaskSelector` uses minimal logic to bias toward high-value learning opportunities without being an engine itself.

### Step 3: Implement Metrics Logger
- **Task**: Create a `MetricsLogger` class to continuously track:
  - `attempts_to_solve[task_id]`
  - `reused_nodes` vs `total_nodes` per task
  - Total `node_count` over time
  - Success rates per exam suite.
- **Conformity Check**: Ensure all critical metrics (learning speed, structural reuse, node growth, transfer, retention, efficiency) are logged.

### Step 4: Implement Core Runtime Loop
- **Task**: Construct the main `run_experiment` loop that orchestrates the `TaskSelector`, `TaskRunner`, `MetricsLogger`, and periodic `ExamRunner`.
- **Task**: Execute the loop over `TOTAL_STEPS`.
- **Conformity Check**: Confirm that tasks are appended over time (environment grows, not resets) and exams run periodically without mutating the forest.

### Step 5: Final Analysis & Documentation
- **Task**: Analyze the logged metrics to determine if the experiment reached the Success Criteria (learning acceleration, high reuse, stable node growth, strong retention/transfer).
- **Task**: Generate a formatted performance summary output.
- **Task**: Create `hpm_fractal_node/experiments/README_developmental_cognitive_system.md`.
- **Task**: Update root `README.md` and `hpm_fractal_node/experiments/README.md`.

## Verification & Testing
- Run the experiment.
- Ensure the system doesn't explode in node count (avoid memorization failure).
- Validate that transfer performance on exams increases across phases.
