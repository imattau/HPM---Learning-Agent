# SP44: Experiment 20 — Developmental Cognitive System (HFN)

## 1. Objective
To validate whether the HFN architecture can build a **stable, reusable, and expanding knowledge structure** over time through curriculum learning. 

Specifically, the experiment tests if the system:
- **Learns faster over time** (Acceleration).
- **Reuses structure across tasks** (Compositionality).
- **Retains old knowledge** (No Catastrophic Forgetting).
- **Adapts to new domains** (Transfer Learning).
- **Remains structurally stable** (No Node Explosion).

## 2. Background & Motivation
Previous experiments have validated specific mechanisms (belief revision, multi-step planning, gap-filling). This "Developmental" experiment tests the **lifelong learning** hypothesis of HPM. True general intelligence must not just solve isolated tasks but accumulate a structural foundation where learning new, complex concepts becomes easier because they are composed of previously mastered simpler ones.

## 3. System Architecture & Setup

### Core Components (Existing)
- **HFN**: Represents the structural knowledge. **Persistent across runs** to build cumulative priors.
- **Observer**: Manages the learning loop (weight updates, compression, falsification).
- **Meta-HFN**: Tracks state and performance for selection bias.
- **Decoder/Executor**: Translates plans into executable actions.

### New Scaffolding (Minimal Additions)
- **Task Loader**: Parses standardized YAML/JSON task files for curriculum and exams.
- **Task Runner**: Executes individual tasks and interacts with the Observer.
- **Task Selector**: Chooses the next curriculum task based on success, novelty, and overlap.
- **Exam Runner**: Periodically evaluates the agent on a held-out set of retention and transfer tasks.
- **Metrics Logger**: Continuously tracks learning speed, node counts, reuse rates, and exam performance.
- **Persistence Manager**: Saves and loads the HFN Forest and Observer state to/from disk (`data/developmental_forest.*`).

## 4. Task Data Structure (Standardized File Format)
Tasks are loaded from a curriculum file (e.g., `hpm_fractal_node/experiments/tasks/developmental_curriculum.yaml`).

```yaml
id: reverse_list_basic
type: curriculum   # or exam
goal: "reverse a list"
input:
  - [1,2,3]
expected_output:
  - [3,2,1]
tags: [list, loop, transform]
difficulty: 2
variants:
  - input: []
    expected_output: []
  - input: [1]
    expected_output: [1]
dependencies: [iteration, indexing]
```

## 5. Curriculum & Exams

### Curriculum Phases (Evolving)
- **Phase 1 (Child)**: `return constant`, `add numbers`, `access list element`.
- **Phase 2 (Basic)**: `sum list`, `find max`, `filter even`.
- **Phase 3 (Intermediate)**: `reverse list`, `nested loops`, `conditionals`.
- **Phase 4 (Advanced)**: `map/filter composition`, `grouping`, `stateful transforms`.
- **Phase 5 (Adult)**: `multi-step transformations`, `open-ended goals`.
*Key Rule*: Tasks are appended over time; the environment grows, it does not reset.

### Periodic Exams (e.g., every 100 steps)
- **Retention**: Re-test early curriculum tasks (e.g., `sum list`).
- **Transfer**: Apply learned logic to new structures (e.g., `filter nested list`).
- **Perturbation**: Test edge cases of known tasks.
- **Interference**: Switch domains entirely (e.g., from `lists` to `strings` or `trees`).

## 6. Core Runtime Loop
The main loop orchestrates learning and evaluation:
1. **Select Task**: Use `select_task(task_pool, state)` biased toward near-misses and novelty.
2. **Run Task**: Generate plan, decode, execute, evaluate, and `observer.update(eval)`.
3. **Log Metrics**: Record attempts to solve, nodes used, and forest size.
4. **Exams**: If `step % EXAM_INTERVAL == 0`, freeze learning and run the Exam Suite.

## 7. Evaluation Metrics (Continuous Logging)
1. **Learning Speed (`attempts_to_solve`)**: Expected to decrease over time as priors get richer.
2. **Structural Reuse (`reused_nodes / total_nodes`)**: Expected to increase over time (>60-70%).
3. **Node Growth (`node_count`)**: Expected to grow quickly early on, then plateau/stabilize (compression dominates).
4. **Transfer Performance (`exam_success_rate`)**: Expected to increase across phases.
5. **Retention**: Performance on old tasks must remain stable (no drift).
6. **Efficiency (`new_nodes_per_task`)**: Expected to decrease as the system relies more on existing structure.

## 8. Success vs Failure Criteria

### Success (Evidence of General Knowledge Accumulation)
- Learning accelerates for later phases.
- Reuse dominates (>60%).
- Node growth stabilizes.
- Transfer improves without retention degrading.

### Failure Modes (Watch Carefully)
- **Memorization**: High success but low reuse (creating a unique node for every exact task).
- **Explosion**: Node count grows linearly/exponentially with no consolidation.
- **Drift**: Old tasks fail later in the curriculum (catastrophic forgetting).
- **Fragility**: Perturbation exams fail completely.