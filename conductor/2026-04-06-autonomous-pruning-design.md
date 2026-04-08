# SP46: Experiment 22 — Autonomous Graph Pruning (Simulation Dreams)

## 1. Objective
To validate the HFN architecture's ability to autonomously refine its structural priors and compositional chunks through an internal "dreaming" phase. The system must proactively compose abstract program graphs, render and execute them in a sandbox, and use the execution feedback (success, TypeError, SyntaxError) to penalize invalid topological connections *before* encountering them in a real goal-directed task.

## 2. Background & Motivation
In Experiment 21 (SP45), the agent successfully built deeply nested compositional graphs (e.g., a 7-step map operation). However, the planning process was heavily reliant on either exhaustive DFS search or guided scaffolding (curiosity) upon failure. Furthermore, the agent only learned that a structure was invalid (e.g., attempting to iterate over an integer) when a specific curriculum task failed.

True sovereign intelligence requires **proactive consolidation and pruning**. During "downtime" (dreams), the agent should explore the combinatorial space of its structural primitives, execute those combinations, and map the causal boundaries of its knowledge. If `COMPOSE(FOR_LOOP, CONST_1)` results in a `TypeError` (because `CONST_1` produces a scalar, not an iterable), the agent should permanently penalize that specific edge in the HFN, drastically reducing the search space for future planning.

## 3. Setup & Environment

### Domain: Semantic Program Space with Sandboxed Execution
- **State Representation**: We maintain the 6D semantic vector `[AccumulatorValue, ReturnedFlag, ListLength, ListTargetValue, IteratorActive, ListInitFlag]` from SP45.
- **Action Space**: The same structural primitives (`CONST_1`, `VAR_INP`, `OP_ADD`, `LIST_INIT`, `FOR_LOOP`, `ITEM_ACCESS`, `LIST_APPEND`, `RETURN`).

### The "Dream" Phase
Before any goal-directed tasks are presented, the agent undergoes a `dream()` phase:
1.  **Generative Sampling**: The agent randomly samples pairs or triplets of concepts from its current active nodes (priors and existing chunks).
2.  **Graph Construction**: It composes them into temporary HFN trees (e.g., `COMPOSE(OP_ADD, LIST_APPEND)`).
3.  **Sandbox Execution**: The `CodeRenderer` renders the tree, and the `PythonExecutor` runs it with a set of dummy inputs (e.g., `[1]`, `[[1, 2]]`).
4.  **Causal Feedback (Pruning)**:
    -   If the execution raises an exception (e.g., `TypeError`, `SyntaxError`), the Observer applies a heavy penalty to the newly created composite node or the specific directional edge between the two concepts.
    -   If the execution succeeds (returns a valid state without crashing), the composite node is given a slight boost or simply retained as a valid structural possibility.

### The Curriculum
After the dream phase, the agent is tested on a zero-shot composition task:
-   **Task**: `filter_even` (or a similar composition like `map_add_one`).
-   **The Critical Test**: Because the agent has autonomously pruned invalid paths (like trying to `LIST_APPEND` to an integer) during its dreams, the DFS planner should find the correct, deeply nested solution significantly faster (exploring fewer nodes) compared to an agent without the dream phase.

## 4. Architectural Enhancements

### 1. Autonomous Evaluator
The `TaskRunner` is extended to handle sandbox execution without a defined `expected_output`. The `Evaluator` simply checks for runtime stability (lack of exceptions) and state progression (did the state change?).

### 2. Topological Penalties
Currently, penalties are applied to node weights. In this experiment, we need to ensure that the planner respects the pruned paths. If `COMPOSE(A, B)` was penalized during a dream, the planner should immediately discard `B` as a candidate if the current path ends in `A`.

## 5. Evaluation Metrics
1.  **Dream Pruning Rate**: The percentage of generated random compositions that are identified as invalid and penalized.
2.  **Planning Efficiency (Nodes Explored)**: Compare the number of nodes explored during the capstone task between a "Dreaming Agent" and a "Baseline Agent" (no dreams). The dreaming agent should explore drastically fewer nodes due to a pre-pruned search space.
3.  **Zero-Shot Success Rate**: The ability to solve a complex compositional task on the first attempt without guided scaffolding, relying entirely on the clean structural boundaries mapped during dreams.

## 6. Failure Modes to Watch
- **Over-Pruning (Catastrophic Forgetting)**: The agent randomly generates a valid composition but provides invalid dummy inputs, causing a runtime error that incorrectly penalizes a valid structural concept. (Solution: robust dummy input generation).
- **Combinatorial Explosion in Dreams**: The agent tries to compose too many deep trees during downtime, leading to an infinite dream loop. (Solution: limit dreams to depth 2 or 3 compositions).

## 7. Implementation Steps
- [ ] **Step 1**: Create `hpm_fractal_node/experiments/experiment_autonomous_pruning.py`.
- [ ] **Step 2**: Implement the `dream(self, n_dreams)` method in `DevelopmentalAgent`.
- [ ] **Step 3**: Enhance `PythonExecutor` to return detailed execution statuses (Success, TypeError, SyntaxError).
- [ ] **Step 4**: Update the `plan` method to strongly avoid composing sequences that have a penalized composite node in the forest.
- [ ] **Step 5**: Run the experiment, comparing the planning efficiency (nodes explored) with and without the dream phase.