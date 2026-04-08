# SP45: Experiment 21 — Recursive Complexity Scaling (Algorithmic Curriculum)

## 1. Objective
To validate whether the HFN architecture can scale from simple, flat compositions (e.g., `[CONST_1] -> [OP_ADD]`) to **deeply nested, recursive abstractions** capable of solving complex algorithmic tasks involving loops, lists, and conditional logic.

## 2. Background & Motivation
In Experiment 20 (SP44), we proved that the HFN can build true compositional program graphs (trees) and render them into executable Python code. However, the tasks were limited to basic scalar arithmetic (`return 1`, `return 2`). To achieve Artificial General Intelligence (AGI) levels of reasoning, the system must handle **Recursive Complexity Scaling**:
*   Using a learned "Chunk" (e.g., `increment_by_1`) as a single primitive child within a larger "Chunk" (e.g., `map(increment_by_1, list)`).
*   Planning over deep execution traces (e.g., 20+ operations) by retrieving a shallow graph of high-level abstractions (e.g., 3-4 nodes).

This experiment will transition the agent into an **Algorithmic Curriculum** focused on list manipulation and iterative transformations.

## 3. Setup & Environment

### Domain: Semantic Program Space (Extended)
- **State Representation**: Expanded from 2D to an N-Dimensional semantic tensor capable of representing lists, accumulators, iterators, and flags.
    - Example: `[List_State_Hash, Iterator_Index, Accumulator_Value, ReturnedFlag]`
- **Structural Primitives (CONCEPTS)**:
    - `VAR_INP`: Load input variable (now a list).
    - `LIST_INIT`: Initialize an empty list (`res = []`).
    - `FOR_LOOP`: Begin iteration over an iterable (`for item in x:`).
    - `ITEM_ACCESS`: Access the current iteration item.
    - `OP_ADD`: Add to value (e.g., `item + 1`).
    - `LIST_APPEND`: Append to list (`res.append(item)`).
    - `RETURN`: Return the current accumulator/list.

### The Algorithmic Curriculum
The curriculum will introduce tasks that force recursive chunking:
1.  **Task 1 (Base)**: `return_empty_list` (Goal: `[]` -> `[]`). Requires `LIST_INIT -> RETURN`.
2.  **Task 2 (Iteration)**: `return_input_list` (Goal: `[1, 2]` -> `[1, 2]`). Requires `VAR_INP -> RETURN` or a simple loop copy.
3.  **Task 3 (Transform)**: `add_one_to_item` (Goal: `1` -> `2`). Requires `VAR_INP -> OP_ADD -> RETURN`.
4.  **Task 4 (Composition / The Capstone)**: `map_add_one` (Goal: `[1, 2]` -> `[2, 3]`). 
    *   **The Critical Test**: The agent must retrieve the chunk learned in Task 3 (`add_one_to_item`) and embed it as the body of a `FOR_LOOP` chunk to solve Task 4 efficiently.

## 4. Architectural Enhancements

### 1. Contextual Rendering (The Renderer Update)
The `CodeRenderer` must be updated to handle **Scope and Indentation** for control flow blocks (`FOR_LOOP`). 
*   When rendering a `FOR_LOOP` node, its children must be indented as the loop body.
*   Example: `COMPOSE(FOR_LOOP, COMPOSE(OP_ADD, LIST_APPEND))` ->
    ```python
    res = []
    for item in inp:
        val = item + 1
        res.append(val)
    return res
    ```

### 2. Deep Chunking (The Observer Update)
When a task involving a loop succeeds, the entire loop body (which may already be a composite node) and the loop construct itself must be bundled into a new `Higher-Order Chunk` (e.g., `map_transform_chunk`).

### 3. Causal Feedback (Negative Reinforcement)
Introduce strict pruning. If a rendered code block fails with a `SyntaxError` or `TypeError` (e.g., trying to `OP_ADD` a list to an integer), the system must immediately penalize the specific composite node that generated that invalid structure, forcing the planner to backtrack and explore typologically safe structures.

## 5. Evaluation Metrics
1.  **Planning Depth vs. Execution Depth**: We expect the number of *planned nodes* (HFN depth) to remain small (< 5 steps) even when the rendered Python code executes dozens of operations (due to loops).
2.  **Abstraction Reuse Rate**: Track how frequently the `add_one_to_item` chunk is retrieved to solve the `map_add_one` task.
3.  **Curriculum Velocity**: The number of attempts required to solve Task 4 should decrease significantly if the abstractions from Tasks 1-3 are successfully retrieved and composed.

## 6. Failure Modes to Watch
- **Flat Loop Unrolling**: The agent tries to solve `[1, 2] -> [2, 3]` by planning `OP_ADD -> OP_ADD` instead of discovering the `FOR_LOOP` abstraction. (Solution: Penalize plans that scale linearly with input size).
- **Scope Leakage**: The Renderer fails to properly nest the loop body, leading to syntax errors.
- **Type Mismatching**: The agent tries to append to an integer or iterate over a constant. (Solution: Ensure the semantic state vectors capture basic type information).

## 7. Implementation Steps
- [ ] **Step 1**: Expand the `CONCEPTS` dictionary and update the semantic state dimensions (`S_DIM`) in the agent to handle lists and iterators.
- [ ] **Step 2**: Upgrade `CodeRenderer` to support scoped blocks (indentation for `FOR_LOOP`).
- [ ] **Step 3**: Define the new tasks (`developmental_curriculum_lists.json`).
- [ ] **Step 4**: Implement the Causal Feedback loop (penalizing type/syntax errors from the `PythonExecutor`).
- [ ] **Step 5**: Run the experiment and verify that the agent successfully composes a loop over a previously learned transform chunk.