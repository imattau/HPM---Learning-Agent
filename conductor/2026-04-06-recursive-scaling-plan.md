# Plan: Recursive Complexity Scaling (Algorithmic Curriculum)

## Objective
Implement Experiment 21 to validate that the HFN architecture can scale to deeply nested, recursive abstractions by solving algorithmic tasks like `map_add_one`.

## 1. Expand Semantic State and Concepts
- **Concepts**: Add `LIST_INIT`, `FOR_LOOP`, `ITEM_ACCESS`, `LIST_APPEND`.
- **State Vector (S_DIM)**: Expand to 4D: `[Accumulator_Value, ReturnedFlag, List_State_Hash, Iterator_Active]`.
    - `List_State_Hash`: Simplified representation of list content/length.
    - `Iterator_Active`: Binary flag indicating if we are inside a loop.

## 2. Contextual Code Renderer
- Refactor `CodeRenderer` to maintain an `indent_level`.
- `FOR_LOOP` renders the header and increments `indent_level` for subsequent children.
- Handle terminal `RETURN` properly within or outside scope.

## 3. Algorithmic Curriculum
- Create `hpm_fractal_node/experiments/tasks/developmental_curriculum_lists.json` with the tasks:
    - `return_empty_list`
    - `add_one_to_item` (scalar)
    - `map_add_one` (the composition test)

## 4. Higher-Order Chunking & Causal Feedback
- Update `TaskRunner` to provide strong reinforcement for `compose` nodes that successfully bridge the "Loop -> Body -> Append" gap.
- Implement strict penalties for `SyntaxError` or `TypeError` using `observer.penalize_id`.

## 5. Verification
- Run the experiment and confirm the agent retrieves the `add_one_to_item` chunk when solving `map_add_one`.
- Verify that the final knowledge structure contains a deeply nested abstraction representing the map operation.

## Implementation Steps
- [x] Create `hpm_fractal_node/experiments/tasks/developmental_curriculum_lists.json`.
- [x] Refactor `hpm_fractal_node/experiments/experiment_developmental_cognitive_system.py` into a new file `experiment_recursive_scaling.py` to avoid breaking Experiment 20.
- [x] Implement the expanded state and scoped renderer.
- [x] Run and verify results.
