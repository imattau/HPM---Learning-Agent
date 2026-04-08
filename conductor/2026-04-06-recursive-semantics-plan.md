# Plan: Refine State Semantics for Recursive Scaling

## Objective
Update the semantic state vector (`S_DIM`) in `experiment_recursive_scaling.py` to provide higher resolution during planning. The goal is to ensure the agent doesn't mistake returning an empty list (`[]`) for returning a populated list (`[2, 3]`), forcing it to learn the `map_add_one` abstraction.

## 1. Expand Semantic State Space
- **Current `S_DIM = 4`**: `[AccumulatorValue, ReturnedFlag, ListStateFlag, IteratorActive]`
- **New `S_DIM = 5`**: `[AccumulatorValue, ReturnedFlag, ListLength, ListTargetValue, IteratorActive]`
    - `ListLength`: `0.0` for `[]`, `1.0` or `len(list)` for populated lists.
    - `ListTargetValue`: A proxy for the "content" of the list. For `[2, 3]`, this might be the sum `5.0` or average `2.5`. We'll use the target transformed value (e.g., `2.0` if `add_one_to_item` expects `2`).

## 2. Refine Priors and Transitions
- `LIST_INIT`: Sets `ListLength = 0.0`, `ListTargetValue = 0.0`.
- `VAR_INP`: Loads the input value or list. If it's a list, set `ListLength = len(inp)`.
- `FOR_LOOP`: Sets `IteratorActive = 1.0`.
- `ITEM_ACCESS`: Sets `AccumulatorValue` to the current item's value.
- `OP_ADD`: Increments `AccumulatorValue`.
- `LIST_APPEND`: Increments `ListLength`, sets `ListTargetValue` to `AccumulatorValue`.
- `RETURN`: Sets `ReturnedFlag = 1.0`.

## 3. Update `TaskRunner` and `CodeRenderer`
- **Goal State Generation**:
    - For `return_empty_list`: `[0.0, 1.0, 0.0, 0.0, 0.0]`
    - For `add_one_to_item` (1 -> 2): `[2.0, 1.0, 0.0, 0.0, 0.0]`
    - For `map_add_one` ([1] -> [2]): `[2.0, 1.0, 1.0, 2.0, 0.0]` (Assuming length 1 for simplicity in tracing).

## 4. Fix Redundant Folding
- Implement a check during `solve` to prevent composing identical `RETURN` chains or empty `LIST_INIT` loops if they don't progress the semantic state toward the goal.

## Implementation Steps
- [ ] Read `experiment_recursive_scaling.py` to identify required modification points.
- [ ] Update `S_DIM` to 5 and adjust index references.
- [ ] Modify `_inject_priors` with the new delta mappings.
- [ ] Update the `solve` simulation logic in `plan`.
- [ ] Update `run_task` goal definition and exploration logic.
- [ ] Test the script.
