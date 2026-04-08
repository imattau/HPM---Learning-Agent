# Experiment 21: Recursive Complexity Scaling (Algorithmic Curriculum)

## Objective
Validate whether the HFN architecture can scale from simple, flat compositions to **deeply nested, recursive abstractions** capable of solving complex algorithmic tasks involving loops, lists, and conditional logic.

## Background: "Nested Abstract Synthesis"
This experiment builds on the compositional graphs of SP44. It challenges the agent to move from scalar operations to list manipulation. The agent must discover a complex 7-step iterative loop (`LIST_INIT -> VAR_INP -> FOR_LOOP -> ITEM_ACCESS -> OP_ADD -> LIST_APPEND -> RETURN`) and chunk it into a reusable abstraction to solve the `map_add_one` task.

## Setup
- **Domain**: Semantic Program Space.
- **State Representation**: 6D vector `[AccumulatorValue, ReturnedFlag, ListLength, ListTargetValue, IteratorActive, ListInitFlag]`.
- **Action Space**: Structural primitives (`CONST_1`, `VAR_INP`, `OP_ADD`, `LIST_INIT`, `FOR_LOOP`, `ITEM_ACCESS`, `LIST_APPEND`, `RETURN`).
- **Curriculum**: 
    - `return_empty_list`: Master basic list initialization.
    - `add_one_to_item`: Master scalar transformation.
    - `map_add_one`: **Capstone**: Apply a transformation over a list (Iterative Map).

## Results
The agent successfully demonstrated deep recursive scaling:
- **Algorithmic Chunking**: The system successfully discovered and bundled the 7-step iterative map operation into a massive composite node: `compose(compose(compose(compose(compose(compose(LIST_INIT+VAR_INP)+FOR_LOOP)+ITEM_ACCESS)+OP_ADD)+LIST_APPEND)+RETURN)`.
- **Single-Step Retrieval**: By Attempt 13, the agent stopped searching the massive combinatorial space and instead retrieved this single, high-weight chunk, dropping the number of attempts from 5 to 1.
- **Contextual Rendering**: The `CodeRenderer` successfully managed local variable scopes and scoped indentation for the `FOR_LOOP`, dynamically unrolling the nested structure into executable Python.
- **Curriculum Mastery**: Achieved 100% success rate on the complex algorithmic task after guided scaffolding, proving that deep abstractions can stabilize execution.

## Metrics Summary
| Task | Complexity | Abstraction Discovered | Success Rate (Final) |
|---|---|---|---|
| `return_empty_list` | 2 steps | `compose(LIST_INIT+RETURN)` | 100% |
| `add_one_to_item` | 3 steps | - | 100% |
| `map_add_one` | 7 steps | `compose(..+FOR_LOOP+..+LIST_APPEND+..)` | 100% |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_recursive_scaling.py
```