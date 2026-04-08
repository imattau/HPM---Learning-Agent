# Experiment 23: Structural Refinement (Self-Debugging)

## Objective
To validate the HFN architecture's capacity for **Iterative Engineering**. The system must demonstrate the ability to perform local graph edits (splicing and patching) to fix logical errors in an existing compositional program graph.

## Background: "Iterative Refinement"
In previous stages, the agent would discard an entire program if it produced the wrong output. This experiment introduces **Causal Fault Localization**: the agent calculates the "Residual Delta" between its output and the goal, identifies the semantic locus of failure within the HFN tree, and splices in a corrective patch node.

## Setup
- **Domain**: Semantic Program Space.
- **State Representation**: 6D vector (Value, Returned, Length, TargetVal, Iterator, Init).
- **The Base Abstraction**: `map_add_one` (7-step iterative loop).
- **Perturbed Tasks**:
    - `map_add_two`: Requires substituting/splicing an extra `OP_ADD`.
    - `map_add_one_and_double`: Requires splicing an `OP_MUL2` node.

## Results
The agent successfully demonstrated self-debugging:
- **Fault Localization**: For `map_add_two`, the agent detected a residual delta of `+1.0` and correctly targeted the `OP_ADD` node for patching.
- **Topological Splicing**: The system successfully spliced `prior_rule_OP_ADD` into the nested loop structure without breaking syntax or indentation.
- **Multi-Stage Repair**: For `map_add_one_and_double`, the agent spliced an `OP_MUL2` node, transforming the output from `[2, 3]` to `[4, 6]` in a single iterative step.
- **Efficiency**: Repairing an existing 7-leaf tree took only **1 retrieval cycle**, compared to the thousands of nodes explored during a full from-scratch search.

## Metrics Summary
| Task | Initial Result | Residual Delta | Patch Applied | Final Result | Status |
|---|---|---|---|---|---|
| `map_add_two` | `[2, 3]` | `+1.0` | `OP_ADD` | `[3, 4]` | **SUCCESS** |
| `map_add_one_and_double` | `[2, 3]` | `+2.0` | `OP_MUL2` | `[4, 6]` | **SUCCESS** |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_structural_refinement.py
```
