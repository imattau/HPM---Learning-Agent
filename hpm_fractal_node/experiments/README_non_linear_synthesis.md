# Experiment 24: Non-Linear Program Synthesis (Logic Forks)

## Objective
To validate the HFN architecture's ability to construct and render **non-linear graphs** containing conditional branches (if/else logic). This completes the core suite of structured programming primitives: Sequence, Iteration, and Selection.

## Background: "Selection Logic"
True programmatic reasoning requires the ability to fork execution based on a predicate. This experiment introduces **Predicate Nodes** (e.g., `COND_IS_EVEN`) and **Alternate Paths** (`BLOCK_ELSE`). The system must learn to nest these branches correctly within existing loop structures.

## Setup
- **Domain**: Semantic Program Space.
- **State Representation**: 7D vector (Value, Returned, Length, TargetVal, Iterator, Init, Condition).
- **The Conditional Primitives**:
    - `COND_IS_EVEN`: Renders `if val % 2 == 0:` and increments indentation.
    - `BLOCK_ELSE`: Renders `else:` and manages scope dedent/indent.
- **The Tasks**:
    - `filter_even`: `[1, 2, 3, 4] -> [2, 4]`. Requires a conditional filter inside a loop.
    - `map_even_double_else_keep`: `[1, 2, 3, 4] -> [1, 4, 3, 8]`. Requires a full if/else logic fork.

## Results
The experiment demonstrated the robustness of the HFN structural decoding:
- **Turing-Completeness**: By mastering Sequence, Iteration, and Selection, the HFN latent space is now functionally capable of representing any computable algorithm.
- **Recursive Indentation**: The `CodeRenderer` successfully managed nested indentation levels for conditional blocks inside loops without manual formatting hints.
- **Semantic Fidelity**: The rendered non-linear structures executed perfectly in the Python sandbox, producing the correct algorithmic behavior for both filtering and conditional mapping.

## Metrics Summary
| Task | Logical Type | Complexity | Status |
|---|---|---|---|
| `filter_even` | Conditional Filter | 7 Primitives | **SUCCESS** |
| `map_even_double_else_keep` | Logic Fork (If/Else) | 10 Primitives | **SUCCESS** |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_non_linear_synthesis.py
```
