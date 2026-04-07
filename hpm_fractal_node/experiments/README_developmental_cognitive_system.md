# Experiment 20: Developmental Cognitive System (HFN)

## Objective
Validate the integration of structured knowledge accumulation over an evolving curriculum. The experiment tests if the HFN agent can build **true compositional program graphs** (nested trees) in semantic space and render them into executable code.

## Background: "Code is Rendered from Structure"
This experiment implements the core HPM philosophy: **intelligence lies in building nested structure, not generating strings.** 
1. **HFN**: Acts as a "compiler front-end" that constructs a **nested binary tree** of semantic operations via hierarchical folding.
2. **CodeRenderer**: A recursive tree-walker that translates structural concepts into valid Python syntax sequentially.
3. **Chunking**: Successfully executed plans are bundled into new, permanent composite nodes. These "chunks" become reusable priors, turning multi-step reasoning into single-step retrieval.

## Setup
- **Domain**: Semantic Program Space.
- **State Representation**: 2D vector `[AccumulatorValue, ReturnedFlag]`.
- **Action Space**: Structural primitives (`CONST_1`, `CONST_5`, `VAR_INP`, `OP_ADD`, `RETURN`).
- **Curriculum**: 
    - `return_constant_1`: Master base constants.
    - `return_constant_5`: Generalize constants.
    - `return_input`: Master variables.
    - `return_2`: Master composition (`CONST_1` + `OP_ADD` + `RETURN`).

## Results
The agent successfully demonstrated hierarchical developmental growth:
- **Hierarchical Folding**: Transitioned from flat sequences to nested trees (e.g., `((CONST_1 -> OP_ADD) -> RETURN)`).
- **Abstraction Discovery**: Created 4+ reusable "Chunks" (composite nodes) during the curriculum.
- **Mastery**: Achieved 100% success across the curriculum, including multi-step composition.
- **Persistence**: Built a stable, expanding prior library that reduced planning depth for future tasks.

## Metrics Summary
| Step | Task | Success | Abstraction Discovered |
|---|---|---|---|
| 1 | `return_constant_1` | True | `compose(CONST_1+RETURN)` |
| 10 | `return_2` | True | (Compositional Mastery) |
| 14 | `return_input` | True | `compose(VAR_INP+RETURN)` |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_developmental_cognitive_system.py
```
