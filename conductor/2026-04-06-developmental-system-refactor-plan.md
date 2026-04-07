# Plan: Developmental Cognitive System Refactor

## Objective
Refactor Experiment 20 to align with the "Rendering from Structure" feedback. The agent should construct program structures in semantic space, which are then rendered into Python code for execution.

## 1. Structural semantic state
- Move from raw VOCAB indices to a 2D semantic state: `[AccumulatorValue, ReturnedFlag]`.
- This ensures the HFN is reasoning about *program state* rather than *token sequences*.

## 2. Code Renderer
- Implement `CodeRenderer.render(structure)` which takes a sequence of semantic concepts (e.g., `["CONST_1", "RETURN"]`) and produces executable Python code (`"return 1"`).
- This decouples structural reasoning from syntactic generation.

## 3. Structural Priors
- Define priors as high-level transitions:
    - `CONST_1`: Action that adds 1 to `AccumulatorValue`.
    - `RETURN`: Action that sets `ReturnedFlag` to 1.
- Use a high scale for Delta (e.g., 50.0) to ensure these signals dominate retrieval in high-dimensional HFN space.

## 4. Enhanced Task Runner
- Update `TaskRunner` to use the `Renderer` and `PythonExecutor`.
- Implement "Guided Curiosity" during exploration: if the agent fails, it is shown the correct 2-step semantic sequence (`CONST_X` -> `RETURN`) to reinforce the structure.

## 5. Verification
- Run the experiment and verify that `return_constant` is solved within 1-3 steps.
- Verify that the plan stabilizes to `['CONST_1', 'RETURN']` or equivalent.
- Verify that node growth is controlled and represents reusable structure.

## Implementation Steps
- [ ] Refactor `hpm_fractal_node/experiments/experiment_developmental_cognitive_system.py`.
- [ ] Implement `CodeRenderer` and `PythonExecutor`.
- [ ] Update `DevelopmentalAgent` planning and prior injection.
- [ ] Run and verify results.
