# SP54: Experiment 31 — State Transition Markers (Delta Observables)

## Objective
To address the brittleness and lack of generalisation in Execution-Guided Synthesis (Experiment 30). While structural markers successfully broke state aliasing, they led to "hacky" overfitting (e.g., relying on uninitialized variables that happened to match the first test case). We will introduce **State Transition Markers (Delta Observables)** to provide directionality and a true gradient, without hardcoding structural priors.

## Proposed Changes
1. **Robust Test Cases:**
   Add more rigorous and diverse test cases to the "Map double" task (e.g., `[3, 5] -> [6, 10]`) to penalize brittle programs that overfit to specific initial values (like `val = 0`).
2. **Extend State Vector (Delta Features):**
   Increase `S_DIM` to capture state transitions between the parent AST and the new AST. Instead of just static structural presence, we will explicitly measure:
   - `delta_len_changed`: Did the length of the output list change compared to the previous step?
   - `delta_value_changed`: Did the numerical values change?
   - `delta_type_changed`: Did the output type change?
3. **Oracle & Planner Updates:**
   The planner will compute the empirical state of the current path and append boolean/continuous transition markers representing the change from the parent state. This gives the beam search a sense of "momentum" or directionality.

## Implementation Steps
- [ ] Update the `tasks` dictionary in the experiment script with robust validation inputs.
- [ ] Modify `EmpiricalOracle.compute_state` or the planner's transition logic to calculate and append delta markers.
- [ ] Adjust the distance weights to incorporate transition markers.
- [ ] Run the experiment to verify it discovers a clean, generalizable compositional structure without state fragmentation.