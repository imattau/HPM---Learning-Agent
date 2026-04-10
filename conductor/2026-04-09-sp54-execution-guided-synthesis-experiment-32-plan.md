# SP54: Experiment 32 — Multi-Objective Scoring (Progress Bonus)

## Objective
To align the search ranking function (the distance metric) with the search guidance provided by the Retriever and empirical priors. Currently, the Euclidean distance metric is semantically myopic; it heavily penalizes intermediate programs that temporarily worsen numerical output (like creating an empty list) even if they make structural progress towards the goal. We will fix this by introducing a **progress bonus** based on the State Transition Markers (Delta Observables).

## Proposed Changes
1. **Multi-Objective Scoring:**
   Update the beam search scoring function from a simple Euclidean distance to a multi-objective score that tolerates temporary regression for future structural gain.
   
   The score will incorporate bonuses for beneficial state transitions:
   ```python
   # new_state[17]: delta_has_append
   # new_state[18]: delta_len_changed
   # new_state[19]: delta_value_changed
   
   alpha = 5.0 # Weight for delta_len_changed
   beta = 2.0  # Weight for delta_value_changed
   gamma = 3.0 # Weight for delta_has_append
   
   score = dist - (alpha * new_state[18]) - (beta * new_state[19]) - (gamma * new_state[17]) + (0.05 * len(new_path))
   ```

2. **Beam Width Reduction:**
   With aligned objectives, we should no longer need an artificially inflated beam width (150) to keep structurally sound but numerically distant paths alive. We can reduce the beam width to `50` (or even lower) to prove the search is now efficient and gradient-aligned.

## Implementation Steps
- [ ] Implement the multi-objective scoring equation in `experiment_execution_guided_synthesis.py`.
- [ ] Reduce `unique_beam` max size to 50.
- [ ] Run the experiment and verify that "Map double" is successfully and efficiently synthesized.