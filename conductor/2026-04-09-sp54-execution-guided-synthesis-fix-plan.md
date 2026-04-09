# SP54: Experiment 30 Fix — State Aliasing and Learning

## Objective
Address the critical limitations identified in the Execution-Guided Synthesis experiment, specifically:
1. **State Aliasing:** Intermediate structural steps (like creating a loop or an empty list) produce identical empirical states, causing the beam search to discard valid partial programs prematurely.
2. **Disabled Learning:** The `observer.observe()` call was previously commented out for speed, meaning the agent wasn't actually learning empirical priors online.

## Changes
1. **Expand Empirical State (10D -> 14D):** Update `S_DIM = 14`.
2. **Add Structural Markers (Option C):** Modify `EmpiricalOracle.compute_state` to accept the rendered `code` string and extract structural markers into the 4 new state dimensions:
   - `s[10] = 1.0 if 'for ' in code else 0.0`
   - `s[11] = 1.0 if '.append(' in code else 0.0`
   - `s[12] = 1.0 if 'if ' in code else 0.0`
   - `s[13] = 1.0 if '+=' in code or '*=' in code or '-=' in code else 0.0`
3. **Update Planner Distance Metric:** Assign a weight of `0.0` to these new structural markers in the planner's distance calculation. This ensures the agent is not directly penalized for adding a loop when the goal state doesn't explicitly require it. However, because they are part of the state vector, they will automatically **break state aliasing** during beam search deduplication, preserving diverse structural paths.
4. **Re-enable Online Learning:** Uncomment `self.observer.observe(vec)` in the beam search loop so the `TieredForest` learns the semantic effects of the concepts during the execution search.

## Verification
- Run `experiment_execution_guided_synthesis.py` and verify that the "Map double" task successfully synthesizes in 7-9 steps, proving that breaking state aliasing allows deep structural discovery.