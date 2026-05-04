# Cartpole SWA Integration Plan

## Objective
Integrate the `ScoringWeightAdaptationAgent` (SWA) into the `CartpoleBenchmark` to dynamically learn the optimal weighting for the Pattern Engine's `alpha` (accuracy), `beta` (density), `gamma` (context), and `delta` (utility) metrics. This replaces the hardcoded goal weights and allows the core to adapt its settings to the high-dimensional continuous physics domain.

## Key Components

### 1. Initialize SWA Agent
In `CartpoleBenchmark.__init__`:
- Instantiate `ScoringWeightAdaptationAgent(learning_rate=0.1, exploration_rate=0.15)`.

### 2. Dynamically Inject Weights
In the `CartpoleBenchmark.run` step loop:
- Retrieve current weights using `self.swa.current_weights("cartpole")`.
- Merge these weights with static goal overrides (like `plan_horizon: 3`, `sequence_execution: True`).
- Pass the merged goal into `self.pipeline.step(obs, goal=goal, context=context)`.

### 3. Observe Rewards
After `pipeline.step` and `env.step`:
- Retrieve the selected pattern from `result.action.selected_pattern`.
- Retrieve the preprocessed state from `result.input.state`.
- Calculate an immediate "advantage" or just use the immediate reward (1.0 for survival, 0.0 for failure) combined with the TD error penalty if needed. Alternatively, just pass the raw environment `reward` to `self.swa.observe_reward("cartpole", pattern, state, reward)`.
- If SWA expects `pattern` and `state`, we use the ones returned by the pipeline.

## Constraints
- Keep changes localized to `hpm_ai_v5/planning/cartpole.py`.

## Implementation Steps
1. Import `ScoringWeightAdaptationAgent` in `cartpole.py`.
2. Add SWA initialization to `CartpoleBenchmark`.
3. Update the `goal` construction in the run loop to pull from SWA.
4. Add a `self.swa.observe_reward` call at the end of each step.
5. Add `explore=True` logic to SWA by adding random perturbations to the weights if needed, or just let SWA's observe method handle it. Wait, `ScoringWeightAdaptationAgent` has `_perturb` but it's private and only used in `select()`. Since we bypass `select()` to use the pipeline, we can just manually perturb weights with epsilon decay if we want, or call `observe_reward` to let it learn.
