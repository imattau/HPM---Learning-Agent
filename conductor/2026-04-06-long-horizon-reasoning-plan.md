# Plan: SP42 — Long-Horizon Goal Reasoning (Depth Test)

## Objective
Implement Experiment 18 to test reasoning stability over long horizons (up to 20 steps) with distractors and misleading priors.

## Implementation Steps

### Step 1: Initialize Experiment File
- [x] **Task**: Create `hpm_fractal_node/experiments/experiment_long_horizon_reasoning.py` with basic imports and structure.
- [x] **Conformity Check**: Ensure imports align with HFN core library.

### Step 2: Implement `GraphEnvironment`
- [x] **Task**: Create a class that generates linear state chains and diverging dead-end branches.
- [x] **Task**: Implement `reset()` and `step(action_id)`.
- [x] **Conformity Check**: Ensure state transitions are vector-based and compatible with HFN [State, Action, Delta] encoding.

### Step 3: Rule Injection & Agent Setup
- [x] **Task**: Implement a function to generate `HFN` nodes for each valid transition and for misleading "shortcut" distractors.
- [x] **Task**: Setup `UnifiedAgent` with `WeightAwareRetriever` (using `dist / weight` penalty).
- [x] **Conformity Check**: Verify `target_weight` and `weight_penalty` allow the retriever to prioritize trusted rules over geometric shortcuts that lead to dead ends.

### Step 4: Advanced Planning with Backtracking
- [x] **Task**: Enhance the `plan` method to handle "Dead Ends". If no rule from the current state leads closer to the goal without hitting a dead end (simulated), backtrack or select next-best candidate.
- [x] **Conformity Check**: Align with "Stable Pruning" and "Graceful Degradation" goals.

### Step 5: Execution & Metrics
- [x] **Task**: Loop through `depths = [3, 5, 10, 20]`.
- [x] **Task**: Collect `success_rate`, `average_steps`, `branching_factor`, and `drift_score`.
- [x] **Task**: Output a formatted report.
- [x] **Conformity Check**: Verify all metrics specified in the design are captured.

## Verification & Testing
- [x] Run the script and verify successful goal reach for depth 3 and 5 initially.
- [x] Verify that the agent avoids known dead ends.
- [x] Achieved 100% success on 20-step chains.
