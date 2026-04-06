# SP42: Experiment 18 — Long-Horizon Goal Reasoning (Depth Test)

## 1. Objective
To evaluate the stability and scalability of the HFN agent's multi-step planning and retrieval mechanisms when tasked with increasingly distant goals. The core question is: **Does goal-conditioned reasoning remain coherent and stable beyond short chains, or does it succumb to drift, looping, or combinatorial explosion?**

## 2. Background & Motivation
In Experiment 17 (SP41), the Unified Cognitive Loop successfully demonstrated short-chain reasoning (e.g., sequences of 3-4 actions) and adaptation to environmental shifts. However, robust agency requires planning over long horizons. This experiment isolates the reasoning depth variable, introducing deliberate "dead ends" and misleading priors to test the agent's ability to maintain trajectory focus, prune unpromising paths, and avoid cycles as the required plan length scales up to 20 steps.

## 3. Setup & Environment

### Task Definition
A synthetic graph-based state space where states are N-dimensional vectors. Moving between states requires applying specific transition rules.
- **Start State:** $A$
- **Goal State:** $F$ (or scaling targets like $Goal_3, Goal_5, Goal_{10}, Goal_{20}$)
- **Valid Path:** A linear or near-linear dependency chain of specific state transformations ($A \rightarrow B \rightarrow C \dots \rightarrow Goal$).
- **Distractors:** Promising but ultimately invalid paths that diverge from the main chain (e.g., $A \rightarrow X \rightarrow Y \rightarrow Z$ [Dead End]).

### Implementation
1. **Rule Representation:** Transition rules (both valid and misleading) are encoded as `HFN` relation nodes (e.g., `[State_t, Action, Delta]`).
2. **Prior Injection:**
   - **Correct Priors:** The valid steps necessary to reach the goal.
   - **Misleading Priors:** "Distractor" rules that seem geometrically promising (e.g., a rule that immediately reduces distance to the goal but leads to a state with no further valid outward transitions).
3. **Execution:**
   - Run the agent on tasks requiring increasing sequence depths: `target_depths = [3, 5, 10, 20]`.
   - The agent uses the `WeightAwareRetriever` to select rules at each step.

## 4. Evaluation Metrics
For each target depth condition, the following metrics will be tracked per episode:

1. **`success_rate(depth)`**: Percentage of episodes where the agent successfully reaches the goal state within an allowed step budget (e.g., `depth + 5`).
2. **`average_steps_to_solution`**: For successful episodes, how many steps did the plan actually take? (Measures efficiency vs optimality).
3. **`branching_factor` (nodes explored)**: The average number of candidate nodes retrieved and considered per step during the planning phase.
4. **`drift_score`**: A measure of how far the agent's intermediate simulated states deviated from the optimal valid path trajectory before returning to it (or failing).

## 5. Potential Failure Modes
Tracking these failure modes will highlight specific architectural weaknesses in the Retriever or Planner:
- **Early Collapse:** The agent rigidly sticks to the very first promising path it finds, even if it hits a dead end, failing to backtrack or explore alternatives.
- **Branch Explosion:** The agent's attention becomes overly diffuse. It considers too many distractors without aggressively pruning, leading to computational paralysis or exceeding the step limit.
- **Looping:** The agent selects a sequence of rules that form a cycle (e.g., $A \rightarrow B \rightarrow A \rightarrow B \dots$), getting trapped in a local geometric minimum.
- **Drift:** The agent loses the goal trajectory entirely, wandering into state space regions that look locally optimal but are globally disconnected from the target.

## 6. Definition of Success
The experiment will be considered successful if the agent demonstrates:
- **Graceful Degradation:** The `success_rate` decreases predictably and smoothly as depth increases (e.g., 100% at depth 3, >80% at depth 10, >50% at depth 20), rather than hitting a hard failure wall.
- **Stable Pruning:** The `branching_factor` and search time scale linearly or sub-linearly with depth, proving that the `WeightAwareRetriever` effectively suppresses distractors and maintains focus on the goal trajectory.

## 7. Implementation Roadmap
- [ ] **Step 1:** Create `hpm_fractal_node/experiments/experiment_long_horizon_reasoning.py`.
- [ ] **Step 2:** Implement the `GraphEnvironment` capable of generating valid chains of varying lengths and injecting dead-end distractor branches.
- [ ] **Step 3:** Inject a controlled prior library of rules (HFN nodes) into the agent's forest before execution.
- [ ] **Step 4:** Adapt the `UnifiedAgent.plan` method to support basic backtracking or alternative exploration if the simulated path hits a dead end.
- [ ] **Step 5:** Run the evaluation loop over `max_steps = [3, 5, 10, 20]` and compile the metrics report.