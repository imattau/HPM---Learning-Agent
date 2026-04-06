# Objective
Modify the Unified Cognitive Loop experiment so that the agent reaches its goal even after the environment shifts. This requires making the goal actually reachable under the new rules and allowing the agent multiple attempts to explore, fail, replan, and eventually succeed.

# Background & Motivation
In the current implementation of Experiment 17, the environmental shift changes the rule for Action 0 from "adds 1" to "adds -1" to Dimension 0. The goal requires adding 2 to Dimension 0. Under the new rules, there is literally no action that increases Dimension 0, making the goal impossible to reach. The agent correctly avoids the falsified Action 0, but it cannot reach the goal.

To truly test the "Unified Cognitive Loop," the agent needs a way to reach the goal. We will modify the shift so that Action 9 now adds 1 to Dimension 0. The agent will then need to run through a continuous loop of planning, acting, failing, exploring, and replanning until it discovers this new rule and uses it to reach the goal.

# Scope & Impact
- Modify `hpm_fractal_node/experiments/experiment_unified_cognitive_loop.py`
  - Update `PHASE 3` to add a new rule for Action 9.
  - Rewrite `PHASE 4` as a loop of `Plan -> Act -> Explore if stuck` allowing multiple attempts to reach the goal.

# Proposed Solution
1. **Phase 3 (Shift):** 
   - `env.rules[9] = 0`
   - `env.shift_rule(9, 1.0)`
2. **Phase 4 (Adaptation Loop):**
   - Create a `for` loop with `max_attempts = 10`.
   - Reset the environment.
   - Generate a plan using `agent.plan()`.
   - Execute the plan and observe outcomes. Surprise triggers falsification.
   - If the final state is within `0.1` of the goal, break and declare success.
   - If not, execute an exploration phase (random walk) to discover the newly shifted rule for Action 9.

# Implementation Steps
- [x] Edit `hpm_fractal_node/experiments/experiment_unified_cognitive_loop.py`.
- [x] Implement Phase 3 and 4 modifications.
- [x] Verify the agent reaches the goal after discovering Action 9.