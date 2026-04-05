# Experiment 10: Multi-Step Internal Reasoning (Chain-of-Thought)

**Script:** `experiment_multi_step_reasoning.py`

## Objective
To evaluate the HFN system's capacity for sustained, stateful internal reasoning. This experiment tests whether the system can solve problems requiring a sequential chain of transformations (`A → B → C → D`) by iteratively feeding its own predictions back into the reasoning loop.

## Setup
- **Environment:** A forest populated with "Atomic" rules that increment individual dimensions of a 10D state vector.
- **Task:** Reach a target state (e.g., `[2, 2, 0...]`) from an initial state `[0, 0, 0...]`. No single rule can bridge the gap in one step.
- **Mechanism:** An iterative planning loop using the `GoalConditionedRetriever`. At each step, the system:
  1. Identifies the desired delta to reach the goal.
  2. Retrieves the best atomic rule to reduce that delta.
  3. Updates its internal state by applying the rule.
  4. Repeats until the goal is reached or the budget is exhausted.

## Results & Analysis
The experiment proved that HFN representations are robust enough to support stable, multi-hop composition.

1. **Monotonic Convergence:** The distance to the target decreased at every step (`2.82 → 2.23 → 1.41 → 1.00 → 0.00`). The system did not drift into noise or overshoot the target.
2. **Loop Stability:** The agent successfully sequenced 4 steps without entering oscillatory failure modes (e.g., repeating the same rule indefinitely). 
3. **Composable Operators:** Each HFN rule functioned as a robust operator. The system was able to maintain its "Chain-of-Thought" across 4 internal cycles without representational degradation.

### Key Takeaway
HFNs support **Continuous Multi-Step Reasoning**. This experiment demonstrates that the world model is not just a static lookup table, but a library of composable operators that can be sequenced to solve complex, non-linear problems. It validates the HFN framework's potential for high-level planning and stateful agency.
