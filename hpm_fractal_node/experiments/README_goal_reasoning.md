# Experiment 9: Goal-Conditioned Reasoning (Agency)

**Script:** `experiment_goal_reasoning.py`

## Objective
To validate the system's transition from a passive pattern learner to an active, intent-driven agent. This experiment tests whether the structural knowledge encoded in the HFN forest is **actionable**—specifically, whether an agent can use an operational goal (e.g., "transform state A into state B") to deliberately retrieve and apply the correct transformation.

## Setup
- **Environment:** A forest populated with 50 synthetic transformation rules.
- **Ambiguity:** Every 5 rules share the same input context but result in different deltas, creating a retrieval challenge that requires intent to solve.
- **Task:** Given an input vector `A` and a target state `B`, find the rule that bridges the gap.
- **Mechanism:** Uses the `GoalConditionedRetriever` to prioritize the "Delta Slice" (B - A) during search.

## Results & Analysis
The experiment proved that introducing explicit goals effectively prunes the hypothesis space and enables efficient problem-solving.

1. **Intent-Driven Retrieval:** By conditioning retrieval on the target delta, the agent was able to identify the correct transformation rule at **Rank 1**.
2. **Actionable Structure:** The system successfully executed a `goal + input → plan → execute → evaluate` loop. It synthesized a "Plan Node" using the retrieved rule and verified that it satisfied the target state `B`.
3. **Efficiency:** The agent solved the transformation in exactly **1 step**. In contrast, a passive agent searching only by input context would have faced a 1-in-5 chance of picking the correct rule, potentially requiring multiple trial-and-error iterations.

### Key Takeaway
HFN structures are not just memories; they are **operators**. With the addition of goal-conditioning, the system moves beyond passive observation into **Agency**, using its world model to deliberately reach desired environmental states.
