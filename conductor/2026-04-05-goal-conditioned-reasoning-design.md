# SP33: Experiment 9 — Goal-Conditioned Reasoning (First Step to Agency)

## 1. Overview and Rationale
The **Goal-Conditioned Reasoning** experiment evaluates the HFN system's transition from a passive pattern learner to an active, intent-driven agent. Until now, the `Observer` has passively consumed observations (`input → explain → update`). This experiment introduces **Goals**: `goal + input → plan → execute (via decoder) → evaluate`.

By providing the system with an explicit target (e.g., "transform state A into state B"), we test whether the structural knowledge encoded in the `Forest` and `Meta-Forest` is actionable. This is the critical threshold where a "learner" becomes an "agent."

## 2. Setup & Execution
Instead of passing single states to the Observer, the agent receives an operational objective.
- **Task:** The system is presented with an input vector `A` and a target state `B`.
- **Intent-Driven Retrieval:** The system must use the `goal` to condition its retrieval, searching the `Forest` for transformation nodes (rules or relational priors) that bridge the gap between `A` and `B`.
- **Execution:** The selected transformation nodes are applied to `A` using the `Decoder` to generate a predicted state `A'`.
- **Evaluation:** The system evaluates the distance between `A'` and `B`. If the goal is not met, the system receives a penalty, prompting it to select a different hypothesis or composite transformation.

## 3. Evaluation Metrics
1. **Success Rate vs Random:** How often does the system successfully reach the goal state `B` compared to randomly selecting active nodes?
2. **Number of Steps Required:** How many planning/execution iterations does the system need to find the correct transformation to satisfy the goal?
3. **Retrieval Efficiency:** Does goal-conditioning successfully narrow the hypothesis space compared to unconditioned retrieval?

## 4. Why This Matters
*Without goals, you don't have an agent, just a learner.*
To achieve true agency, the HFN framework must prove that its learned structures (the "world model") can be deliberately queried and sequenced to solve problems and reach desired environmental states. This experiment tests whether structural self-organization (learning) produces structures that are actually useful for taking action.

## 5. Implementation Roadmap
1. **Goal-Conditioned Retriever:** Implement a search method that prioritizes nodes whose `mu` or `sigma` align with the vector delta `(B - A)`.
2. **Action/Plan Loop:** Implement a cognitive loop where the system iteratively retrieves, decodes, and evaluates progress toward the goal.
3. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_goal_reasoning.py` to run a curriculum of A → B transformations, tracking the success rate and step counts against a random baseline.
