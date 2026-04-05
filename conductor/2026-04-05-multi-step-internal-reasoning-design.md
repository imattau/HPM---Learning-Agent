# SP34: Experiment 10 — Multi-Step Internal Reasoning (Chain-of-Thought)

## 1. Overview and Rationale
The **Multi-Step Internal Reasoning** experiment evaluates the HFN system's capacity for sustained, stateful internal reasoning—the continuous equivalent of "Chain-of-Thought." Instead of solving a problem in a single `input → explain/predict` step, the system faces a problem requiring a sequential chain of transformations: `A → B → C → D`.

By iteratively feeding the `Observer`'s output (predicted state) back into itself as the new input state, we test whether the structural knowledge encoded in the `Forest` can support multi-step composition without collapsing into noise or entering infinite oscillatory loops.

## 2. Setup & Execution
- **Curriculum:** The `Forest` is seeded with "atomic" transformation rules (e.g., `+1`, `*2`, `reverse`).
- **Task:** The system is given an initial state `A` and a final target state `D` (e.g., `A=1`, `D=4` requiring `1 -> 2 -> 4` via `*2` twice or `+1` three times). Critically, no single rule in the forest can achieve `A → D` directly.
- **Iterative Loop:**
  ```python
  state = input
  for step in range(k):
      # Goal-conditioned retrieval + decoding
      state = agent.plan_and_execute(state, target=D)
      if state == D:
          break
  ```
- **Observation:** We track the intermediate states (`B`, `C`) to see if the structure stabilizes across steps, converges toward the target `D`, or drifts away into meaningless noise.

## 3. Evaluation Metrics
1. **Convergence vs. Drift:** Does the Euclidean distance to the target `D` decrease monotonically, or does the sequence diverge/drift?
2. **Loop Stability:** Does the system fall into oscillatory failure modes (e.g., applying `+1` then `-1` repeatedly)?
3. **Step Efficiency:** How many internal steps are required to reach the target compared to the theoretical minimum path?

## 4. Why This Matters
*Can HFN sustain internal stateful reasoning loops?*
True reasoning requires holding and manipulating intermediate states over time. If the system's representations degrade when fed back into themselves (a common issue in continuous latent spaces), it cannot plan ahead. Proving stability across `k` steps demonstrates that HFNs can act as robust, composable operators for complex, multi-hop problem solving.

## 5. Implementation Roadmap
1. **Synthetic State Space:** Define a simple continuous state space (e.g., 10D vectors representing quantities or positions).
2. **Atomic Rules:** Populate the forest with rules that induce known, fixed translations in the state space.
3. **Multi-Step Engine:** Implement the iterative loop using the `GoalConditionedRetriever` and `Decoder`.
4. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_multi_step_reasoning.py` to run the task, tracking and logging the trajectory of the intermediate states to detect convergence, drift, or oscillation.
