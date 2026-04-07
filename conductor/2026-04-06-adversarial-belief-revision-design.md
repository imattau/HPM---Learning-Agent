# SP43: Experiment 19 — Adversarial Belief Revision (Truth Under Conflict)

## 1. Objective
To test whether the HFN system can successfully unlearn incorrect structural knowledge when presented with conflicting evidence, especially when the incorrect belief was initially held with high confidence.

## 2. Background & Motivation
In previous experiments (like SP37 Belief Revision and SP41 Unified Cognitive Loop), the system successfully adjusted to rule shifts. However, robust intelligence requires the ability to overcome entrenched, highly confident, and deeply ingrained adversarial priors (misleading worldviews). This experiment pushes the limits of the Observer's weight dynamics, testing if long-held "truth" (which is actually a local minimum or adversarial trap) can be systematically dismantled and replaced without causing catastrophic forgetting or endless oscillation.

## 3. Setup & Environment

### Task Definition
A sequential transition environment with an adversarial rule flip.
- **Phase 1 (Misleading World):** The environment follows a reliable rule sequence $A \rightarrow B \rightarrow C$. The agent is exposed to this repeatedly until it builds a high-confidence belief in the $B \rightarrow C$ transition.
- **Phase 2 (True World):** The environment suddenly flips to $A \rightarrow B \rightarrow D$. This new reality directly contradicts the strongly held belief in $C$.

### Implementation
1. **Rule Representation:** 
   - States and transitions are modeled as `HFN` relation nodes (e.g., `[State_t, Action, Delta]`).
   - The conflict occurs at State $B$. Action $x$ previously yielded Delta $C-B$, but now yields Delta $D-B$.
2. **Phase 1 Execution (Strong Early Bias):**
   - Inject or repeatedly expose the agent to the $A \rightarrow B \rightarrow C$ sequence for an extended number of epochs (e.g., 50-100 times) to ensure the weight of the $B \rightarrow C$ node approaches a maximum/saturation point.
3. **Phase 2 Execution (Conflict):**
   - Switch the environment to the $A \rightarrow B \rightarrow D$ sequence.
   - Continue execution and track the dynamic competition between the old belief ($C$) and the emerging truth ($D$).

## 4. Evaluation Metrics
The following metrics will be tracked dynamically during Phase 2:

1. **`belief_shift_time`**: The number of steps (or observations) it takes for the weight of the new belief ($D$) to surpass the weight of the old belief ($C$).
2. **`residual_conflict`**: The duration (in steps) during which both beliefs co-exist with relatively high weights (e.g., both weights > 0.1) before the old belief is fully suppressed.
3. **`weight_trajectory`**: A time-series log of the weights for node $C$ and node $D$. Does $C$ decay smoothly, persist stubbornly, or plummet instantly?

## 5. Potential Failure Modes
- **Confirmation Bias:** The agent's `WeightAwareRetriever` or `Observer` is so biased by the high weight of $C$ that it refuses to instantiate or reinforce $D$, effectively blinding the agent to the new reality.
- **Catastrophic Overwrite:** The system deletes or penalizes $C$ so aggressively upon the first failure that it loses stability, potentially forgetting other valid knowledge in the process.
- **Oscillation:** The system flips repeatedly between preferring $C$ and $D$, failing to converge on the new stable truth.

## 6. Definition of Success
The experiment will be considered successful if the agent demonstrates:
- **Gradual Weight Decay:** The incorrect belief ($C$) decays smoothly and predictably as contradictory evidence accumulates.
- **Stable Convergence:** The new correct structure ($D$) emerges, gains weight, and stably dominates the retrieval process.
- **Retention of Uncertainty:** The system allows $C$ and $D$ to co-exist temporarily while evidence is gathered, retaining uncertainty where appropriate for a probabilistic HFN, rather than snapping violently from one absolute truth to another.

## 7. Implementation Roadmap
- [ ] **Step 1:** Create `hpm_fractal_node/experiments/experiment_adversarial_belief_revision.py`.
- [ ] **Step 2:** Implement the two-phase environment ($A \rightarrow B \rightarrow C$ then $A \rightarrow B \rightarrow D$).
- [ ] **Step 3:** Setup the Agent and explicitly track the weights of the specific HFN nodes representing the $C$ and $D$ transitions.
- [ ] **Step 4:** Execute the loop and log the `belief_shift_time`, `residual_conflict`, and `weight_trajectory`.
- [ ] **Step 5:** Output a clear analytical report of the weight trajectories to verify gradual decay and stable convergence.