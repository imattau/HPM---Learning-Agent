# Experiment 13: Competing Hypotheses (Belief Revision)

**Script:** `experiment_belief_revision.py`

## Objective
To evaluate the HFN system's capacity for **Belief Revision**. This experiment tests whether the system can maintain multiple competing hypotheses in an ambiguous environment and autonomously switch its commitment to the correct one when presented with disambiguating evidence.

## Setup
- **Phase 1 (Ambiguity):** Observations support two orthogonal rules equally:
  - **Rule A:** "Add 2 to Input"
  - **Rule B:** "Set Output to 5"
  - (If Input is 3, both rules result in Output 5).
- **Phase 2 (Disambiguation):** Observations shift to Input 8. Rule A now predicts Output 10 (Wrong), while Rule B correctly predicts Output 5 (Right).
- **Mechanism:** We track the internal weights of Rule A and Rule B in the `Meta-Forest` as the system processes the curriculum.

## Results & Analysis
The experiment successfully demonstrated the capacity for autonomous belief correction.

1. **Uncertainty Maintenance:** During the ambiguous phase, both hypotheses were successfully maintained in the forest. The system did not prematurely collapse to a single winner.
2. **Dynamic Falsification:** Upon entering Phase 2, the weight of the previously successful Rule A was immediately and severely penalized by the `Observer`'s loss dynamics.
3. **Belief Switch:** Rule B rapidly surpassed Rule A in dominance. Rule A's weight dropped from its peak of **0.1000** to near-zero (**0.0036**), while Rule B's weight grew to **0.1764**.
4. **Resilience to Bias:** The system avoided confirmation bias, proving that its world model is plastic and guided primarily by predictive success rather than first-impression persistence.

### Key Takeaway
HFN dynamics are **Correctional**, not just accumulative. The system can "change its mind" based on evidence. By holding multiple overlapping explanation trees and using weight dynamics to prune or promote them, HFN achieves an adaptive balance between stability and plasticity.
