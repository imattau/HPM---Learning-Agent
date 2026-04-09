# SP54: Experiment 33 — HPM-Native Planning Objective

## Objective
To resolve the fundamental architectural misalignment identified in the execution-guided synthesis pipeline. Currently, the system features a highly faithful HPM core (`HFN`, `Forest`, `Observer`, `Evaluator`) but utilizes a non-HPM, deterministic Euclidean distance metric for search ranking. This experiment will refactor the synthesis planner to use native HPM Evaluator dynamics, transforming the hybrid system into a fully coherent, end-to-end HPM agent.

## Background & Rationale
As noted in the architectural review:
- **World 1 (HPM Core):** Probabilistic, hierarchical, evaluator-driven, compositional.
- **World 2 (Current Planner):** Deterministic execution, Euclidean scoring, hard beam pruning.

To align World 2 with World 1, the search objective must transition from minimizing raw distance (`score = ||output - target||`) to maximizing the utility of the pattern hypothesis as defined by the framework:
`Utility = Accuracy (log_prob) - Complexity (description length) + Coherence (structural progress)`

## Proposed Changes
1. **Integrate HPM Evaluator:**
   The `ExecutionGuidedAgent` will instantiate the native HPM `Evaluator` (Layer 3).
2. **Probabilistic Scoring (`log_prob`):**
   Instead of calculating Euclidean distance between the empirical state of the outputs and the goal state, the planner will treat the candidate AST as a generative pattern. Accuracy will be measured using a probabilistic likelihood of the target outputs given the AST's empirical outputs.
3. **Multi-Factor Utility:**
   The final beam search score will be driven by the `Evaluator`, natively balancing:
   - **Epistemic:** `log_prob(target | empirical_output)`
   - **Resource/Affective:** Complexity penalty based on AST depth (`len(path)`).
   - **Coherence:** The structural progress transition markers (`delta_has_append`, etc.) will be formalized as coherence bonuses.

## Implementation Steps
- [ ] Import and instantiate the `Evaluator` within `ExecutionGuidedAgent`.
- [ ] Refactor the scoring block inside the `plan` method's beam search.
- [ ] Wrap the candidate path's empirical state and transitions into an ephemeral `HFN` representation so it can be evaluated by the standard `Evaluator`.
- [ ] Run the benchmark tasks ("Add one", "Add two", "Map double") to verify that HPM-native scoring successfully guides compositional synthesis.