# Implementation Plan: SP43 — Adversarial Belief Revision (Truth Under Conflict)

## Objective
Implement Experiment 19 to test the HFN's ability to unlearn an entrenched, high-confidence but incorrect belief when faced with contradictory evidence.

## Implementation Steps

### Step 1: Initialize Experiment File
- [x] **Task**: Create `hpm_fractal_node/experiments/experiment_adversarial_belief_revision.py`.
- [x] **Task**: Add standard imports (`numpy`, `HFN`, `Observer`, etc.) and the `AdversarialEnvironment` class.
- [x] **Conformity Check**: Ensure imports and class structure align with previous HFN experiments.

### Step 2: Implement Adversarial Environment & Agent Logic
- [x] **Task**: Implement `AdversarialEnvironment` with `Phase 1 (A->B->C)` and `Phase 2 (A->B->D)`.
- [x] **Task**: Implement `BeliefRevisionAgent` which wraps an `Observer` and provides targeted weight tracking for specific "Belief" nodes.
- [x] **Conformity Check**: Verify the environment correctly switches transitions based on a `phase` flag.

### Step 3: Phase 1: Entrenchment (Strong Bias)
- [x] **Task**: Run 50 observations of `A->B->C` to ensure the "C" belief node is high-weight and entrenched.
- [x] **Conformity Check**: Log weight of Node C to verify it is significantly high (> 0.5).

### Step 4: Phase 2: Conflict & Metric Collection
- [x] **Task**: Switch to Phase 2 (`A->B->D`).
- [x] **Task**: Run until Node D's weight exceeds Node C's weight, or for a maximum of 100 steps.
- [x] **Task**: Collect `belief_shift_time`, `residual_conflict`, and `weight_trajectory`.
- [x] **Conformity Check**: Ensure the `Observer`'s surprise detection triggers falsification of the old belief.

### Step 5: Final Analysis & Documentation
- [x] **Task**: Generate a formatted report of the results.
- [x] **Task**: Create `hpm_fractal_node/experiments/README_adversarial_belief_revision.md`.
- [x] **Task**: Update root `README.md` and `hpm_fractal_node/experiments/README.md`.
- [x] **Conformity Check**: Verify all specified failure modes and success criteria are addressed in the report.

## Verification & Testing
- [x] Run the experiment script.
- [x] Verify that Node C weight decays while Node D weight grows.
- [x] Confirm `belief_shift_time` is recorded.
