# Plan: Within-Pattern Learning via Online EM

**Goal:** Implement rigorous online EM (Baum-Welch) for hierarchical patterns to allow individual patterns to adapt their parameters to the data stream.

## Architecture
Each `HierarchicalPattern` will maintain sufficient statistics (expected counts) for its transition and emission matrices. These statistics will be updated incrementally with a decay factor (forgetting) whenever `observe()` is called.

## Changes

### 1. `hpm_ai_v4/pattern.py`
- [ ] Add `SS_A3`, `SS_A32`, `SS_A21`, `SS_B` and `statistics_decay` to `HierarchicalPattern.__init__`.
- [ ] Implement `_forward_backward(obs_seq)` to compute joint posteriors over all latent levels.
- [ ] Refactor `observe(obs, learning_rate)` to:
    - Update sufficient statistics using the posterior inferred from the new observation.
    - Re-estimate `A3`, `A32`, `A21`, and `B` by normalizing the statistics.
- [ ] Update `HierarchicalPattern.flat` to ensure its `observe` remains consistent with the new statistical approach (though a simple nudge is already a form of online EM).

### 2. `hpm_ai_v4/agents/agent.py`
- [ ] (Verification) Ensure `HPMAgent.perceive_and_learn` correctly calls `p.observe(obs)` before updating utilities. (Currently it does).

## Verification & Testing
- [ ] Create `hpm_ai_v4/tests/test_learning_dynamics.py` to verify:
    - A pattern's parameters converge toward the true dynamics of a fixed environment.
    - Patterns with within-pattern learning outperform static patterns in predictive accuracy.
- [ ] Run all existing HPM AI v4 tests to ensure no regressions in framework predictions.
