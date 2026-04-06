# Objective
Fix the failure in "Experiment 17 — Unified Cognitive Loop Test" by making the `Retriever` weight-aware, ensuring the agent prefers nodes with high weights (trusted beliefs) over low weights (falsified beliefs) during planning.

# Background & Motivation
In Experiment 17, the agent failed to adapt to an environmental shift because its planning mechanism used a falsified rule (Action 0 adds 1) instead of a newly learned one (Action 0 adds -1). This happened because the `GoalConditionedRetriever` only considers geometric distance (`mu`). Since the desired target was `Delta=4`, the old rule (`Delta=1`) was geometrically closer than the new rule (`Delta=-1`), despite its weight having dropped in the `Observer`.

By making the `Retriever` weight-aware, we bridge the gap between the "World Model" (Forest structure) and the "Belief System" (Observer weights).

# Scope & Impact
- **Modify `hfn/retriever.py`**: Add weight-awareness to `GoalConditionedRetriever`.
- **Modify `hpm_fractal_node/experiments/experiment_unified_cognitive_loop.py`**: Update `UnifiedAgent` to pass the `Observer`'s weight provider to the `Retriever`.
- **Impact**: Enables autonomous adaptation and belief revision to drive action, not just internal state.

# Proposed Solution
1. **Update `GoalConditionedRetriever`**:
   - Add `weight_provider: Callable[[str], float] | None = None` to `__init__`.
   - Add `weight_penalty: float = 100.0` to `__init__`.
   - Update `retrieve` to apply the penalty: `score = dist + (1.0 - weight) * weight_penalty`.

2. **Update `UnifiedAgent`**:
   - In `__init__`, pass `self.observer.get_weight` as the `weight_provider` to the `GoalConditionedRetriever`.

3. **Verify**:
   - Run `python hpm_fractal_node/experiments/experiment_unified_cognitive_loop.py`.
   - Confirm the agent now generates a "New Plan" that avoids the falsified rule or uses the new rule correctly.

# Implementation Steps
- [x] **Step 1: Modify `hfn/retriever.py`**
  - Import `Callable` from `typing`.
  - Update `GoalConditionedRetriever` class.
- [x] **Step 2: Modify `hpm_fractal_node/experiments/experiment_unified_cognitive_loop.py`**
  - Update `UnifiedAgent.__init__` to pass the weight provider.
- [x] **Step 3: Verification**
  - Run the experiment and analyze the results.

# Verification & Testing
- Expected outcome: Agent reaches the goal even after the rule shift.
- Monitor: Surprise levels and plan generation in Phase 4.
