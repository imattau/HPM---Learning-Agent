# SP81 Stretch Test Plan

## Objective
Implement and execute the SP81 stretch test to verify HPM's ability to compose a filter macro with a map macro sequentially without additional training.

## Background
The agent previously succeeded at SP80 (learning filter and map operations). This stretch test evaluates zero-shot compositional abstraction by chaining these macros to solve `filter_positive_then_double`.

## Implementation Steps
1. Create `hpm_ai_v2/experiments/experiment_sp81_stretch_test.py`.
2. Define an `SP81Agent` that mixes `SequentialCompositionMixin` into `BaseHFNAgent`.
3. Provide a multi-stage study phase:
    - Phase 1: Learn `scalar_add1` and `scalar_mul2`.
    - Phase 2: Learn `filter_positive` and `double` via 2-shot examples.
4. Execute training phase to solve `filter_then_double` using `_try_sequential_compose`.
5. Run tests for `k=1,2,3,5` and record accuracy.

## Verification
Run the SP81 benchmark script. Expected results:
- `k=1`: ≥ 80% accuracy
- `k=2,3,5`: 100% accuracy