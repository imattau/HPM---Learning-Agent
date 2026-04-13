# Bug Fix Plan for SP67 Phase 6

## Objective
Fix the failure in SP67 Phase 6 (Blackboard Scaffolding) where the Alice agent fails to record failures to the blackboard when encountering an impossible task.

## Root Cause
The `solve_with_social` method in `SocialAnalogicalAgent` calls `self.try_recombination(inputs, outputs)` if analogy fails. If `try_recombination` throws an exception (e.g., due to an execution error when generating permutations), the method exits early and the `post_failure` call is never reached.

## Proposed Changes
1. **Update `SocialAnalogicalAgent.solve_with_social`:** Wrap the `try_recombination` call in a `try...except Exception` block to ensure that execution proceeds to the `post_failure` step even if recombination fails or throws an error.
2. **Update `SocialForest.post_failure`:** Add a debug print `[BLACKBOARD] Recorded failure for {task_id}` to confirm execution during the test.

## Verification
Run `python3 hpm_fractal_node/experiments/experiment_sp67_social_recombination.py`. Phase 6 should now pass, demonstrating that institutional scaffolding (blackboard) correctly records failures.