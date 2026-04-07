# Plan: Developmental Cognitive System Refactor (Part 2)

## Objective
Refactor Experiment 20 to implement true compositional reasoning by making the `CodeRenderer` sequential and introducing partial credit and negative signals into the learning loop, as suggested in the recent feedback.

## 1. Make Renderer Sequential (CRITICAL)
- **Problem**: The current `CodeRenderer` treats the program structure as a "bag of tokens" and collapses the structure (e.g., ignoring ordering).
- **Solution**: Refactor `CodeRenderer.render` to process concepts in order, maintaining local state (e.g., variable `x`).
    - `CONST_X` translates to `x = X`.
    - `VAR_INP` translates to `x = inp`.
    - `OP_ADD` translates to `x += 1` (only if `x` exists).
    - `RETURN` translates to `return x`.

## 2. Add Partial Credit
- **Problem**: Learning is currently success-triggered (binary), leading to slow generalization.
- **Solution**: Update `TaskRunner.run_task` to compute a partial score based on the difference between the execution `result` and `s_goal`. Use this score to scale the credit assignment when reinforcing structures.

## 3. Penalize Bad Paths (Negative Signal)
- **Problem**: The system never penalizes wrong sequences or useless nodes, allowing bad structures to persist.
- **Solution**: If execution results in `None` or an error, use `observer.penalize_id` on the nodes used in the failed plan to suppress them.

## 4. Define Semantics for All Concepts
- Ensure that the concepts `OP_ADD`, `FOR_LOOP`, `LIST_INIT`, and `LIST_APPEND` either have defined structural priors and renderer support, or remove them to keep the concept space clean for this specific experiment phase. For now, we will focus on `OP_ADD` and remove list-related concepts if unused.

## Implementation Steps
- [ ] Refactor `CodeRenderer` in `experiment_developmental_cognitive_system.py`.
- [ ] Update `TaskRunner` to handle partial scoring and negative penalties.
- [ ] Verify the agent learns to compose `CONST_1` and `OP_ADD` to reach a goal of `2`, proving true compositional learning.