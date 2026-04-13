# SP54: Experiment 44 Fixes — Unified Perception-Action Schema Learning

## Objective
Address critical indexing bugs and structural leaks in the first iteration of the unified perception-action experiment. The `extract_perceptual_ops` function currently stores the learned transformation delta in the wrong index, resulting in the grounded operations having zero effect during execution. We will also restore structural tracking to the extracted nodes and boost their retrieval signal to ensure they successfully compete with existing macros and primitives.

## Proposed Changes
1. **Delta Index Mismatch (Critical Bug):**
   Fix the index where `delta` is stored in the planning embedding. It must be `S_DIM + DIM + 0` (index 34) so the AST Renderer correctly reads and applies the numerical transformation.
2. **Missing Structural Linkage:**
   Add a placeholder `source_node` to the `inputs` list of the generated `GROUNDED_OP` nodes. This preserves the perceptual origin and is necessary for accurate macro compression and tracking later on.
3. **Strengthen Retrieval Signal:**
   Set `op_mu[S_DIM] = 1.0` (action presence) and `op_mu[13] = 1.0` (mutation state flag) to ensure the `GoalConditionedRetriever` actually selects these nodes when trying to satisfy a target delta.
4. **Enforce Purity in Execution (Optional):**
   Explicitly ignore legacy symbolic operators (`OP_ADD`, `OP_SUB`, `OP_MUL2`) in the `ASTRenderer` to prove that the system is relying 100% on the newly learned grounded operations to solve the tasks.

## Implementation Steps
- [ ] Correct the delta indexing in `extract_perceptual_ops`.
- [ ] Add `inputs=[source_node]` to the generated `GROUNDED_OP` HFN.
- [ ] Boost the retrieval flags in `op_mu`.
- [ ] Add purity block to `ASTRenderer.render` to skip legacy symbolic ops.
- [ ] Re-run the experiment and verify Task B and Task C correctly utilize the `GROUNDED_OP` nodes to build `MAP` schemas.