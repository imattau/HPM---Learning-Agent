# SP74 Macro Reuse Fix Plan (v2)

## Objective
Demonstrate true macro creation and reuse in the Graph Domain. The previous run used a single primitive (`ADD_STAR`) which didn't require composition. We will modify the experiment to require a 2-step transformation, ensuring HPM creates a composite macro in Phase 1 and reuses it in Phase 2. We will also retain the `MacroBoostRetriever` to ensure the macro is prioritized during retrieval.

## Scope & Impact
- Modify `hpm_ai_v2/domains/graph_domain.py` to remove the "cheat" `ADD_STAR` primitive.
- Modify `hpm_ai_v2/domains/graph_renderer.py` to remove `ADD_STAR` rendering logic.
- Modify `hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py`:
    - Task: Add two new nodes to the graph.
    - Introduce `MacroBoostRetriever` to prioritize macros.
    - Verify Phase 1 creates a macro and Phase 2 reuses it via `exact`.

## Proposed Solution
1. **Remove `ADD_STAR`:**
   Clean up the graph domain by removing the `ADD_STAR` shortcut. This forces the agent to use composition for complex tasks.
2. **Implement 2-Step Task:**
   Change the experiment goal to adding two nodes. This requires `[ADD_NODE, ADD_NODE]`.
3. **Macro Priority Retrieval:**
   Keep the `MacroBoostRetriever` in the experiment script to ensure that once the `[ADD_NODE, ADD_NODE]` macro is learned, it is ranked above the individual `ADD_NODE` primitives during Phase 2 retrieval.

## Implementation Steps
- [ ] Remove `ADD_STAR` from `hpm_ai_v2/domains/graph_domain.py`.
- [ ] Remove `ADD_STAR` from `hpm_ai_v2/domains/graph_renderer.py`.
- [ ] Update `hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py`:
    - Set target output to original + 2 nodes.
    - Keep/Add `MacroBoostRetriever`.
- [ ] Run and verify.

## Verification
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py`.
- Phase 1 should solve via `bfs` (depth 2) and create a macro.
- Phase 2 should solve via `exact` by reusing that macro.
