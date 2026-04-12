# SP60: Cumulative Abstraction via Multi-Polygraph HFNs (Implementation Plan)

## Phase 1: Support for Multi-Polygraph Execution
- [ ] Update `hpm_fractal_node/code/sp57_operators.py`:
    - Add `PolygraphOperator` subclass that can be instantiated from an `HFN` node with `inputs`.
    - Implement `apply()` for `PolygraphOperator` by recursively executing its children.
    - Ensure `get_params()` correctly summarizes the aggregate transformation for future retrieval.

## Phase 2: Structural Abstraction Mechanism
- [ ] Create `hpm_fractal_node/experiments/experiment_cumulative_abstraction.py`:
    - Setup the 90D HFN Forest and `GeometricRetriever`.
    - Implement `MultiPolygraphBeamSearch`:
        - Use `log_prob` for re-ranking candidates (Mahalanobis distance).
        - Correctly decode `macro` nodes into `PolygraphOperator` instances.
    - Implement `compress_to_polygraph(chain, forest)`:
        - Take a winning operator chain.
        - Create a parent HFN node with `inputs=[child_nodes...]` and `relation_type="macro"`.
        - Register this node in the `Forest`.

## Phase 3: Hierarchical Validation Loop
- [ ] Implement Task A: $x \to (x + 1) \times 2$.
    - Perform synthesis (Depth 2).
    - Trigger `compress_to_polygraph`.
- [ ] Implement Task B: $x \to ((x + 1) \times 2) \pmod{10}$.
    - Run Task B in two modes:
        1. **Naive Mode**: Temporarily hide the new macro node.
        2. **Abstracted Mode**: Use the full forest.
    - Track metrics: Search Depth and Nodes Evaluated.

## Phase 4: Finalization
- [ ] Document result analysis in `hpm_fractal_node/experiments/README_SP60.md`.
- [ ] Update root `GEMINI.md`.
