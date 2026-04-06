# Objective
Update the `README.md` and `hpm_fractal_node/experiments/README.md` to document critical architectural omissions regarding the Meta-Forest, cost-aware attention, density-aware creation, stability mechanisms, and dynamic priors.

# Background & Motivation
The current documentation understates the complexity and self-organizing capabilities of the HFN Observer. The user noted five critical missing pieces:
1. **Meta-Forest**: Second-order learning via a `TieredForest(D=4)` that tracks node state.
2. **Cost-aware attention**: The `priority = surprise - weight` formula making attention learned.
3. **Density-aware creation**: Lacunarity and multifractal behaviors.
4. **Stability mechanisms**: Pruning, weight decay, and absorption thresholds.
5. **Dynamic Priors**: Clarifying that priors can drift.

# Scope & Impact
- Modify `README.md` (Root)
- Modify `hpm_fractal_node/experiments/README.md`
- Provide accurate descriptions of the Observer's role as a dynamics, control, and policy engine.

# Proposed Solution
1. Rewrite the `Core components` list in `README.md` to include a new bullet for `Meta-Forest`, expand the `Observer` bullet with sub-bullets for attention, stability, and density, and clarify the `Prior library` bullet regarding drift.
2. Rewrite the `Background` components list in `hpm_fractal_node/experiments/README.md` to include the same updates.

# Implementation Steps
- [x] Run `replace` on `README.md` to insert Meta-Forest, update Observer, and clarify Priors.
- [x] Run `replace` on `hpm_fractal_node/experiments/README.md` to apply the identical updates.