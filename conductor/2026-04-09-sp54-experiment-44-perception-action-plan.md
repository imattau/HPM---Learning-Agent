# SP54: Experiment 44 — Unified Perception-Action Schema Learning

## Objective
To bridge the gap between perceptual concept learning and deliberative planning by replacing symbolic operations (`OP_ADD`, `OP_MUL2`) with grounded operators derived from raw perception. This tests whether the HPM agent can use perceptually learned representations as executable actions within its structural planning loop, achieving cross-modal compositional generalization.

## Background & Rationale
Currently, the system discovers structural schemas (like `MAP`) but fills the operational slots with hard-coded symbolic priors (`OP_ADD`, `OP_MUL2`). In a true AGI architecture, these operations should be grounded concepts learned from perception (e.g., observing an icon and learning its associated transformation `delta`).

By extracting nodes from a perceptual forest, converting their learned deltas into planning-compatible `GROUNDED_OP` concepts, and removing the symbolic priors, we force the planner to compose schemas using entirely learned, grounded representations.

## Proposed Changes
1. **Mock Perceptual Forest:** Create a simulated perceptual forest containing nodes that map 64D icons to 1D transition deltas (simulating the output of Experiment 39).
2. **Extract Perceptual Ops:** Implement `extract_perceptual_ops` to convert these 65D perceptual nodes into 20D+ planning-compatible `HFN` nodes with `relation_type="grounded_op"`.
3. **Inject Priors:** Remove `OP_ADD`, `OP_SUB`, and `OP_MUL2` from the planner's available priors and replace them with the extracted perceptual ops.
4. **Update AST Renderer:** Allow the renderer to translate `GROUNDED_OP` nodes back into executable Python operations by directly applying the learned numerical delta (e.g., `x += delta`).

## Implementation Steps
- [ ] Copy `experiment_schema_transfer.py` to `experiment_unified_perception_action.py`.
- [ ] Implement `extract_perceptual_ops` and a mock perceptual forest generator.
- [ ] Update `ASTRenderer._get_concept` and `render` to handle `GROUNDED_OP` and apply the `delta`.
- [ ] Filter out symbolic `OP_` rules from the planner's initial priors and inject the perceptual ops.
- [ ] Run the curriculum (Task A, Task B, Task C) and verify that the agent successfully synthesizes the `MAP` schema using `GROUNDED_OP` nodes.