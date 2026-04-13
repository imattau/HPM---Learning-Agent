# SP54: Experiment 44 Final Fixes — Unified Perception-Action Schema Learning

## Objective
Finalize the structural alignment of the unified perception-action experiment by ensuring the perceptual source nodes (icons) are properly registered in the global forest, preserving full traceability and enabling future macro compression across the perception-action boundary. We will also refine the retrieval balance to ensure the grounded operations compete effectively with structural priors.

## Proposed Changes
1. **Register Source Nodes:** Modify `extract_perceptual_ops` to return both the generated operator nodes and their mock perceptual source nodes (`icon_X`). Update the `SchemaTransferAgent` to register these source nodes in the global `forest`, explicitly linking the planning space to the perception space.
2. **Refine Delta Encoding:** Re-evaluate the delta broadcasting. Setting the delta across all value metrics (mean, min, max, first, last) is mathematically correct for a `MAP(ADD)` operation, so we will maintain it to provide a strong, accurate retrieval signal for the grounded ops.
3. **Execution Purity:** Ensure the `ASTRenderer` strictly ignores legacy symbolic operators to guarantee the agent relies entirely on the new grounded representations.

## Implementation Steps
- [ ] Update `extract_perceptual_ops` to return `(ops, sources)`.
- [ ] Update `SchemaTransferAgent.__init__` to register the `sources`.
- [ ] Run the final experiment curriculum.