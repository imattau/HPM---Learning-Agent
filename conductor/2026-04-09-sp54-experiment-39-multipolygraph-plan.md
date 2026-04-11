# SP54: Experiment 39 — Multi-Arity Composition (Multipolygraph)

## Objective
To upgrade the `HFN` data structure and the execution-guided synthesis pipeline from a binary tree-based compositional algebra to a true multi-arity (multipolygraph-like) composition. This will allow the system to represent parallel structure, shared substructures (e.g., `MAP(list, function)`), and clean non-linear semantics without duplication, moving beyond simple sequential pairwise folding.

## Background & Rationale
Currently, the system builds ASTs by folding nodes in a binary tree: `parent = (child_a + child_b)`. A true multipolygraph node should have explicit `inputs`, `outputs`, and a `relation_type`. By introducing these fields to the `HFN` class, we can compose nodes over sets rather than just pairs, preserving compatibility with the existing system while unlocking true compositional reuse.

## Proposed Changes
1. **Extend HFN Class:**
   Modify `hfn/hfn.py` to add `inputs`, `outputs`, and `relation_type` fields to the `HFN` dataclass. Add an `add_relation(inputs, outputs, relation_type)` method.
2. **True Multi-Arity Composition:**
   In `experiment_schema_transfer.py`, replace `_fold_nodes` with `compose_sequence(nodes)`. Instead of iteratively pairing nodes, this function will create a single `HFN` parent whose `inputs` are the entire sequence of nodes and `relation_type="sequence"`.
3. **Update AST Renderer:**
   Modify `ASTRenderer._get_hfn_leaves()` to flatten `node.inputs` instead of just `node.children()`.
4. **Update Recombination/Compression:**
   Modify the `Recombination` or global `Observer` compression logic to create multi-input nodes rather than binary trees, enabling reusable macro structures like `MACRO(LIST_INIT, FOR_LOOP, ...)`.

## Implementation Steps
- [ ] Add `inputs`, `outputs`, `relation_type` to `HFN` in `hfn/hfn.py`.
- [ ] Implement `compose_sequence(self, nodes: List[HFN]) -> HFN` in `SchemaTransferAgent` to replace `_fold_nodes`.
- [ ] Update `ASTRenderer._get_hfn_leaves` to handle the new `inputs` list.
- [ ] Update the `plan` method to use `compose_sequence` for rendering and returning the final AST.
- [ ] Update `hfn/observer.py` or `hfn/recombination.py` if necessary to ensure `_check_compression_candidates` generates multi-input macros (if not already handled natively by passing the sequence).
- [ ] Run the transfer experiments to verify faster convergence on Task B and robust transfer to Tasks C and D.