# SP54: Experiment 42 — Factorised Relational Abstraction

## Objective
To take the final leap from an HPM-consistent search process into true compositional generalization by addressing the last representational limitation: the collapse of multi-input relations. We will separate the "structure node" from the "execution embedding," preserving the identity of inputs (like `MAP(list, f)`) so schemas can be transferred independently of their specific internal operations (e.g., separating the `MAP` iteration from `OP_ADD` or `OP_MUL2`). 

## Background & Rationale
Currently, the system successfully tracks multipolygraph structural sequences (`node.inputs = [...]`) and correctly identifies the net transition delta. However, the execution embedding `mu` still mathematically collapses the sequence into a single fixed transition. 
As a result, `[LIST_INIT, FOR_LOOP, ITEM_ACCESS, OP_ADD, LIST_APPEND]` and `[... OP_MUL2 ...]` are treated as separate hard-coded schemas (`MAP_ADD1` vs `MAP_MUL2`). 

To achieve true abstraction reuse, the composed node must explicitly separate the structural scaffold (`MAP`) from the parameterised function slot (`f`).

Additionally, we need to fix the stochastic parent selection vector `p_vec = w_vec * curiosity_bonus`, which is currently linear, to the true replicator exponential `p_vec = np.exp(w_vec / tau) * curiosity_bonus` to sharpen exploitation.

## Proposed Implementation Steps
- [ ] **Fix Parent Selection**: Update `p_vec` calculation in `experiment_schema_transfer.py` to use `np.exp(w_vec / tau)` for true replicator competition.
- [ ] **Implement Factorised Nodes**: Adopt the specific HFN modification to separate "structure nodes" from "execution embeddings" without breaking the existing pipeline. *(Pending specific architectural guidance from the user).*
- [ ] **Verify Abstraction Reuse**: Run the curriculum and prove that Task C (`Map double`) drops drastically in iterations compared to Task B (`Map add one`), definitively proving that the `MAP` schema transferred and simply parameterized a new inner operator.
- [ ] **Verify Compositional Generalization**: Confirm Task D (`Filter positive`) correctly reuses the iteration schema while inserting the condition logic.