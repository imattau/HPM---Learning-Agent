# SP54: Experiment 41 — True Relational Abstraction (Preserving Structure)

## Objective
To resolve the final representational limitation preventing true compositional generalisation: premature information loss during composition. Currently, `compose_sequence` collapses a sequence of nodes into a single mean embedding. We will upgrade the HFN composition logic to act as a true **multipolygraph**, where a node stores its explicit inputs and a separate execution embedding (representing the net transformation), allowing the system to learn parameterised schemas like `MAP(list, f)` rather than hard-coded instances like `MAP_ADD1`.

## Background & Rationale
When we compose `[LIST_INIT, FOR_LOOP, ITEM_ACCESS, OP_ADD, LIST_APPEND]`, averaging the `mu` vectors destroys the structural identity of the inputs. A true relational node must separate its *structural definition* (its inputs) from its *execution embedding* (what it actually does to the state). 

By setting the composed node's `mu` to represent the strict execution transformation (the final state + the net delta) while preserving the exact input list, we stop destroying structure. This allows the Retriever to retrieve the abstract `MAP` schema even if the internal `OP` needs to change, finally enabling Task C and Task D to cleanly reuse the schema discovered in Task B.

## Proposed Changes
1. **Modify `compose_sequence`**: 
   Stop averaging `mu`. Instead, set the execution embedding:
   - `mu[:S_DIM]` = The exact final state of the sequence (`nodes[-1].mu[:S_DIM]`).
   - `mu[S_DIM+DIM:]` = The net transformation delta (`nodes[-1].mu[:S_DIM] - nodes[0].mu[:S_DIM]`).
2. **Update Macro Compression**:
   Apply the exact same execution embedding logic when creating macro nodes in the Observer feedback loop, rather than averaging the inputs.
3. **Add Structural Similarity to Retrieval**:
   Add a light structural bonus to the `GoalConditionedRetriever` (or the planning loop's retrieval post-processing) to favor nodes that structurally align with the query, enabling the selection of generalized macros over raw primitives when appropriate.

## Implementation Steps
- [ ] Update `compose_sequence` in `experiment_schema_transfer.py` to use execution embeddings instead of means.
- [ ] Update the macro compression logic in the expansion loop to use execution embeddings.
- [ ] Add a structural similarity bonus during candidate retrieval to favor macros.
- [ ] Run the curriculum to verify that Task C and Task D solve in an order of magnitude fewer iterations due to true structural reuse.