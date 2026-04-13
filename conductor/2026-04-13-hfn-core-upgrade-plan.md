# HFN Core Upgrade: Structural Retrieval & Enhanced Recombination

## Objective
Implement the suggested core HFN upgrades to support structural retrieval (DAG similarity) and enhanced structural recombination (concatenating macro inputs), ensuring full backward compatibility.

## Implementation Steps

### 1. `hfn/retriever.py`
- [ ] Add `StructuralRetriever(Retriever)` class.
- [ ] Implement `_structural_fingerprint(node)` to extract DAG features (children, inputs, edges, relation types, depth proxy).
- [ ] Implement `retrieve(query, k)` using fingerprint distance.
- [ ] Add `HybridRetriever(Retriever)` class to combine geometric and structural scores.

### 2. `hfn/recombination.py`
- [ ] Add `recombine_structural(macro_a, macro_b, forest, new_id)` method to the `Recombination` class.
- [ ] Implement input concatenation, edge preservation, and intermediate edge connection.

### 3. `hfn/__init__.py`
- [ ] Export `StructuralRetriever` and `HybridRetriever`.

### 4. Verification
- [ ] Write tests in `tests/test_retriever.py` for `StructuralRetriever` and `HybridRetriever`.
- [ ] Write tests in `tests/test_recombination.py` for `recombine_structural`.
- [ ] Ensure the full test suite passes.