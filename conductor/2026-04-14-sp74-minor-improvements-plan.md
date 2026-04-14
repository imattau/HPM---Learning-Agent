# SP74 Minor Improvements Plan

## Objective
Implement the optional minor improvements suggested by the user to refine the SP74 Graph Domain experiment: persist the custom retriever for reusability and test generalization on a larger graph.

## Scope & Impact
- Modify `hfn/retriever.py` to add a new `MacroPrioritizingRetriever` class.
- Update `hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py` to:
    - Import and use `MacroPrioritizingRetriever` instead of the local `MacroBoostRetriever`.
    - Change the test phase (Phase 2) from a 3-node triangle to a 4-node square graph to prove the macro generalizes across different base graph sizes.

## Proposed Solution
1. **Persist `MacroPrioritizingRetriever`:**
   Add the wrapper retriever to `hfn/retriever.py`.
   ```python
   class MacroPrioritizingRetriever(Retriever):
       def __init__(self, base_retriever: Retriever):
           super().__init__(base_retriever.forest)
           self.base_retriever = base_retriever

       def retrieve(self, query: 'HFN', k: int = 10) -> list['HFN']:
           candidates = self.base_retriever.retrieve(query, k=max(k * 2, 20))
           candidates.sort(key=lambda n: 0 if n.relation_type == "macro" else 1)
           return candidates[:k]
   ```

2. **Update Experiment Script:**
   - Remove the local `MacroBoostRetriever`.
   - Import `MacroPrioritizingRetriever` from `hfn.retriever`.
   - Update Phase 2:
     ```python
     # Square graph 0-1, 1-2, 2-3, 3-0
     test_input = nx.Graph()
     test_input.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
     # Target: square + 2 new nodes (4, 5)
     test_output = test_input.copy()
     test_output.add_node(4)
     test_output.add_node(5)
     ```

## Implementation Steps
- [ ] Edit `hfn/retriever.py` to include `MacroPrioritizingRetriever`.
- [ ] Edit `hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py` to use the new retriever and the 4-node square graph.
- [ ] Run the experiment to verify successful macro reuse on the larger graph.

## Verification
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py`.
- Assert Phase 2 is solved via `exact` on the 4-node square graph.
