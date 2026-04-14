# SP74 Macro Reuse Fix Plan

## Objective
Address the feedback that `_try_exact` failed to reuse the learned graph macro during Phase 2 of SP74. The goal state vector is not invariant across different sized base graphs, and the `HybridRetriever` favors primitive operations (leaf nodes) over macros because the queried goal state is itself a leaf node with no inputs. We will fix this by adjusting the retrieval method in the experiment script to prioritize macros.

## Scope & Impact
- Modify `hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py` only.
- Introduce `MacroBoostRetriever` to wrap the base `HybridRetriever` and re-rank candidate nodes, pushing `relation_type == "macro"` to the top.
- This ensures true macro reuse, avoiding the fallback to `_try_bfs`.
- Does not require any core changes to `BaseHFNAgent` or `GraphOracle`.

## Proposed Solution
1. **Implement `MacroBoostRetriever`:**
   Add a custom retriever class inside `experiment_sp74_graph_fewshot.py` that inherits from `Retriever`. It will take the agent's base retriever and re-rank candidates to prioritize macros.
   ```python
   class MacroBoostRetriever(Retriever):
       def __init__(self, base_retriever: Retriever):
           super().__init__(base_retriever.forest)
           self.base_retriever = base_retriever

       def retrieve(self, query, k=10):
           candidates = self.base_retriever.retrieve(query, k=max(k * 2, 20))
           candidates.sort(key=lambda n: 0 if n.relation_type == "macro" else 1)
           return candidates[:k]
   ```

2. **Integrate Custom Retriever:**
   Apply this retriever to the agent inside `run_experiment()`.
   ```python
   agent.retriever = MacroBoostRetriever(agent.retriever)
   # Also update the observer's reference
   agent.observer.retriever = agent.retriever
   ```

3. **Verify Reuse:**
   Run the experiment to confirm that Phase 2 is now solved via `exact` instead of `bfs`.

## Implementation Steps
- [ ] Edit `hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py` to include `MacroBoostRetriever`.
- [ ] Apply `MacroBoostRetriever` to the agent initialization.
- [ ] Run the experiment script.
- [ ] Verify Phase 2 logs "Triangle solved via exact." instead of "bfs".

## Verification
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py`.
- Assert that macro reuse (`exact` strategy) was utilized for Phase 2.
