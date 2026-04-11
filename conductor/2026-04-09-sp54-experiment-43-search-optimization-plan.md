# SP54: Experiment 43 — Search Optimization (Caching & Pruning)

## Objective
To dramatically accelerate the inner loop of the execution-guided synthesis agent by eliminating redundant program construction, execution, and state evaluation. The current system performs hundreds of thousands of full AST renders and Python `exec()` calls, creating a severe computational bottleneck. We will implement multi-level caching and reduce search branching to achieve a 10-50x speedup while maintaining pure HPM dynamics.

## Proposed Optimizations
1. **Execution Caching**: 
   Maintain an `execution_cache` mapping path ID tuples to their corresponding `(outputs, errors)`. This prevents re-executing the same code sequence across different search branches.
2. **Composition Caching**: 
   Maintain a `composition_cache` mapping `(parent_id, rule_id)` pairs to their composed `HFN` relational node. This avoids repeatedly walking and averaging path states from scratch.
3. **Reduced Search Branching**: 
   Lower retrieval `k` values (Global `k=10`, Local `k=3`) to focus search on the most conceptually relevant candidates, significantly reducing the number of executions per iteration.
4. **Faster Deduplication**: 
   Move path-based deduplication to occur *before* code rendering and execution. If a path has already been evaluated, skip the expensive synthesis/execution steps entirely.
5. **EMA Baseline Optimization**: 
   Ensure the EMA baseline is updated efficiently without full population rescans.

## Implementation Steps
- [ ] Initialize `execution_cache` and `composition_cache` within the `plan` method.
- [ ] Update the inner expansion loop to check these caches before calling `renderer.render` and `executor.run_batch`.
- [ ] Reduce retrieval counts in the `plan` method.
- [ ] Refactor the `plan` loop to use path-based deduplication before execution.
- [ ] Run the curriculum and verify that Task B discovers the MAP schema within 5 minutes, and Tasks C/D demonstrate rapid transfer.