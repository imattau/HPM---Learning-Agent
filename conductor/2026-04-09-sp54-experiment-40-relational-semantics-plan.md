# SP54: Experiment 40 — Relational Semantics and Pure Retrieval

## Objective
Address the final structural leaks and selection heuristic issues identified in the schema transfer implementation:
1. **Unify Delta Scaling:** Remove arbitrary `* 10.0` scaling multipliers from delta transitions to ensure the retrieval query space and the stored state space are perfectly aligned.
2. **Exponential Parent Selection:** Implement true replicator dynamics for parent selection by weighting probabilities as `exp(weight / tau) * curiosity`, ensuring sharp exploitation of high-utility structures.
3. **Unified Retrieval:** Combine candidates from the global `self.retriever` (semantic concepts/macros) and the local `planning_retriever` (ephemeral structural compositions) during the expansion phase, allowing the agent to seamlessly compose both primitive rules and newly discovered partial paths.
4. **Relational Semantics (Foundation):** Lay the groundwork for true multipolygraph relations by storing the explicit transition `delta` within the composed node's representation, rather than just averaging the inputs.

## Implementation Steps
- [ ] Update `experiment_schema_transfer.py` to fix the exponential probability weighting for `parent` selection.
- [ ] Remove all `* 10.0` scaling factors applied to `target_delta` and `remaining_delta`.
- [ ] Merge `local_candidates` from `planning_retriever` into the `candidates` list during expansion.
- [ ] Update `compose_sequence` to inject relational transition semantics (the delta between first and last node) into the composed `mu`.