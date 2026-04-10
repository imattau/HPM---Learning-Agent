# SP54: Experiment 37 — Sharpened Schema Emergence & Transfer Metrics

## Objective
To prove that an HPM-native agent can autonomously discover and transfer general program schemas (like MAP and FILTER) without engineered heuristics. We will sharpen the Replicator Contrast Dynamics to accelerate the emergence of these multi-node attractors and define clear metrics to measure the efficiency of schema transfer across tasks.

## Background & Rationale
Task B ("Map add one") is slow because discovering a multi-node schema (`LIST_INIT -> FOR_LOOP -> ITEM_ACCESS -> OP -> LIST_APPEND`) from scratch relies on rare alignment events. To accelerate this emergent structure without reintroducing bias, we must refine the dynamics:
1. **Partial Attractor Formation:** Don't wait for full success. Compress frequently co-occurring partial structures (e.g., `LIST_INIT + FOR_LOOP`) that exhibit high utility *during* the search.
2. **Sharpened Selection Pressure:** A soft sigmoid allows too many mediocre nodes to survive. We need a sharper contrast function (`exp(advantage / τ)` with low `τ`) to ruthlessly select for high-utility structures.
3. **Stabilized Baseline:** Use an Exponential Moving Average (EMA) for the population baseline utility to reduce noise.
4. **Anti-Diffusion:** Only backpropagate positive advantage, preventing weak structures from accumulating weight.

If successful, Task B will eventually converge and compress the MAP schema. Task C ("Map double") should then solve in an order of magnitude fewer iterations, definitively proving Schema Transfer.

## Metrics for Schema Emergence
- **Iterations to Solve:** The primary metric for transfer efficiency.
- **Nodes Expanded:** Total computational effort.
- **Macros Discovered:** Track when `compressed(...)` nodes enter the global forest.
- **Macro Utilization:** Verify if the winning path in Task C/D actually uses the compressed macro.

## Implementation Steps
- [ ] Implement EMA baseline utility.
- [ ] Sharpen advantage-to-weight function (`sigmoid(advantage / 3.0)`).
- [ ] Add anti-diffusion to backprop (only propagate if `advantage > 0`).
- [ ] Trigger partial attractor compression during search for high-advantage paths.
- [ ] Add explicit metric logging per task.