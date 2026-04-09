# SP54: Experiment 34 — Pure Observer-Driven Dynamics (Removing Beam Search)

## Objective
To take the final step towards a fully coherent, end-to-end HPM agent. While Experiment 33 successfully aligned the *planning objective* (using HPM Utility instead of Euclidean distance), the *search dynamics* remained heuristic (a hard-pruned beam search: sort -> truncate -> discard). This experiment will replace the heuristic beam search entirely with native HPM `Observer` and `Forest` population dynamics (weight update -> competition -> persistence/absorption).

## Background & Rationale
As highlighted in the architectural review, true HPM expects soft population dynamics driven by continuous weight evolution, coexistence, and soft competition, not hard elimination. 
If we treat partial programs (ASTs) as genuine HPM nodes in an "Active Planning Forest", the `Observer` can manage their lifecycle natively. The `Evaluator` computes their utility, the `Observer` updates their weights based on this utility, and the `Retriever` naturally samples from the highest-weighted nodes for the next step of expansion (`recombine`).

## Proposed Architecture
1. **The Active Planning Forest:**
   Instead of a `beam` list, the agent maintains an active `Forest` of partial AST nodes.
2. **Expansion via Recombination:**
   At each step, the `Retriever` samples a high-weight partial AST node and a prior concept node. These are `recombined` to form a new, deeper AST node.
3. **Soft Selection via Observer:**
   The new AST is executed, and its empirical state is evaluated by the `Evaluator` (Accuracy - Complexity + Coherence). This utility score is passed to the `Observer` as a reinforcement signal.
4. **Natural Decay & Dominance:**
   The `Observer` natively decays the weights of all nodes over time. Nodes that consistently yield high utility expansions gain weight and dominate the population, while useless paths organically decay into obscurity without needing a hard "beam width" cutoff.

## Implementation Steps
- [ ] Initialize an ephemeral `Forest` inside the `plan` method to hold the population of search paths.
- [ ] Refactor the search loop to `retrieve` parent paths from this forest based on their `Observer` weights, rather than iterating over a deterministic beam list.
- [ ] Execute the generated ASTs and use the `Evaluator` to compute the HPM Utility.
- [ ] Feed the utility into the `Observer` to update the node's weight in the population.
- [ ] Run the benchmark tasks to verify that soft population dynamics can successfully and efficiently converge on the "Map double" solution.