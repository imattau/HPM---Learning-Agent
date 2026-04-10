# SP54: Experiment 36 — Replicator Contrast Dynamics & Schema Transfer

## Objective
Address the structural overgrowth and credit diffusion risks of the newly implemented HPM planning dynamics by introducing **Replicator Contrast Dynamics**. Then, test whether the fully HPM-native agent can discover general program schemas (like `MAP` and `FILTER`) and transfer them across entirely different tasks.

## Background & Rationale
While temporal credit assignment (utility backpropagation) correctly bridges sparse reward valleys, it risks "credit diffusion" where utility spreads so broadly that all paths look "kind of good," dulling the selection pressure. 

True replicator dynamics rely on *relative* fitness. By calculating a running `baseline` utility for the population and updating weights based on the `advantage` (utility - baseline), we restore proper contrast dynamics: only paths that perform *better than average* grow in probability.

Once this is stable, the ultimate test of the system is **Schema Transfer**:
If the agent learns to synthesize a "Map add one" program, the global `Observer` should capture the `LIST_INIT -> FOR_LOOP -> ITEM_ACCESS -> LIST_APPEND` co-occurrences. When faced with a novel "Map double" or "Filter positive" task, the agent should transfer these compressed structural priors to solve the task significantly faster.

## Proposed Changes
1. **Replicator Contrast Dynamics:**
   Modify the utility backpropagation to use baseline subtraction:
   `baseline = mean(population_utilities)`
   `advantage = utility - baseline`
   `weight = sigmoid(gamma^d * advantage)`
2. **Schema Transfer Curriculum:**
   Expand the task list to form a transfer learning curriculum:
   - Task A: "Map add one" (Forces discovery of the MAP schema).
   - Task B: "Map double" (Tests transfer of the MAP schema).
   - Task C: "Filter positive" (Tests discovering a new FILTER schema by composing MAP with a new COND_IS_POSITIVE concept).
3. **Concept Expansion:**
   Add `COND_IS_POSITIVE` to the `CONCEPTS` and `ASTRenderer` to support filtering tasks.

## Implementation Steps
- [ ] Create `experiment_schema_transfer.py` based on Experiment 35.
- [ ] Add `COND_IS_POSITIVE` concept.
- [ ] Implement baseline subtraction in the planning backprop loop.
- [ ] Define the transfer learning curriculum tasks.
- [ ] Run the experiment to verify cross-task schema transfer and efficient replicator dynamics.