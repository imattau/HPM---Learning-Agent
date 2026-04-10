# SP54: Experiment 35 — Global Credit Assignment & Abstraction Reuse

## Objective
Address the final limitations of the Execution-Guided Synthesis pipeline to achieve true, generalized HPM alignment without heuristics:
1. **Local Credit Assignment & Hand-Tuned Coherence:** Currently, utility is evaluated only on the immediate new state, forcing the use of hand-tuned coherence weights (strong priors like `30 * len_change`) to guide search across reward valleys (e.g., creating a loop). We will implement **temporal credit assignment** (backpropagation of utility up the planning tree) so early structural steps naturally gain weight from future successes. This allows us to remove the hand-tuned coherence heuristics.
2. **No Abstraction Reuse:** The system repeatedly rediscovers list mapping from scratch. We will enable **HPM Recombination** so that successful compositional paths (e.g., `LIST_INIT -> FOR_LOOP -> ITEM_ACCESS -> OP_MUL2 -> LIST_APPEND`) are compressed into reusable macro-patterns for future tasks.
3. **Planning Retriever Misalignment:** The `planning_retriever` currently targets only the absolute state slice, while the global retriever uses the delta transition. We will align the `planning_retriever` to use a consistent transition-driven query.

## Proposed Architecture
1. **Backpropagated Utility:** 
   When a new node achieves a utility score, we will propagate a discounted reward (`gamma * utility`) up its ancestor chain in the `planning_forest`. The `planning_observer` will update the ancestors' weights, natively solving the sparse delayed reward problem.
2. **Remove Hand-Tuned Coherence:** 
   With backpropagation resolving credit assignment, we will eliminate the `coherence` multipliers, relying entirely on the generative epistemic accuracy and complexity penalty.
3. **Macro-Pattern Compression:** 
   Upon a successful synthesis, the agent will pass the winning explanation tree to the global `Observer` to track co-occurrences. If a sequence proves consistently useful, it will be naturally compressed into a new `MACRO` concept via `Recombination.compress()`.
4. **Delta-Aligned Planning Retrieval:** 
   Update `planning_retriever`'s `target_slice` to encompass the transition delta, unifying the search objective inside the population with the global action selection.

## Implementation Steps
- [ ] Implement recursive utility backpropagation within the `planning_forest`.
- [ ] Eliminate the hand-tuned `coherence` multipliers from the utility equation.
- [ ] Fix the `planning_retriever` `target_slice` to align with transition-driven dynamics.
- [ ] Trigger global `Observer._track_cooccurrence` and `_check_compression_candidates` upon finding a successful `path`.
- [ ] Run the benchmark tasks and verify that "Map double" synthesizes natively and begins to form compressed macro-abstractions.