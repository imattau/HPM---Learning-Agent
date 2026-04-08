# SP47: Experiment 23 — Structural Refinement (Self-Debugging)

## 1. Objective
To validate the HFN architecture's capacity for **Iterative Engineering**. The system must demonstrate the ability to perform local graph edits (splicing and patching) to fix logical errors in an existing compositional program graph, rather than discarding the entire structure and re-planning from scratch upon failure.

## 2. Background & Motivation
In Experiment 22 (SP46), the agent successfully mapped its causal boundaries by autonomously pruning structurally invalid sequences (like iterating over an integer). However, when a plan is syntactically valid but *logically incorrect* (e.g., it produces `[2]` instead of `[3]`), the current planner treats it as a complete failure and searches for a new path from the origin.

Human programmers rarely rewrite a 100-line function from scratch when it's off-by-one. They **localize the fault** and **patch** the specific operation. For HFN to achieve scalable synthesis, it must treat its constructed graphs as malleable objects. If a graph is 90% correct, the agent should identify the 10% gap (residual surprise) and retrieve a specific node to splice into the topology to bridge that gap.

## 3. Setup & Environment

### Domain: Semantic Program Space
- **State Representation**: 6D semantic vector `[AccumulatorValue, ReturnedFlag, ListLength, ListTargetValue, IteratorActive, ListInitFlag]`.
- **Action Space**: Structural primitives including `CONST_1`, `CONST_5`, `VAR_INP`, `OP_ADD` (+1), `OP_SUB` (-1), `OP_MUL2` (*2), `LIST_INIT`, `FOR_LOOP`, `ITEM_ACCESS`, `LIST_APPEND`, `RETURN`.

### The Curriculum (Perturbation Tasks)
The agent is trained on a base task and then tested on a "perturbed" task that requires a minor structural modification.
1.  **Base Task**: `map_add_one` (`[1, 2] -> [2, 3]`). The agent builds and chunks the 7-step graph: `LIST_INIT -> VAR_INP -> FOR_LOOP -> ITEM_ACCESS -> OP_ADD -> LIST_APPEND -> RETURN`.
2.  **Perturbed Task A (Substitution)**: `map_add_two` (`[1, 2] -> [3, 4]`).
    *   *The Test*: The agent will retrieve its high-weight `map_add_one` chunk, but execution will yield `[2, 3]`. Instead of failing, the agent must localize the fault at the `OP_ADD` node and substitute it or splice another `OP_ADD` into the graph.
3.  **Perturbed Task B (Insertion)**: `map_add_one_and_double` (`[1, 2] -> [4, 6]`).
    *   *The Test*: The agent retrieves the base chunk, finds it lacking, and splices an `OP_MUL2` node immediately after the `OP_ADD` node.

## 4. Architectural Enhancements

### 1. Causal Fault Localization
When a rendered program executes successfully but fails the strict evaluation (e.g., wrong output value), the `TaskRunner` computes the **Residual State Delta**: `Target State - Actual Output State`.
The agent then traverses its constructed program graph to find the structural boundary where this residual delta can be safely inserted or substituted without violating topological constraints.

### 2. Topological Splicing (Graph Mutation)
The `DevelopmentalAgent` requires a new capability: `patch_graph(root_node, target_delta)`.
-   **Identify Target Node**: Find the node in the tree responsible for the target dimension (e.g., `AccumulatorValue`).
-   **Retrieve Patch**: Use the `GoalConditionedRetriever` to query for a node that provides the `target_delta`.
-   **Mutate**: Replace the target node with a new `COMPOSE(target_node, patch_node)` or substitute it entirely.

### 3. Granular Credit Assignment
When a patched graph succeeds, the newly composed subgraph (the patch) is registered and boosted, while the underlying generic structure is preserved.

## 5. Evaluation Metrics
1.  **Patching Efficiency**: Compare the number of nodes explored when patching an existing graph vs. re-planning from scratch. Patching should require $O(k)$ retrievals (where $k$ is the patch size) rather than $O(b^d)$ full search.
2.  **Splicing Success Rate**: The percentage of attempted graph mutations that result in syntactically valid and executable code.
3.  **Chunk Adaptation Rate**: How frequently the agent successfully reuses the base `map_add_one` chunk to solve perturbed variants.

## 6. Failure Modes to Watch
- **Destructive Interference**: Splicing a new node corrupts the downstream logic (e.g., adding an operation after `RETURN` or modifying the list length unexpectedly). (Solution: Strict topological constraint checks before splicing).
- **Local Minimum Trap**: The agent repeatedly patches the wrong node (e.g., modifying the loop initialization instead of the accumulator logic) because it's locally closer in semantic space.

## 7. Implementation Steps
- [ ] **Step 1**: Expand `CONCEPTS` to include new primitives (`OP_SUB`, `OP_MUL2`) and update priors.
- [ ] **Step 2**: Implement the `patch_graph` method in `DevelopmentalAgent` to traverse an HFN tree, locate the semantic locus of failure, and retrieve a patch.
- [ ] **Step 3**: Update `TaskRunner` to detect logical failures (valid execution, wrong result), compute the residual state delta, and trigger the patching phase.
- [ ] **Step 4**: Create the perturbation curriculum tasks.
- [ ] **Step 5**: Run the experiment and compare the efficiency of the "Self-Debugging" agent against the baseline "Restart-on-Fail" agent.