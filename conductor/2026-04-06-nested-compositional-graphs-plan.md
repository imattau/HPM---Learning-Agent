# Redesign Plan: True Compositional Program Graphs

## Diagnosis & Motivation
The user feedback highlighted that while the system now returns an HFN node representing the program, the structure is still a flat list of children attached to a single root. This prevents true hierarchical abstraction and reuse. 

To achieve true compositionality, the system must build nested graphs (trees) where `(Step A -> Step B)` becomes a new chunk, which can then be composed with `Step C` to form `((A -> B) -> C)`.

## 1. Nested Composition in Planning
- **Problem**: The planner currently appends all nodes to a single root.
- **Solution**: Refactor `DevelopmentalAgent.plan` to fold nodes hierarchically.
    - Instead of `root.add_child(n)` for all `n`, use a `fold_left` approach.
    - `current = path_nodes[0]`
    - `for n in path_nodes[1:]:`
        - `parent = HFN(...)`
        - `parent.add_child(current)`
        - `parent.add_child(n)`
        - `current = parent`
    - `return current`

## 2. Refined Code Renderer
- **Problem**: The renderer uses fragile string matching (`in node.id`) which breaks on composite node IDs like `plan_CONST_1_OP_ADD`. It also uses execution hacks like `if 'x' in locals()`.
- **Solution**: 
    - The renderer must distinguish between leaf primitives and internal composite nodes.
    - If a node is a known primitive (checking its `mu` action slice or a clean ID flag), render it directly.
    - If a node is a composite, recursively render its left and right children and concatenate.
    - Remove execution hacks. Assume the structure is valid (e.g., `x = 1` always precedes `x += 1` if the structure is correct).

## 3. Subgraph Chunking & Reuse
- **Problem**: The agent creates chunks but doesn't strongly prefer them in future planning.
- **Solution**: 
    - When a nested composite node is successfully executed, register it in the Forest and aggressively boost its weight.
    - The DFS planner naturally considers these composite nodes because they now exist in the Forest with high weight. If retrieved, a composite node represents a multi-step jump in state space.

## 4. Execution Semantic Purity
- **Problem**: The execution loop still relies on heuristic state updates during planning.
- **Solution**: Maintain the heuristics for now to bootstrap planning, but rely on the `Evaluator` and `Observer` for ground truth.

## Implementation Steps
- [ ] **Step 1: Refactor `DevelopmentalAgent.plan`** to construct nested binary trees (`COMPOSE(left, right)`) instead of flat lists.
- [ ] **Step 2: Refactor `CodeRenderer`** to cleanly traverse binary trees without relying on brittle ID string matching.
- [ ] **Step 3: Update `TaskRunner`** to chunk and register these new nested structures.
- [ ] **Step 4: Verify** that the system successfully chunks `(CONST_1 -> OP_ADD)` and reuses it.