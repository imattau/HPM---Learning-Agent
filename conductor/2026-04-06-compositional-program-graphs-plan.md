# Redesign Plan: Compositional Program Graphs

## Diagnosis & Motivation
The user feedback highlighted that while the system now correctly renders code from semantic structure, the structure itself is still a flat, linear sequence (e.g., `[CONST_1, OP_ADD, RETURN]`). To unlock true abstraction, scalable reasoning, and complex behavior, the system must transition from linear sequences to **compositional program graphs** (DAGs/trees). 

By building compositional graphs, the system can "chunk" successful sequences into reusable subgraphs, turning multi-step reasoning into single-step retrieval for future tasks.

## 1. Hierarchical Semantic Structure
- Programs are no longer lists of strings. They are composed `HFN` nodes with children.
- A program is represented as a tree of operations, linked by relational composition. For example, a `COMPOSE` node has two children representing operations executed in sequence.

## 2. Graph-Based Code Renderer
- Refactor `CodeRenderer.render(node: HFN) -> str` to be a recursive tree-walker.
- **Primitives**: If a node is a leaf primitive (e.g., `CONST_1`), it renders directly to `x = 1`.
- **Composites**: If a node has children, the renderer recursively renders the children and concatenates them. 
- **Example**: An HFN node representing `COMPOSE(CONST_1, OP_ADD)` recursively resolves to `x = 1 \n x += 1`.

## 3. Compositional Planning & Chunking
- Update `DevelopmentalAgent.plan` to construct and return a composed `HFN` tree rather than a list.
- **Bottom-Up Chunking**: When the DFS planner finds a successful sequence of steps (e.g., Step A -> Step B -> Step C), it bundles them into a new parent HFN node.
- **Structural Memory**: Upon a successful task execution, this new composite node is permanently registered in the `Forest` and its weight is boosted. 
- **The Result**: The sequence `CONST_1 -> OP_ADD` becomes a single, reusable node in the latent space. Future planning can retrieve this chunk in one step, drastically reducing the search space for complex tasks like `return 5`.

## 4. Enhanced Execution Loop
- `TaskRunner.run_task` will receive the root `HFN` node of the plan.
- The `CodeRenderer` walks the node to generate the Python string.
- If successful, the entire subgraph (and the new composite parent) receives credit assignment, embedding the new abstraction into the agent's priors.

## Implementation Steps
- [ ] **Step 1: Refactor `CodeRenderer`**: Implement recursive traversal of HFN nodes to generate code.
- [ ] **Step 2: Refactor `DevelopmentalAgent.plan`**: Modify the planner to compose and return hierarchical HFN nodes instead of flat lists.
- [ ] **Step 3: Implement Chunking (Subgraphs)**: Update `run_task` to register successful multi-step plans as new, permanent composite nodes in the `Forest`.
- [ ] **Step 4: Verification**: Test on a curriculum of `return_constant` tasks to verify that the agent builds, stores, and subsequently reuses composed subgraphs (e.g., reusing an `ADD_2` chunk to quickly solve `return 4`).