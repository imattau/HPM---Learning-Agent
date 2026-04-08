# SP49: Experiment 25 — Modular Procedural Abstraction (Functions)

## 1. Objective
To validate that the HFN architecture can transition from **Inline Chunking** (expanding learned macros directly into a task script) to **Parameterized Procedure Calls**. The system must learn to abstract a previously learned chunk into a callable "Black Box Function" with a defined interface (inputs/outputs) and a local scope, allowing it to be invoked repeatedly across different programmatic contexts.

## 2. Background & Motivation
In Experiment 21 (SP45: Recursive Complexity Scaling), the agent successfully created "Chunks" representing linear sequences like `add_one_to_item`. However, these chunks were effectively macro-expansions. When solving `map_add_one`, the agent literally re-rendered the `add_one` sequence inside the for-loop.

As complexity scales, this leads to massive, unreadable program traces and risk of variable name collisions. Human intelligence manages complexity via **Encapsulation**. A process like "Sorting" is abstracted into a function `sort(list)`. The internal workings are hidden from the caller. For HFN to achieve true Artificial General Intelligence scaling, it must learn to abstract chunks into **Modules (Functions)** that define boundaries between *What* a process does (its Semantic Delta) and *How* it does it (its internal subgraph).

## 3. Setup & Environment

### Domain: Semantic Program Space (Modularized)
- **State Representation**: 8D semantic vector `[Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStackDepth]`.
    - `CallStackDepth`: Prevents infinite recursion during rendering and execution.
- **Structural Primitives (CONCEPTS)**:
    - Add `DEF_FUNC`: The wrapper node that defines a local scope. `def func_id(x):`
    - Add `CALL_FUNC`: The invocation node. `val = func_id(val)`
- **The Modality Shift**:
    - The `CodeRenderer` must now support a dual-pass rendering. 
    1.  **Global Pass**: Renders all defined functions at the top of the file.
    2.  **Execution Pass**: Renders the main execution block that calls those functions.

### The Procedural Curriculum
1.  **Task 1 (The Kernel)**: `increment_val` (Goal: `[1] -> [2]`).
    *   *Result*: The agent learns `VAR_INP -> OP_ADD -> RETURN` and chunks it. We manually wrap this chunk in a `DEF_FUNC` to create a library procedure.
2.  **Task 2 (Simple Invocation)**: `map_increment` (Goal: `[1, 2, 3] -> [2, 3, 4]`).
    *   *The Test*: Instead of expanding the `OP_ADD` logic inline, the agent must retrieve the `CALL_FUNC` prior, parameterized with the `increment_val` semantic delta, and place it inside the `FOR_LOOP`.
3.  **Task 3 (Compositional Invocation)**: `filter_and_map_increment` (Goal: `[1, 2, 3, 4] -> [3, 5]`).
    *   *The Test*: A complex task combining filtering (from Exp 24) and the new modular `CALL_FUNC` logic.

## 4. Architectural Enhancements

### 1. The Procedural Library (Function Registry)
The `Forest` (or a dedicated `Library` subsystem) will act as a registry for defined functions.
When a chunk is successfully validated across multiple tasks, it undergoes a "Promotion" dream where it is encapsulated by a `DEF_FUNC` node. Its internal structure is frozen, and its external interface (its `mu` vector delta) becomes a new generic prior in the agent's retriever.

### 2. Parameterized `CALL_FUNC` Nodes
A `CALL_FUNC` node must hold a reference to the `DEF_FUNC` it intends to invoke. 
During planning, when the agent evaluates a `CALL_FUNC` node, it applies the semantic delta of the referenced function *without* needing to simulate or traverse the internal nodes of that function. This provides an $O(1)$ planning step for an $O(N)$ execution process.

### 3. Modular Code Rendering
The `CodeRenderer` needs an architecture upgrade to maintain a "Global Scope" string buffer (for `def` blocks) and a "Main Scope" string buffer. It will recursively extract dependencies: if the main block calls `func_A`, it must prepend the definition of `func_A` to the execution payload.

## 5. Evaluation Metrics
1.  **Rendering Fidelity**: The generated code must contain distinct `def` blocks and valid `function_call()` syntax, with proper variable scoping (avoiding global state pollution).
2.  **Planning Compression**: The number of DFS nodes explored for `map_increment` using `CALL_FUNC` must be strictly fewer than the number explored if it had to expand the `OP_ADD` chunk inline.
3.  **Reusability Success**: 100% success rate on the `filter_and_map_increment` task utilizing the library function.

## 6. Failure Modes to Watch
- **State Pollution**: The inner function accidentally modifies the global `res` list or `x` input directly due to poor renderer scoping. (Solution: Strict parameter passing `def func(x): return x_new`).
- **Dependency Hell**: The renderer fails to include the definition of a called function, leading to a `NameError`.
- **Retrieval Aliasing**: The planner cannot distinguish between `CALL_FUNC(add)` and an inline `OP_ADD` because their net state deltas are mathematically identical. (Solution: the `DEF_FUNC` chunk should have a slightly higher prior weight due to "promotion", making it the preferred path).

## 7. Implementation Steps
- [ ] **Step 1**: Expand `CONCEPTS` with `DEF_FUNC` and `CALL_FUNC`. Extend state to 8D.
- [ ] **Step 2**: Major overhaul of `CodeRenderer` to support dual-pass rendering (Function Definitions + Main Execution).
- [ ] **Step 3**: Update the DFS Planner in `ModularAgent` to handle $O(1)$ semantic jumps for `CALL_FUNC` nodes.
- [ ] **Step 4**: Create `hpm_fractal_node/experiments/tasks/procedural_curriculum.json`.
- [ ] **Step 5**: Run the experiment, inject the function into the library, and verify modular synthesis.