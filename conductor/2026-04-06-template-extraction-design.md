# SP50: Experiment 26 — Higher-Order Template Extraction (Refactoring)

## 1. Objective
To validate that the HFN architecture can autonomously discover **Higher-Order Invariants** across its procedural library. The system must transition from learning specific, concrete loop structures (e.g., `map_increment`, `map_double`) to extracting a generic **Higher-Order Template** (`MAP(list, procedure)`). This template should then be registered as an L3 structural prior, capable of taking any compatible `CALL_FUNC` node as an argument.

## 2. Background & Motivation
In Experiment 25 (SP49), the agent successfully achieved **Procedural Encapsulation**, learning to call a function (`increment`) from within a loop. However, the agent still constructs the boilerplate loop structure (`LIST_INIT -> FOR_LOOP -> ITEM_ACCESS -> CALL_FUNC -> LIST_APPEND -> RETURN`) from scratch for every new mapping task.

Human programmers recognize that the "Loop-Append-Return" structure is a constant template, and only the inner "Body Procedure" varies. By implementing **Autonomous Refactoring (Template Extraction)**, the agent can de-duplicate its knowledge. This introduces **Higher-Order Functions** into the HFN latent space, creating "Structural Immunity" to logic errors in boilerplate code and reducing planning complexity for any future mapping/filtering task to $O(1)$.

## 3. Setup & Environment

### Domain: Semantic Program Space (Higher-Order)
- **State Representation**: 9D semantic vector `[Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot]`.
    - `TemplateSlot`: A specialized state dimension indicating that a higher-order template is waiting for an argument (a procedure) to fill its internal logic block.
- **Structural Primitives (CONCEPTS)**:
    - Retain all primitives from SP49.
    - Introduce **Template Nodes** dynamically during the "Refactoring Dream" phase (e.g., `TEMPLATE_MAP`, `TEMPLATE_FILTER`).

### The Refactoring Curriculum
1.  **Task 1 (Concrete Map A)**: `map_increment` (Learn loop + `increment` call).
2.  **Task 2 (Concrete Map B)**: `map_double` (Learn loop + `double` call).
    *   *The Refactoring Trigger*: The agent recognizes structural isomorphism between the graphs of Task 1 and Task 2.
    *   *The Extraction*: The agent strips the specific `CALL_FUNC` nodes, replacing them with a generic `SLOT` node, and registers `TEMPLATE_MAP`.
3.  **Task 3 (Zero-Shot Template Application)**: `map_decrement` (Goal: `[2, 3] -> [1, 2]`).
    *   *The Test*: Instead of planning the 7-step loop from scratch, the agent retrieves the `TEMPLATE_MAP` prior and simply parameterizes it with the `decrement` function.

## 4. Architectural Enhancements

### 1. Structural Isomorphism Detection (The Refactoring Dream)
The `Observer` or `Forest` must include a mechanism to compare the topology of registered chunks. 
If two chunks, $C_A$ and $C_B$, share identical graph structures except for one specific node $N_A \neq N_B$, the agent isolates the shared structure $T = C_A \setminus N_A$.

### 2. Template Parameterization (Slots)
The extracted template $T$ contains a "Hole" or "Slot". In HFN terms, this is a node with a high variance ($\sigma$) that accepts any subgraph matching specific semantic constraints (e.g., a function that takes 1 input and returns 1 output).
During planning, if a `TEMPLATE_` node is selected, the planner must immediately execute a sub-query to fill the `TemplateSlot` before the state transition is complete.

### 3. Higher-Order Code Rendering
The `CodeRenderer` must understand how to render templates. It will render the boilerplate code and inject the rendered code of the "argument" node into the designated slot.
*Example*: `TEMPLATE_MAP(f)` renders as:
```python
res = []
for item in x:
    val = item
    val = f(val) # Injected slot
    if res is not None: res.append(val)
return res
```

## 5. Evaluation Metrics
1.  **Autonomous Discovery**: The agent must successfully detect the isomorphism between `map_increment` and `map_double` and register the `TEMPLATE_MAP` prior without manual intervention.
2.  **Zero-Shot Application**: The agent must solve `map_decrement` by utilizing the `TEMPLATE_MAP` prior in exactly **2 planning steps** (Retrieve Template -> Retrieve Argument).
3.  **Syntactic Integrity**: The rendered higher-order functions must compile and execute correctly in the Python sandbox.

## 6. Failure Modes to Watch
- **Over-Generalization**: The agent extracts a template that is too generic (e.g., stripping out the `RETURN` statement), making it useless or semantically ambiguous. (Solution: Strict topological matching rules for extraction).
- **Slot Mismatch**: The planner attempts to fill the `TemplateSlot` with an incompatible node (e.g., filling a `MAP` body with a `RETURN` primitive). (Solution: Semantic constraint checking on the slot).

## 7. Implementation Steps
- [ ] **Step 1**: Expand State to 9D (`TemplateSlot`).
- [ ] **Step 2**: Implement `RefactoringAgent` with an `extract_template(chunk_a, chunk_b)` method that identifies topological similarities and generates a `TEMPLATE_` node with a `SLOT`.
- [ ] **Step 3**: Update `CodeRenderer` to handle `TEMPLATE_` nodes and render their parameterized arguments.
- [ ] **Step 4**: Update DFS Planner to handle two-stage retrieval when a `TEMPLATE_` node is selected.
- [ ] **Step 5**: Create `hpm_fractal_node/experiments/tasks/higher_order_curriculum.json`.
- [ ] **Step 6**: Run the experiment: train on two concrete maps, trigger the Refactoring Dream, and test zero-shot application on a third task.