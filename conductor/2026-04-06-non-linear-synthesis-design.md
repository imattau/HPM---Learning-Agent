# SP48: Experiment 24 — Non-Linear Program Synthesis (Logic Forks)

## 1. Objective
To validate that the HFN architecture can construct and render **non-linear graphs** containing conditional branches (e.g., `if/else` logic). This completes the core set of structured programming primitives (Sequence, Iteration, Selection) needed for Turing-complete algorithmic reasoning in the HFN latent space.

## 2. Background & Motivation
In previous experiments, the HFN has successfully mastered linear sequences (SP44), hierarchical folding for loops and maps (SP45), and local graph editing/patching (SP47). However, all constructed programs were fundamentally linear in their execution trace (even if nested structurally).

True programmatic reasoning requires **Selection (Logic Forks)**. The agent must learn that a transition is not just `[State, Action] -> [Delta]`, but can be conditioned on a **Predicate Node** (e.g., `if x % 2 == 0`). This requires the HFN to build non-linear trees where a single node splits the execution path into two distinct logical futures (True/False or If/Else branches).

## 3. Setup & Environment

### Domain: Semantic Program Space (Extended)
- **State Representation**: Expanding the semantic state vector to handle condition evaluation. `S_DIM = 7`: `[AccumulatorValue, ReturnedFlag, ListLength, ListTargetValue, IteratorActive, ListInitFlag, ConditionFlag]`.
    - `ConditionFlag`: Tracks the active predicate state (e.g., 1.0 for True, -1.0 for False, 0.0 for Not Evaluated).
- **Structural Primitives (CONCEPTS)**:
    - Add `COND_IS_EVEN`: Evaluates if the current item is even (`if val % 2 == 0:`). Sets `ConditionFlag`.
    - Add `COND_IS_ODD`: Evaluates if the current item is odd (`if val % 2 != 0:`). Sets `ConditionFlag`.
    - Add `BLOCK_ELSE`: Defines the alternate execution path (`else:`).

### The Conditional Curriculum
The curriculum introduces tasks that require conditional logic inside loops:
1.  **Task 1 (Base Transform)**: `return_input` (Verify stability of base loop/map).
2.  **Task 2 (Simple Filter)**: `filter_even` (Goal: `[1, 2, 3, 4] -> [2, 4]`).
    *   *The Test*: The agent must compose `FOR_LOOP -> ITEM_ACCESS -> COND_IS_EVEN -> LIST_APPEND`.
3.  **Task 3 (Conditional Transform)**: `map_even_double_else_keep` (Goal: `[1, 2, 3, 4] -> [1, 4, 3, 8]`).
    *   *The Capstone*: The agent must build a complex fork: `FOR_LOOP -> COND_IS_EVEN -> (True: OP_MUL2 -> LIST_APPEND) & (False: BLOCK_ELSE -> LIST_APPEND)`.

## 4. Architectural Enhancements

### 1. Branched Composition in Planning
The `DevelopmentalAgent.plan` method must be upgraded to construct branched trees. When a `COND_` node is selected, the planner must split its search, solving for the True path and (optionally) the False path, and merging them under a specialized `COMPOSE_BRANCH` parent node.

### 2. Scoped Non-Linear Rendering
The `CodeRenderer` must be updated to handle logic forks and dynamic indentation scopes.
*   `COND_IS_EVEN` renders `if val % 2 == 0:` and increments the indent level.
*   `BLOCK_ELSE` resets the indent to the parent level, renders `else:`, and increments the indent level for the alternate branch.

### 3. Predicate State Simulation
During planning, the DFS solver must simulate the `ConditionFlag` to properly route logic. For simplicity, the `s_goal` can provide a hint about the required condition (e.g., target values that are only even).

## 5. Evaluation Metrics
1.  **Branching Success Rate**: Ability to generate syntactically valid `if/else` blocks that execute without IndentationErrors or SyntaxErrors.
2.  **Logic Discovery Rate**: How many attempts it takes to discover that `LIST_APPEND` must be placed *inside* the conditional block rather than outside it.
3.  **Curriculum Mastery**: 100% success on the complex `filter_even` and `map_even_double_else_keep` tasks.

## 6. Failure Modes to Watch
- **Scope Leakage**: The `LIST_APPEND` operation accidentally falls outside the `if` block, acting as a standard map instead of a filter. (Solution: Strict tree traversal in the Renderer).
- **Dead Branches**: The agent generates an `else:` block but puts no operations inside it, or puts operations that don't affect the goal.
- **Combinatorial Explosion**: Branching effectively squares the search space for the DFS planner. (Solution: Rely on previously learned chunks like `compose(LIST_APPEND+RETURN)` to keep the planning depth manageable).

## 7. Implementation Steps
- [ ] **Step 1**: Expand `CONCEPTS` with `COND_IS_EVEN`, `COND_IS_ODD`, and `BLOCK_ELSE`.
- [ ] **Step 2**: Update `CodeRenderer` to handle branching, indentation, and `if/else` generation based on tree traversal.
- [ ] **Step 3**: Update `DevelopmentalAgent.plan` to support branched tree construction (`COMPOSE_BRANCH`).
- [ ] **Step 4**: Create `hpm_fractal_node/experiments/tasks/conditional_curriculum.json`.
- [ ] **Step 5**: Run the experiment and verify the successful construction and rendering of non-linear program graphs.