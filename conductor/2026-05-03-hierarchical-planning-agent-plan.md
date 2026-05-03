# Hierarchical Planning Agent (NPM-style) Plan

## Objective
Implement the `HierarchicalPlanningAgent` for the "Multi-step prerequisite chains with subgoals" task family (specifically targeting the Nested Prerequisite Maze / NPM).

## Motivation
Currently, the `NestedPrerequisiteMazePlanner` solves the maze by generating candidate strategies, simulating them end-to-end, and scoring them. The new `HierarchicalPlanningAgent` introduces explicit, agent-managed goal stack decomposition. Instead of planning a full trajectory up front, the agent dynamically decomposes a high-level goal into prerequisites, manages them via a stack, and delegates execution of each subgoal to a `PatternEngine`.

## Key Components

### 0. Constraints
- **Core Isolation:** The `hpm_ai_v5/core` module MUST NOT be modified without explicit reason and prior approval. The agent should be implemented entirely in the `agents` layer and rely on the existing public API of `PatternEngine`.

### 1. Goal Stack Management
- A stack (LIFO) of active subgoals.
- The agent continually inspects the top goal.
- If the top goal has unmet prerequisites, those prerequisites are pushed onto the stack.

### 2. Goal Decomposition Logic
- A mechanism (e.g., an oracle or an inference rule base derived from the environment's dependency chain) to determine the prerequisites of a goal.
- Example: "reach_G" $\rightarrow$ requires "unlock_D2", which requires "collect_K2", etc.

### 3. Pattern Engine Integration
- **Engine Mapping:** "One PatternEngine (or one per subgoal)". We will start by instantiating a dedicated `PatternEngine` for the active subgoal, or using a single shared engine with its context scoped to the active subgoal.
- **Execution:** The top subgoal is passed to the engine. The engine's `act()` method is called with a subgoal-specific goal dictionary.

### 4. Completion Detection
- After the engine acts, the agent checks the new state/reward.
- If the state indicates the top subgoal is met (e.g., state change shows key is collected), the subgoal is popped from the stack.

## Implementation Steps

1. **Create `hpm_ai_v5/agents/hierarchical_planner.py`**
   - Define `HierarchicalPlanningAgent` implementing the `Agent` protocol or wrapping `BaseAgent`.
   - Implement `_decompose_goal(goal)` to query prerequisites.
   - Implement the `step(input)` loop:
     - Check subgoal completion.
     - Decompose if necessary.
     - Route to `PatternEngine`.

2. **Integrate with Nested Prerequisite Maze (NPM)**
   - Create a test or adapt the existing `nested_maze.py` benchmark to run with `HierarchicalPlanningAgent`.
   - Ensure the dependency chain (`collect_K1` -> `unlock_D1` -> etc.) is correctly parsed into the goal stack.

3. **Verification**
   - Write a unit test in `hpm_ai_v5/tests/test_hierarchical_planner.py` that verifies:
     - The goal stack correctly pushes `['collect_K1', 'unlock_D1', ...]` in reverse order.
     - The engine completes subgoals one by one.
     - The goal stack empties upon full completion.
