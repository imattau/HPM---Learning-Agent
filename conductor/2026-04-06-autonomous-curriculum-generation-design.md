# SP51: Experiment 27 — Autonomous Curriculum Generation (Self-Play & Curiosity)

## 1. Objective
To validate that the HFN architecture can transition from human-provided training curricula to **Unsupervised Algorithmic Discovery**. The agent must enter a "Self-Play" phase where it autonomously composes its structural priors, renders them, executes them against random inputs, and automatically catalogs novel, non-crashing transformations as new, formal `Task` definitions in its persistent knowledge base.

## 2. Background & Motivation
In all previous experiments (SP44-SP50), the agent relied on a rigid `curriculum.json` file. It was told exactly what semantic state (`goal_state`) to achieve. While this proved the HFN's ability to synthesize logic, it remained a reactive problem-solver rather than an autonomous explorer.

True AGI must build its own world model. By implementing an AlphaZero-style **Unsupervised Play** phase, the agent can use its existing library (Sequences, Loops, Conditionals, and Templates) to randomly explore the boundaries of its latent space. When a randomly generated program produces a stable, novel state transformation (e.g., discovering how to reverse a list, sort numbers, or compute a sum), the agent formalizes that transformation into a new, reusable chunk. This effectively "bootstraps" a massive, grounded knowledge base without human labels.

## 3. Setup & Environment

### Domain: Semantic Program Space (Unbounded)
- **State Representation**: 9D semantic vector `[Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot]`.
- **The Sandbox**: A highly robust `PythonExecutor` that provides random, typed inputs (e.g., lists of integers, single integers) and catches all exceptions (Timeout, TypeError, RecursionError).

### The Autonomous Discovery Loop
1.  **Generative Sampling**: The agent randomly selects 2-4 nodes from its `TieredForest` (Priors, Library Functions, or Templates) and composes them into a temporary HFN tree.
2.  **Sandbox Execution**: The `CodeRenderer` generates the Python code. The `PythonExecutor` runs the code against a suite of random inputs.
3.  **Semantic Mapping (The Oracle)**: If the code executes without errors across all inputs, the agent observes the *Initial State* and the *Final Output State*. It maps this transformation to a 9D Semantic Delta.
4.  **Novelty Detection**: The agent checks its `Observer`. If the computed Semantic Delta is "novel" (high residual surprise compared to existing library functions), the agent registers the tree.
5.  **Task Formalization**: The agent automatically creates a new `Task` definition (`id: auto_discovered_task_X`, `expected_output: Y`) and persists it.

## 4. Architectural Enhancements

### 1. The Curiosity Agent (`SelfPlayAgent`)
A new top-level agent that orchestrates the discovery loop. It does not take a `goal_state` as input. Instead, it takes a `budget` (e.g., 500 self-play cycles) and a `sandbox_inputs` list.

### 2. State Delta Oracle (Inverse Planning)
Currently, we provide the agent with a goal state. We must build a lightweight `StateOracle` that takes a Python `input` and a Python `output` and computes the 9D Semantic State representation of that transformation. This allows the agent to self-label its discoveries.

### 3. Novelty Filter (The Gatekeeper)
Random generation will produce many useless programs (e.g., `x = 1; x += 1; x -= 1; return x`). The `NoveltyFilter` evaluates the execution trace and the semantic delta. If the delta is trivial (e.g., `[0, 0, 0...]`), or if it maps exactly to an already-learned chunk (low residual surprise), the tree is discarded.

## 5. Evaluation Metrics
1.  **Discovery Yield**: The total number of *novel, non-trivial, non-crashing* functions discovered and registered within a given budget (e.g., 100 cycles).
2.  **Algorithmic Emergence**: Qualitative assessment of the discovered functions. Did the agent stumble upon recognizable algorithms (e.g., `sum`, `constant_array`, `filter_odd`)?
3.  **Curriculum Bootstrapping**: Can the auto-generated curriculum be successfully re-solved by a fresh agent from scratch?

## 6. Failure Modes to Watch
- **The Infinite Loop Trap**: Random composition of `FOR_LOOP` and condition blocks easily leads to infinite loops. (Solution: Hard execution timeouts in the `PythonExecutor`).
- **Combinatorial Explosion of Junk**: The agent might register thousands of functionally identical but structurally slightly different programs. (Solution: Strict geometric deduplication in the `TieredForest` using `hausdorff_absorption_threshold`).
- **Semantic Ambiguity**: The `StateOracle` might not accurately capture complex transformations (e.g., sorting) in just 9 dimensions. (Solution: Define novelty simply by input/output mapping stability for now, expanding semantic dimensions later if needed).

## 7. Implementation Steps
- [ ] **Step 1**: Implement `StateOracle` to dynamically compute a 9D state vector from raw Python `input` and `output`.
- [ ] **Step 2**: Build `SelfPlayAgent.explore(cycles)`, incorporating the `CodeRenderer` and a hardened `PythonExecutor` with timeouts.
- [ ] **Step 3**: Implement the `NoveltyFilter` to reject trivial or redundant state transformations.
- [ ] **Step 4**: Run a 100-cycle Unsupervised Play phase.
- [ ] **Step 5**: Export the discovered, stable functions into `hpm_fractal_node/experiments/tasks/auto_curriculum.json` and persist the structures in the `TieredForest`.