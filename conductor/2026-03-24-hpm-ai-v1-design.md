# HPM AI v1: Recursive, Self-Refactoring Synthetic Intelligence
**Design Specification (v1.1 - Aligned with HPM §1.25)**

## 1. Overview
The HPM AI v1 is a self-evolving system that transitions from **Stochastic Patching** to **Manifold Alignment**. It treats its source code as a primary algebraic substrate, modelling the logic of the code rather than guessing changes. It utilizes recursive recombination of relational invariants to evolve its own architecture toward maximum elegance and performance.

## 2. Core Architecture

### 2.1 The Knowledge Mine (Cross-Domain Anchoring)
- **Substrates**: `MathSubstrate`, `PyPISubstrate`, `WikipediaSubstrate`.
- **Function**: Mine external relational laws (e.g., "Big O" complexity, "Pareto Law").
- **Anchoring**: Uses the **SVD_PROCRUSTES** alignment engine to map these external invariants directly onto the Agent's weight-pruning and resource-cost logic.

### 2.2 Relational Code Ingestion
- **Module**: `hpm_ai_v1.substrates.code_substrate`
- **Function**: Standardized `LocalCodeSubstrate` parsing `hpm/` and `benchmarks/` into AST.
- **Representation**: Maps AST sub-trees to **Relational Tokens** in a high-density L1-L3 manifold.

### 2.3 Generative Recombination Head (The Decoder)
- **Module**: `hpm_ai_v1.transpiler.decoder`
- **Logic**: Replaces stochastic mutation with **Relational Recombination**.
- **Process**:
  1. Queries the Librarian for high-weight L3 patterns (successful optimizations) from other agents.
  2. Uses the `RecombinationOperator` to "cross" the target function's AST with discovered optimization laws.
  3. Reconstructs the **Child AST** using `ast.unparse()` for direct code generation.

### 2.4 Autonomous Succession Loop
- **Module**: `hpm_ai_v1.main`
- **Execution**: Continuous **Generation Succession** model.
- **Recursive Cycle**:
  1. Propose Mutation -> 2. AST-Native Refactor -> 3. Sandbox Validation -> 4. L5 Pareto Gating -> 5. Succession (Commit).
- **Stagnation Trigger**: If Surprise ($S$) remains below 0.01 for 5 generations, the `AutonomousBenchmarkGenerator` introduces new constraints (e.g., restricted memory) to stimulate further evolution.

### 2.5 Elegance-First Gating (L5 Compiler)
- **The L5 Monitor**: Enforces the **HPM Elegance Principle**.
- **Pareto Constraint**: A mutation is rejected unless:
  - $(Accuracy_{new} \geq Accuracy_{old})$ AND $(NodeCount_{new} < NodeCount_{old})$
  - OR it reduces `cost_time` by >15% if complexity increases.
- **Elegance Recovery**: Prioritizes the Minimal Description Length (MDL) implementation of any relational law.

### 2.6 Lineage & Persistence
- **Module**: `hpm_ai_v1.store.concurrent_sqlite`
- **Parentage**: Captures the **Lineage** of every mutation, storing which L3 Law was used to forge the new version and its ancestral L4 weights.
