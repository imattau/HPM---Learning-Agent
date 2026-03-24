# HPM AI v2.1: Geometric Relational Intelligence
**Design Specification (v1.3 - Vectorized MMR & Dialect Genesis)**

## 1. Overview
HPM AI v2.1 hardens the "Synthetic Intelligence" loop by transitioning the Middle-Manifold Representation (MMR) from string-labeled graphs to **Vectorized Topologies**. It introduces **Autonomous Benchmark Expansion** to drive open-ended evolution and implements a formal **Unified Diff Decoder Head** for safe, reviewed self-modification.

## 2. Core Architecture

### 2.1 Vectorized MMR (Code-as-Geometry)
- **Module**: `hpm_ai_v1.transpiler.mmr`
- **Function**: Replaces node labels with **L3 Embeddings**.
- **Logic**:
    - Every AST node type is mapped to a coordinate in `L3_CODE_SPACE` (32-dim).
    - Geometric distance between vectors represents semantic similarity (e.g., `Add` is closer to `Sub` than to `If`).
    - **Manifold Crossover**: Redefined as interpolation and subspace grafting between relational graphs.

### 2.2 Autonomous Benchmark Generator
- **Module**: `hpm_ai_v1.core.benchmark_generator`
- **Trigger**: Stagnation ($S < 0.05$ for 3 generations).
- **Function**: Synthesizes new test cases targeting **Conflict Pairs** in the agent's current model.
- **Example**: If complexity between $O(n \log n)$ and $O(n)$ is unresolved, generate a benchmark with $N=10^7$ to create clear performance divergence.

### 2.3 Unified Diff Decoder Head
- **Module**: `hpm_ai_v1.sandbox.executor`
- **Logic**: Moves away from direct file replacement to formal **Unified Diffs** (`.patch`).
- **Gating**: L5 reviews the patch for **Structural Immunity**, ensuring the delta doesn't introduce logical contradictions before applying it to the main repository.

### 2.4 Dialect Genesis (SP20)
- **Objective**: Direct agent-to-agent communication via the MMR manifold.
- **Process**: Agents exchange "High-Density Logic Tokens" (Relational Vectors) instead of Python strings, effectively creating a private internal dialect for self-simulation.

## 3. Implementation Roadmap
- **Phase 1**: Vectorize the MMR engine.
- **Phase 2**: Implement the Unified Diff generator and patch-based sandbox.
- **Phase 3**: Implement the Autonomous Benchmark Generator stub.
- **Phase 4**: Integrate the stagnation-driven synthesis loop in `main.py`.
