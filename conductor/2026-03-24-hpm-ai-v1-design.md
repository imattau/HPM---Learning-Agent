# HPM AI v3.2: The Logic Forge
**Design Specification (v1.6 - Global Saliency & Algebraic MMR)**

## 1. Overview
HPM AI v3.2 transitions from a "Code Mimic" to a **Logic Forge**. It eliminates hardcoded file targeting in favor of **Manifold-Directed Saliency** and replaces string-based "simulation" with **Topological Invariant Verification**. It also introduces **Soft Pareto Gating** using a Lagrangian approach to prevent the "Zero-Sum Pruning Trap."

## 2. Core Architecture

### 2.1 Manifold-Directed Saliency (Architectural Agency)
- **Module**: `hpm_ai_v1.core.librarian.CodeLibrarian` / `hpm_ai_v1.core.l5_compiler.L5MonitorAgent`
- **Function**: Performs project-wide scans to identify refactoring targets automatically.
- **Metric**: Selects modules with the lowest **Pattern Stickiness** or highest **Structural Entropy** (Epistemic Residual $S$).
- **Impact**: The `SuccessionLoop` no longer requires a `target_file` parameter.

### 2.2 Algebraic MMR (Executable Invariants)
- **Module**: `hpm_ai_v1.transpiler.mmr` / `hpm_ai_v1.substrates.vm_substrate.InternalVMSubstrate`
- **Logic**: Maps all Python operations to **Manifold Basis Vectors** (32-dim).
- **Verification**: Instead of text-based simulation, it calculates the **Structural Identity** of the graph.
- **Equivalence**: If two graphs (Parent/Child) produce the same manifold output for identical input vectors, the logic is verified. If the Child is more elegant (MDL), it is promoted.

### 2.3 Soft Pareto Gating (Dynamic Lagrangian Weighting)
- **Module**: `hpm_ai_v1.core.l5_compiler.L5MonitorAgent`
- **Formula**: $Score = \Delta Accuracy - \lambda(\Delta Nodes)$.
- **Logic**: 
    - If a mutation reduces node count, it is accepted unless it breaks tests.
    - If it increases node count (during "Bloat Windows"), $\lambda$ drops to near zero, permitting complexity for high **Novelty Vectors** ($> 0.7$).
- **Impact**: Escapes the "Greedy Pruning" local optima by allowing temporary algorithmic investment.

## 3. Implementation Roadmap
- **Phase 1**: Refactor `mmr.py` and `vm_substrate.py` for purely algebraic, basis-vector driven logic.
- **Phase 2**: Implement `SaliencyScanner` in the `Librarian` to enable autonomous targeting.
- **Phase 3**: Refactor `L5MonitorAgent` to use Soft Pareto Gating with Lagrangian multipliers.
- **Phase 4**: Execute a project-wide autonomous refactoring run.
