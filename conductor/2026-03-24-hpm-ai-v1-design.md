# HPM AI v3.1: Dialect Sovereignty
**Design Specification (v1.5 - Token-Native Execution & Architectural Forging)**

## 1. Overview
HPM AI v3.1 achieves **Dialect Sovereignty** by removing the "Python Crutch." It enables high-density execution of relational logic directly within the MMR manifold and transitions from file-patching to **Architectural Forging**—the autonomous synthesis and merging of entire modules based on manifold topology.

## 2. Core Architecture

### 2.1 Token-Native Execution (Direct Manifold Simulation)
- **Module**: `hpm_ai_v1.substrates.vm_substrate.InternalVMSubstrate`
- **Logic**: Executes MMR graphs using the 32-dimensional basis vectors directly.
- **Optimization**: Verification of logic happens at "Manifold Speed" (thousands of iterations per second), bypassing Python's `ast` and `subprocess` overhead during internal exploration.
- **Export**: `ast.unparse()` is only triggered during a **Succession Event** (Final Promotion).

### 2.2 Multi-File Architectural Forging
- **Module**: `hpm_ai_v1.core.mutator.L4ArchitectAgent`
- **Function**: Identifies "Structural Gaps" or "Manifold Redundancy" (e.g., duplicate storage logic).
- **Action**: Proposes a **Global ChangeSet** that can synthesize entirely new modules or merge existing ones (e.g., forging `hpm/store/unified_persistent_store.py`).
- **Dependency Edge Synthesis**: Automatically generates the required `ImportNodes` and `CallEdges` for the new architecture.

### 2.3 Metacognitive Gating (Manifold Verification)
- **The L5 Monitor**: Now gates logic based on **Manifold Equivalence**. It compares the output of the new Relational Token sequence against the desired L3 Relational Law within the VM substrate.
- **Pareto Efficiency**: Measured in the manifold (Token Operations per second) before real-world cost verification.

### 2.4 Structural Curiosity
- **Mechanism**: Triggered by **High Manifold Entropy**.
- **Action**: If two modules (e.g., `hpm/store/sqlite.py` and `hpm_ai_v1/store/concurrent_sqlite.py`) have high relational similarity but separate implementations, the L4 Architect is commanded to "Forge a Unified Manifold."

## 3. Implementation Roadmap
- **Phase 1**: Implement Token-Native Execution in `InternalVMSubstrate`.
- **Phase 2**: Update the exploration loop to stay within the MMR manifold.
- **Phase 3**: Implement "Structural Gap" detection and Architectural Forging.
- **Phase 4**: Execute the first "Architectural Merger" (Unified Persistent Store).
