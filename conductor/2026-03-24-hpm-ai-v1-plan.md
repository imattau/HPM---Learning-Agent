# HPM AI v3.1 Implementation Plan

**Goal**: Establish Dialect Sovereignty via Token-Native Execution and Multi-File Architectural Forging.

---

### Phase 1: Token-Native Execution
- [ ] **Refactor `InternalVMSubstrate`**:
    - Implement a runner that operates on 32-dim Relational Vectors (embeddings) directly.
    - Map basis vectors to executable operations (Add, Sub, etc.) without string checks.
    - Enable batch simulation of MMR graph variations.

### Phase 2: Manifold-Only Exploration
- [ ] **Decouple L4/L5 Loop from Python**:
    - Update `SovereignOrchestrator` to perform internal exploration (L4 intuition -> L5 verification) within the `InternalVMSubstrate`.
    - Only trigger `ast.unparse()` and `SandboxExecutor` once a manifold configuration passes L5 internal gating.

### Phase 3: Architectural Forging (Multi-File Synthesis)
- [ ] **Identify Structural Gaps**:
    - Implement a "Manifold Redundancy" analyzer in the `Librarian`.
    - Detect overlap between `hpm/store/sqlite.py` and `hpm_ai_v1/store/concurrent_sqlite.py`.
- [ ] **Forge Unified ChangeSet**:
    - Command `L4ArchitectAgent` to synthesize a unified Persistent Storage module.
    - Generate dependency repairs for all modules that formerly used the split implementations.

### Phase 4: Verification & Succession
- [ ] **Architectural Merger Execution**:
    - Run the loop to replace split storage with the unified version.
    - Verify that all project tests pass with the new architecture.
- [ ] **Report Results**:
    - Measure generation speed increase (Manifold vs. Python Sandbox).
    - Document the reduction in Manifold Entropy.
