# HPM AI v3.0 Implementation Plan

**Goal**: Transition to Multi-Agent Structured Hierarchy and Manifold-Based Cascading Repair.

---

### Phase 1: Agent Promotion
- [ ] **Refactor `L5Compiler` to `L5MonitorAgent`**:
    - Inherit from `Agent`.
    - Implement sensory mapping for "Dependency Breaks" ($S=1.0$).
    - Implement `evaluate_changeset` as a task-evaluator.
- [ ] **Refactor `CodeMutationActor` to `L4ArchitectAgent`**:
    - Inherit from `Agent`.
    - Use internal `L4GenerativeHead` to propose `ChangeSets`.

### Phase 2: Structured Orchestration
- [ ] **Implement `SovereignOrchestrator` in `hpm_ai_v1/main.py`**:
    - Use `StructuredOrchestrator` pattern.
    - Setup negotiation loop via `kappa` incompatibility matrix.
- [ ] **Dynamic Topology Integration**:
    - Link `ProjectTopology` updates to the `Librarian` / `ContextualPatternStore`.

### Phase 3: Manifold-Based Cascading Repair
- [ ] **Implement "Structural Shift" Broadcasting**:
    - When a module is refactored, broadcast its new L3 manifold vector.
- [ ] **Automated Litmus Turns**:
    - Trigger repair tasks for dependent agents to align their code with the new manifold.
- [ ] **Metacognitive Repair Turn**:
    - Implement logic to handle "Global Contradictions" by forcing a multi-point `ChangeSet`.

### Phase 4: Project-Root Sovereignty
- [ ] **Sovereign Ingestion**:
    - Confirm `CodeSubstrate` is fully functional at project root `./`.
    - Execute a live refactor across multiple core files (e.g., `hpm/store/` and `hpm/agents/`) simultaneously.

### Phase 5: Dialect Genesis (SP20)
- [ ] **Manifold Logic Testing**:
    - Use `InternalVMSubstrate` to verify logic in the MMR Manifold before Python unparsing.
    - Confirm "Grammar Errors" in Python are handled as superficial L1 fixes.
