# SP23: Project Sovereignty - Cascading Dependency Repair Implementation Plan

**Goal**: Implement Global Project Ingestion and Cascading Dependency Repair to resolve the "Global Contradiction" loop.

---

### Phase 1: Global Dependency Mapping
- [ ] **Extend `hpm_ai_v1/transpiler/mmr.py`**:
    - Update `MMRNode` to include `filepath` and `lineno`.
    - Implement `ProjectTopology` to track `exports` (Function/Class names) and `in_edges` (Call-sites).
    - Map the entire repository manifold by walking the directory tree.

### Phase 2: Multi-File Change-Sets
- [ ] **Update `hpm_ai_v1/core/mutator.py`**:
    - Implement a `ChangeSet` data structure to track mutations across multiple files.
    - When a function name or signature changes, use `ProjectTopology` to identify and mutate all impacted call-sites in dependent files.

### Phase 3: Recursive Repair Loop (Litmus Step)
- [ ] **Update `hpm_ai_v1/sandbox/executor.py`**:
    - Return full traceback logs on failure.
    - Implement `ErrorAnalyzer` to parse logs for `TypeError`, `AttributeError`, or `NameError`.
- [ ] **Update `hpm_ai_v1/core/l5_compiler.py`**:
    - Implement the `Litmus Repair Loop`.
    - If a sandbox run fails with a dependency error, trigger a "Sub-Generation Repair" to align the calling code.

### Phase 4: Project-Root Sovereignty
- [ ] **Modify `hpm_ai_v1/main.py`**:
    - Set default `repo_path` to `./`.
    - Implement `CascadingSuccessionController` to orchestrate multi-file repairs.
    - Execute a live refactor on `hpm_ai_v1/core/l5_compiler.py` and ensure project-wide consistency.

---

## Verification
- **Test Case**: Rename a method in `L5Compiler`.
- **Expected Outcome**: The AI automatically identifies the call-site in `main.py`, generates a repair patch for it, and the project-wide `pytest` suite passes.
