# HPM AI v3.2 Implementation Plan

**Goal**: Establish the "Logic Forge" via Global Saliency, Algebraic MMR, and Soft Pareto Gating.

---

### Phase 1: Purely Algebraic MMR & VM
- [ ] **Refactor `mmr.py`**:
    - Eliminate all string-based node matching in `from_relational_graph`.
    - Map AST operations directly to their 32-dim Basis Vectors.
- [ ] **Update `InternalVMSubstrate`**:
    - Implement **Topological Invariant Verification**: compare manifold outputs of Parent vs. Child graphs across random input vectors.
    - Treat logic as a functional dataflow graph.

### Phase 2: Global Saliency Scanner
- [ ] **Implement `SaliencyScanner` in `Librarian`**:
    - Add `identify_high_entropy_module()` method.
    - Measure **Pattern Stickiness** (reusability) and **Epistemic Residual** across the manifold.
- [ ] **Refactor `SovereignOrchestrator`**:
    - Remove `target_file` from constructor and `run_succession_loop`.
    - Loop logic: Query Librarian for next salient target -> Initiate refactor.

### Phase 3: Soft Pareto Gating
- [ ] **Update `L5MonitorAgent`**:
    - Implement the Lagrangian score formula: $Score = \Delta Acc - \lambda(\Delta Nodes)$.
    - Implement dynamic $\lambda$ adjustment (drop to zero during Stagnation/Bloat Windows).
    - Ensure node reductions are auto-accepted unless tests fail.

### Phase 4: Project-Wide Autonomous Refactoring
- [ ] **Open-Ended Succession Run**:
    - Launch the orchestrator on the project root without a target.
    - Observe the AI autonomously traversing modules (e.g., `store/`, `evaluators/`) to reduce project-wide structural entropy.
- [ ] **Audit Results**:
    - Report the number of unique modules refactored autonomously.
    - Verify that the Lagrangian multiplier successfully enabled "Algorithmic Tunnelling."
