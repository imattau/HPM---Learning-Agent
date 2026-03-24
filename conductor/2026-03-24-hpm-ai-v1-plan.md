# HPM AI v2 Implementation Plan

**Goal**: Establish Abstract Relational Intelligence via MMR, escape local minima via Bloat Windows, and prove law portability via SP19.

---

### Phase 1: The Ontological Leap (MMR)
- [ ] **Implement MMR Engine**: Create `hpm_ai_v1/transpiler/mmr.py`.
  - Implement `to_relational_graph(ast_node)` using a generic node/edge representation.
  - Implement `from_relational_graph(graph)` back to Python AST.
- [ ] **Update Recombination**: Modify `hpm_ai_v1/transpiler/decoder.py` to perform crossovers on MMR graphs instead of raw AST subtrees.

### Phase 2: Metacognitive Exploration (L5)
- [ ] **Implement Bloat Window**: Update `hpm_ai_v1/core/l5_compiler.py`.
  - Add `stagnation_counter`.
  - If `stagnation_counter > 3`, enable `allow_bloat = True` (permit <20% node increase for high-surprise logic).
- [ ] **Refine Pareto check**: Factor in the Novelty Score derived from L5 Surprise.

### Phase 3: Active Prior Injection
- [ ] **Mine Design Patterns**: Update `main.py` or a dedicated harvester to query `PyPISubstrate` for specific optimization patterns (e.g., Memoization).
- [ ] **Inject Blueprints**: Encode these snippets as L3 Relational Tokens and force them into the mutator's candidate pool.

### Phase 4: SP19 - The Rosetta Refactor (Cross-Modal Bridge)
- [ ] **Refactor Symmetry Check**: Task the HPM AI to optimize its own `L5Monitor` symmetry detection.
- [ ] **Export Symmetry Law**: Save the resulting L3 law to the global `Librarian`.
- [ ] **ARC Validation**: Create a benchmark test where an ARC agent pulls this code-derived "Symmetry Law" to solve a visual grid task.

### Phase 5: Verification & Generation Loop
- [ ] **Continuous Succession**: Run the `SuccessionController` for 10+ generations.
- [ ] **Audit Results**: Verify if the AI "tunneled" through complexity to find a fundamentally different (and better) algorithm.
