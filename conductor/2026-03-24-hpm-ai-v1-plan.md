# HPM AI v2.1 Implementation Plan

**Goal**: Transition to Vectorized Topologies and safe Unified Diff generation.

---

### Phase 1: Vectorized Code Topology
- [ ] **Vectorize `MMRNode`**: Update `hpm_ai_v1/transpiler/mmr.py`.
  - Create a static mapping of AST types to 32-dim orthogonal vectors (Basis Vectors).
  - Implement `vector_to_type(vec)` using cosine similarity.
- [ ] **Manifold Crossover**: Redefine the `StructuralTranspiler.crossover_mmr` to perform interpolation or subspace swapping between MMR graphs.

### Phase 2: Unified Diff & Sandbox Safety
- [ ] **Hardcode Diff Generator**: Implement a robust `UnifiedDiffGenerator` in `hpm_ai_v1/sandbox/executor.py` using `difflib`.
- [ ] **Patch-Based Sandbox**: Update `SandboxExecutor.evaluate_code` to apply a `.patch` file using the `patch` command, ensuring it handles indentation and line counts correctly.
- [ ] **Structural Immunity Review**: Update `L5Compiler` to check the patch content for obvious logical contradictions before acceptance.

### Phase 3: Autonomous Benchmark Expansion
- [ ] **Benchmark Generator**: Create `hpm_ai_v1/core/benchmark_generator.py`.
  - Implement `generate_conflict_benchmark(agent_id, store)` to create new `pytest` files that stress-test weak points in the agent's model.
- [ ] **Main Loop Integration**: Update `main.py` stagnation trigger to call the generator and add the new test file to the `pytest` run.

### Phase 4: Integration & Dialect Discovery
- [ ] **MMR Communication**: Allow agents to register MMR graphs (as L3 Relational Tokens) in the `PatternField` directly.
- [ ] **Generation Run**: Execute 10+ generations on a target file and monitor the evolution of the "Geometric Dialect".
