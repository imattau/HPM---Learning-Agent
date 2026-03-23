# SP13: Chem-Logic II Implementation Plan

**Goal:** Stress-test HPM stack with Competitive Inhibition and Latent pH.

---

### Task 1: Update Simulator for Ambiguity
- [ ] **Modify `benchmarks/chem_logic_sim.py`**
  - Add `PRIORITY` dict mapping functional groups to reactivity scores (e.g., amine > hydroxyl).
  - Add `ENV_STATE` (pH) to the task generation.
  - Implement `apply_ambiguous_reaction(reactant, reagent, ph)`:
    - If multiple sites, choose by `PRIORITY` unless `ph` overrides (e.g., amine protonation).
  - Return `latents` in the task dict (hidden from the agent).

### Task 2: Update Encoders
- [ ] **Modify `benchmarks/chem_logic_encoders.py`**
  - `ChemLogicL3Encoder`: Update to handle multi-site deltas.
  - Ensure encoders remain "blind" to the hidden `ph` variable.

### Task 3: Develop Ambiguity Benchmark
- [ ] **Create `benchmarks/structured_chem_logic_v2.py`**
  - Implement the "Ranking" logic:
    - Condition `l2l3`: Fails on multi-site because it doesn't know priority.
    - Condition `l4l5`: L5 should flag surprise when the "intuitive" rule is out-competed.
  - Implement "Latent pH" logic:
    - Provide training pairs with varying pH (hidden).
    - Measure L5 Surprise spike at the transition points.

### Task 4: Verification & Analysis
- [ ] **Create `tests/benchmarks/test_chem_logic_v2.py`**
  - Smoke tests for the new ambiguous tasks.
- [ ] **Run and Document Results**:
  - Show how L5 Surprise scales with environmental uncertainty.
  - Confirm `l4l5` accuracy > `l4_only` in competitive scenarios.
