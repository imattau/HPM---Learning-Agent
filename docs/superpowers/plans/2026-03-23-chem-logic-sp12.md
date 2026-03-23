# Chem-Logic Benchmark — Sub-project 12 Implementation Plan

**Goal:** Implement the "Hidden Law" molecular benchmark.

---

### Task 1: Molecular Simulator
- [ ] **Create `benchmarks/chem_logic_sim.py`**
  - Implement `FunctionalGroup` registry (OH, CHO, COOH, COC, etc.).
  - Implement `Molecule` data class with SMILES and features.
  - Implement `generate_reaction_tasks(n_tasks, seed)` returning (Reactant, Product) pairs.
  - Implement `valence_check(molecule)`: Simulates chemical validity.

### Task 2: Chemical Encoders
- [ ] **Create `benchmarks/chem_logic_encoders.py`**
  - `ChemLogicL1Encoder`: SMILES character distribution (32-dim).
  - `ChemLogicL2Encoder`: Functional Group bits + Epistemic (16-dim).
  - `ChemLogicL3Encoder`: Transformation Deltas (20-dim).

### Task 3: 5-Level Benchmark Script
- [ ] **Create `benchmarks/structured_chem_logic_l4l5.py`**
  - Integrate simulator and encoders.
  - Implement three conditions: `l2l3`, `l4_only`, `l4l5_full`.
  - L5 should specifically use `valence_check` to penalize invalid predictions.

### Task 4: Verification
- [ ] **Create `tests/benchmarks/test_structured_chem_logic_l4l5.py`**
  - Standard smoke test implementation.
- [ ] **Run benchmark** and verify accuracy.
