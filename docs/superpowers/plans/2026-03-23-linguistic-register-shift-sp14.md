# Linguistic Register Shift — Sub-project 14 Implementation Plan

**Goal:** Implement the "Linguistic pH" benchmark.

---

### Task 1: Linguistic Simulator
- [ ] **Create `benchmarks/linguistic_sim.py`**
  - Define `REGISTER_MAP`: Formal/Informal word pairs.
  - Implement `Word` data class with string and feature vector.
  - Implement `generate_register_shift_tasks(n_train, seed)`:
    - Train tasks: Formal pairs.
    - Test task: Informal "Trap" pair.

### Task 2: Linguistic Encoders
- [ ] **Create `benchmarks/linguistic_encoders.py`**
  - `LinguisticL1Encoder`: 32-dim character distribution.
  - `LinguisticL2Encoder`: 16-dim semantic features (root only).
  - `LinguisticL3Encoder`: 20-dim transformation delta.

### Task 3: 5-Level Benchmark Script
- [ ] **Create `benchmarks/structured_linguistic_l4l5.py`**
  - Train L4 on Formal context.
  - Inject Informal "Trap" task.
  - Print Surprise metric and Accuracy per register.

### Task 4: Verification
- [ ] **Create `tests/benchmarks/test_linguistic_v2.py`**
  - Smoke test for SP14.
- [ ] **Run benchmark** and confirm L5 Surprise > 0.2 on register shift.
