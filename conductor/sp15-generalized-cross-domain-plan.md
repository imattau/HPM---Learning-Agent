# SP15: Generalized Cross-Domain Alignment Implementation Plan

**Goal:** Implement and evaluate generalized transfer across 6 domains.

---

### Task 1: Create Multi-Domain Loaders
- [ ] **Modify `benchmarks/multi_domain_alignment.py` (New File)**
  - Implement loaders for `ds1000`, `chem`, and `linguistic`.
  - Reuse and adapt loaders for `math`, `phyre`, and `arc`.
  - Implement a universal `get_encoders(domain)` that supports all 6.

### Task 2: Implement Multi-Domain Procrustes
- [ ] **Develop `align_multiple_domains(matrices, ref_idx)`**
  - Inputs: List of $M_d$ matrices.
  - Outputs: Single $M_{global}$ and list of rotation matrices $R_d$.
  - Algorithm: Reference-based alignment (align all to matrices[ref_idx]).

### Task 3: Cross-Domain Execution
- [ ] **Implement `run_transfer_experiment(source_domains, target_domain)`**
  - Compute $M_{global}$ from source domains.
  - Evaluate on target domain using `score_with_anchor`.
- [ ] **Define the "Boss Fight" rotations**:
  - `[PhyRE, ARC, DS-1000] -> Chem`
  - `[DS-1000, Chem] -> Math`
  - `[Math, PhyRE, ARC, DS-1000, Chem] -> Linguistic`

### Task 4: Surprise Transfer (Advanced)
- [ ] **Implement `transfer_surprise_threshold(source_domain, target_domain)`**
  - Measure L5 Surprise on a register/pH shift in source.
  - Use that threshold to trigger "Latent Variable Discovery" in the target domain.

### Task 5: Verification & Reporting
- [ ] **Create `tests/benchmarks/test_multi_domain_alignment.py`**
  - Smoke test for 6-domain alignment.
- [ ] **Generate Final Report**:
  - Document accuracy gains/losses.
  - Confirm if the "Symbolic Gap" has been narrowed.
