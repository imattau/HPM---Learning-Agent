# SP10: Delta Alignment — Structure-Preserving Cross-Domain Transfer

## Goal

Enable genuine cross-domain analogical transfer by aligning the *relational topology* (ΔL2→ΔL3
mappings) across domains via Procrustes rotation, rather than aligning raw feature distributions.
Tests whether HPM's L2→L3 abstraction is structurally invariant across physics, algebra, and
visual pattern completion.

## Background

SP9 (naive zero-padding) produced a negative result: cross-domain L4 substantially underperformed
domain-native L2L3 on all three leave-one-domain-out rotations. Root cause: zero-padding creates
heterogeneous feature spaces — Math L2 and PhyRE L2 vectors are structurally incomparable.

Hypothesis: domains share relational structure (how features *change*) even when absolute feature
values are incomparable. Align the ΔL2→ΔL3 operators via Procrustes, then use the shared
operator for cross-domain scoring.

## Approach: Relational Delta Alignment

### Three Phases

**Phase 1 — Local delta matrices (per training domain)**

For each training domain with N train pairs, compute all N(N-1) pairwise deltas:
- ΔL2_ij = L2_i − L2_j (zero-padded to 14-dim)
- ΔL3_ij = L3_i − L3_j (zero-padded to 14-dim)

Fit local mapping M_d ∈ ℝ^{14×14} via ridge regression: M_d @ ΔL2 ≈ ΔL3.

N(N-1) observations (vs N-1 sequential) provides a stable local operator even with 3-4 train pairs.

**Phase 2 — Procrustes alignment → shared matrix**

Given M_d1 and M_d2 from two training domains:
- Compute SVD of M_d2 @ M_d1ᵀ = U Σ Vᵀ
- Check determinant: if det(V @ Uᵀ) < 0, flip last column of V (force proper rotation R ∈ SO(14))
- R = V @ Uᵀ
- M_shared = (M_d1 + Rᵀ @ M_d2) / 2

**Phase 3 — Per-task anchor scoring (test domain)**

For each test task:
- Compute mean anchor (L2_ref, L3_ref) = centroid of all train pairs
- For each candidate: ΔL2 = L2_cand − L2_ref (padded), ΔL3_pred = M_shared @ ΔL2
- ΔL3_actual = L3_cand − L3_ref (padded)
- Score = −‖ΔL3_actual − ΔL3_pred‖ (higher = better)
- Return idx of highest-scoring candidate

### Three Leave-One-Domain-Out Rotations

| Train domains    | Test domain |
|-----------------|-------------|
| Math + PhyRE    | ARC         |
| Math + ARC      | PhyRE       |
| PhyRE + ARC     | Math        |

## Components

### benchmarks/phyre_delta_alignment.py

- `compute_delta_pairs(tasks, domain, l2_enc, l3_enc) -> list[tuple[np.ndarray, np.ndarray]]`
  All N(N-1) pairwise (ΔL2, ΔL3) from train pairs, zero-padded to 14-dim.

- `fit_domain_matrix(delta_pairs, alpha=0.01) -> np.ndarray`
  Ridge regression: M_d ∈ ℝ^{14×14}.

- `procrustes_align(M1, M2) -> np.ndarray`
  SVD alignment with determinant check (R ∈ SO(14)). Returns M_shared = (M1 + Rᵀ @ M2) / 2.

- `score_with_anchor(task, domain, M_shared, l2_enc, l3_enc) -> int`
  Centroid anchor, scores candidates by −‖ΔL3_actual − ΔL3_pred‖, returns best idx.

- `run_rotation(train_domains, test_domain, n_per_family=15) -> dict[str, float]`
  Builds M_shared, evaluates l2l3 baseline and delta_alignment on test domain.

- `main()`
  All 3 rotations, prints results table including SP9 cross_domain_l4 for comparison.

### tests/benchmarks/test_phyre_delta_alignment.py

- Assert M_d shape is (14, 14) for each domain
- Assert Procrustes R is orthogonal: ‖R @ Rᵀ − I‖ < 1e-10
- Assert det(R) ≈ +1.0 (proper rotation, not reflection)
- Assert pairwise delta count = N*(N-1) for N training pairs
- Smoke: fit Math+PhyRE, eval ARC, accuracy in [0, 1]

### Unchanged

- `L4GenerativeHead`, all encoders, task generators, phyre_tasks.pkl

## Expected Output

```
SP10 Delta Alignment Benchmark
Train domains      Test    l2l3   cross_domain_l4  delta_alignment
------------------------------------------------------------------
Math + PhyRE       ARC     0.800       0.267           ?
Math + ARC         PhyRE   0.583       0.167           ?
PhyRE + ARC        Math    1.000       0.222           ?
```

## Success Criteria

- **Strong:** delta_alignment > l2l3 on ≥2 rotations — Procrustes alignment enables genuine analogical transfer
- **Partial:** delta_alignment > cross_domain_l4 but < l2l3 — relational structure partially invariant, alignment improves over naive zero-padding
- **Negative:** delta_alignment ≤ cross_domain_l4 — ΔL2→ΔL3 structure not consistent enough across domains

Partial is a meaningful positive result: it validates the relational delta principle.

## HPM Interpretation

If strong or partial result holds: the L2→L3 abstraction captures domain-agnostic relational
structure — the same operator (in aligned coordinates) governs how feature changes propagate
across physics, algebra, and visual patterns. This validates HPM's core claim that hierarchy of
abstraction is universal, not domain-specific.
