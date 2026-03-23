# SP9: Cross-Domain L4 Generalisation — Design Spec

## Goal

Validate that a globally-trained L4GenerativeHead (trained across two domains) can transfer to a
third held-out domain using a shared zero-padded feature space. Tests whether HPM's L2→L3
abstraction hierarchy is genuinely domain-agnostic.

## Background

SP8 showed cross-task L4 (same domain, 80/20 task split) matches but does not exceed L2L3
(58.3% vs 58.3%). The global head learned the right structure but added no new signal.
Hypothesis: cross-domain training provides richer L2→L3 supervision, enabling genuine transfer.

## Approach: Leave-One-Domain-Out (3 rotations)

Three domains available, each with native L2/L3 encoders:

| Domain | L2 dim | L3 dim | Task source |
|--------|--------|--------|-------------|
| Math   | 10     | 12     | structured_math.generate_tasks() |
| PhyRE  | 14     | 12     | phyre_sim.generate_family_tasks() |
| ARC    | 9      | 14     | structured_arc.generate_tasks() |

**Shared space:** zero-pad all L2 to 14-dim, all L3 to 14-dim.
One `L4GenerativeHead(feature_dim_in=14, feature_dim_out=14)` per rotation.

### Three Rotations

| Train domains    | Test domain |
|-----------------|-------------|
| Math + PhyRE    | ARC         |
| Math + ARC      | PhyRE       |
| PhyRE + ARC     | Math        |

### Train Phase (per rotation)
1. Load tasks for each training domain
2. Encode all train pairs as (L2_vec, L3_vec), zero-pad to (14, 14)
3. Accumulate into global `L4GenerativeHead(14, 14)`, call `.fit()`

### Test Phase (per rotation)
1. Load/generate held-out domain tasks
2. For each task: encode 5 candidates → zero-pad L2 → predict L3 → score = -‖pred - actual‖
3. Also run l2l3 baseline (domain-native, no padding) for comparison

## Components

### benchmarks/phyre_cross_domain_l4.py

- `encode_domain_pairs(tasks, domain, l2_enc, l3_enc) -> list[tuple[np.ndarray, np.ndarray]]`
  Encodes all train pairs for a domain, zero-pads to (14, 14).

- `fit_cross_domain_l4(train_domain_pairs) -> L4GenerativeHead`
  Accumulates all (L2_padded, L3_padded) pairs, fits global head.

- `score_cross_domain(task, domain, global_l4, l2_enc, l3_enc) -> int`
  Encodes candidates, pads, predicts, returns idx of best candidate.

- `run_rotation(train_domains, test_domain) -> dict[str, float]`
  Runs one leave-one-out rotation, returns accuracy for l2l3 and cross_domain_l4.

- `main()`
  Runs all 3 rotations, prints results table.

### tests/benchmarks/test_phyre_cross_domain_l4.py

- Smoke test: fit on 5 math + 5 phyre tasks, eval on 3 arc tasks, assert accuracy in [0, 1]
- Assert zero-padding produces correct dims (14, 14) for all domains

### Unchanged

- `L4GenerativeHead` (hpm/agents/l4_generative.py)
- `MathL2Encoder`, `MathL3Encoder` (hpm/encoders/math_encoders.py)
- `PhyreL2Encoder`, `PhyreL3Encoder` (hpm/encoders/phyre_encoders.py)
- `ArcL2Encoder`, `ArcL3Encoder` (hpm/encoders/arc_encoders.py)
- Task generators: structured_math, phyre_sim, structured_arc

## Expected Output

```
SP9 Cross-Domain L4 Benchmark
Train domains          Test domain    l2l3    cross_domain_l4
-------------------------------------------------------------
Math + PhyRE           ARC            ~0.63   ?
Math + ARC             PhyRE          ~0.63   ?
PhyRE + ARC            Math           ~0.97   ?
```

## Success Criteria

- **Strong:** cross_domain_l4 > l2l3 on ≥2 of 3 rotations → cross-domain L4 adds genuine signal
- **Partial:** cross_domain_l4 ≈ l2l3 → global L4 generalises but does not improve
- **Negative:** cross_domain_l4 < l2l3 → zero-padding introduces noise; domains need alignment first

## HPM Interpretation

Strong result validates that L2→L3 abstraction is domain-agnostic — the same generative
structure underlies physics, algebra, and visual pattern completion. This is a core HPM claim:
hierarchy of abstraction is a universal structure, not domain-specific.
