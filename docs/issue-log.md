# HPM Agent Issue Log

## Issue 1: `last_capacity` in AffectiveEvaluator
**Status:** CLOSED — already implemented
`hpm/evaluators/affective.py:52` — `last_capacity(pattern_id)` returns historical capacity, defaulting to 0.0 for unknown patterns.

---

## Issue 2: Metric Reporting Lag After Recombination
**Status:** OPEN
**File:** `hpm/agents/agent.py`
`level_mean` and `level_distribution` are computed from `surviving_patterns` before the recombination block runs. If a recombination is accepted, the new pattern is in the store but absent from these metrics. Return dict should reflect the post-insight state.
**Fix:** Query the store after recombination to recompute level metrics when a pattern was accepted.

---

## Issue 3: Numerical Stability in `sym_kl_normalised`
**Status:** OPEN
**File:** `hpm/dynamics/meta_pattern_rule.py`
Monte Carlo KL approximation can produce `inf`/`nan` when a pattern has very low variance (tight sigma) — samples from one pattern fall in near-zero-probability regions of the other. For `GaussianPattern`, a closed-form KL is available and exact.
**Fix:** Implement closed-form KL for Gaussians; keep Monte Carlo as fallback for non-Gaussian patterns.
Closed-form: `KL(p||q) = 0.5 * (tr(Σ_q^{-1} Σ_p) + (μ_q-μ_p)^T Σ_q^{-1} (μ_q-μ_p) - d + ln(det(Σ_q)/det(Σ_p)))`

---

## Issue 4: Recombination Temperature Sensitivity ("Creative Stress")
**Status:** OPEN
**File:** `hpm/dynamics/recombination.py` + `hpm/config.py`
With `recomb_temp=1.0` and highly concentrated weights, the operator is conservative — likely to draw the same dominant pair repeatedly. When `total_conflict` is high (crisis mode), the agent should explore more diverse parent pairs.
**Fix:** Pass `total_conflict` into `attempt()` and compute effective temperature as `recomb_temp * (1 + conflict_stress_scale * total_conflict)`. Add `conflict_stress_scale` config field (default `0.0` for backward compatibility).
