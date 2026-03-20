# Pattern Density + Stickiness Design Spec

**Date:** 2026-03-21

## Goal

Implement Pattern Density `D(h_i)` as defined in HPM Appendix A.8 and add it as a density bias term to the MetaPatternRule weight update. Patterns that are internally coherent, evaluator-confirmed, and socially amplified resist decay — the "stickiness" effect — even when their epistemic accuracy is temporarily below average.

## Background

The current MetaPatternRule (D5) updates weights via replicator dynamics and conflict inhibition, but has no term for pattern density. HPM Appendix A.8 specifies a density bias `κ_D * D(h_i) * w_i` that allows high-density patterns to remain sticky. This is the mechanism behind phenomena like superstition (a socially amplified, internally coherent pattern that persists despite moderate loss) and expertise (a structurally rich pattern that stabilises under training).

## Formula

### Pattern Density

```
D(h_i) = alpha_conn * structural_i
        + alpha_sat * saturation_i
        + alpha_amp * field_freq_i
```

Each component is in [0, 1]:

**Structural connectivity:**
```
structural_i = (pattern.connectivity() + pattern.compress()) / 2
```
`connectivity()` returns mean absolute off-diagonal correlation of sigma (feature interdependence).
`compress()` returns ratio of largest eigenvalue to total variance (dominant structure).
Both methods exist on `GaussianPattern` (`hpm/patterns/gaussian.py`, lines 29 and 40) and return values in [0, 1].

**Evaluator saturation:**
```
epi_acc_norm_i = loss_i / (1 + loss_i)              # maps [0, ∞) → [0, 1)
saturation_i   = (1 - epi_acc_norm_i) * capacity_i
```
`loss_i` is the running loss L_i(t) — the EMA of the negative log-likelihood (a non-negative scalar; lower = better fit).
`EpistemicEvaluator.update()` returns `A_i = -L_i` (negative accuracy). In `Agent.step()`, the caller computes `loss_i = -epi_acc_i` before passing to `PatternDensity.compute()`. `loss_i ∈ [0, ∞)` is guaranteed because L_i(0) = 0 and each EMA update is a convex combination of the current value and a non-negative NLL.
`capacity_i = 1 - novelty_i` from AffectiveEvaluator — measures stability (how little the pattern is changing).
Low loss × high stability = high saturation. A pattern that fits well AND isn't changing is the most confirmed.

**Field amplification:**
```
field_freq_i = field.freq(pattern_id) if field else 0.0
```
Social field frequency from PatternField (already computed in Agent.step).
`PatternField.freqs_for()` returns `pattern_weight / total_field_mass` — guaranteed in [0, 1] because individual pattern weights are non-negative and their sum cannot exceed total mass.

### Weight Update (D5 + Density Bias)

```
w_i(t+1) = w_i(t)
          + eta * (Total_i - Total_bar) * w_i        # replicator (D5, existing)
          + kappa_D * D(h_i) * w_i                   # density bias (new)
          - beta_c * sum_j kappa_ij * w_i * w_j      # conflict inhibition (D5, existing)
```

Followed by the existing renormalisation step (`w_i /= sum(w_j)`) which already runs after weight updates — no change needed, it naturally handles the density bias term.

`kappa_D = 0.0` by default — fully backward compatible. All 125 existing tests pass unchanged.

## Components

### New: `hpm/dynamics/density.py`

Class `PatternDensity(alpha_conn, alpha_sat, alpha_amp)`:

- `compute(pattern, loss, capacity, field_freq) -> float` — returns `D(h_i)` in [0, 1]; `loss` is the running loss L_i (non-negative)
- Stateless: called once per pattern per step
- Clamps output to [0, 1] in case floating-point drift produces values slightly outside range

### Modified: `hpm/evaluators/affective.py`

Add `last_capacity(pattern_id: str) -> float` method:
- Returns the most recently computed `capacity = 1 - novelty` for the given pattern
- Stored in a new `_last_capacity: dict[str, float]` internal dict, populated during `update()`
- Returns 0.0 for pattern IDs that have never been passed to `update()` (the dict default). In practice, `update()` is always called before `last_capacity()` in `Agent.step()`, so the actual returned value on step 1 is `0.5` (since `delta_A = 0` on first update → `novelty = sigmoid(0) = 0.5` → `capacity = 0.5`).
- Backward compatible: `update()` signature unchanged

### Modified: `hpm/dynamics/meta_pattern_rule.py`

`MetaPatternRule.__init__` gains a new optional parameter with a backward-compatible default:
```python
def __init__(self, eta: float = 0.01, beta_c: float = 0.1, epsilon: float = 1e-4, kappa_D: float = 0.0):
```
Existing instantiations `MetaPatternRule(eta=..., beta_c=..., epsilon=...)` are unaffected.

`MetaPatternRule.step()` gains a new optional parameter:
```python
def step(self, patterns, weights, totals, densities=None) -> np.ndarray:
```
When `densities` is provided (non-None), adds `kappa_D * densities[i] * weights[i]` to each weight update before the existing `np.maximum(..., 0.0)` clip and renormalisation. The floor case (all weights below epsilon) is unchanged and takes priority: if triggered, it returns early and `densities` has no effect. Existing callers with no `densities` argument behave identically.

### Modified: `hpm/config.py`

Four new fields with backward-compatible defaults:
```python
kappa_D: float = 0.0      # density bias weight (0 = off)
alpha_conn: float = 0.33   # structural connectivity weight in D(h)
alpha_sat: float = 0.33    # evaluator saturation weight in D(h)
alpha_amp: float = 0.34    # field amplification weight in D(h)
```

### Modified: `hpm/agents/agent.py`

After evaluators are computed (epistemic_accs and e_affs available) and before MetaPatternRule:
1. Instantiate `PatternDensity` in `__init__` from config
2. In `step()`: collect `capacity_i = self.affective.last_capacity(p.id)` and `field_freq_i` per pattern
3. Compute `densities = [self.pattern_density.compute(p, -epi, cap, ff) for ...]` — note `-epi` converts accuracy A_i to loss L_i
4. Pass `densities` to `self.dynamics.step()`
5. Add `density_mean` to return dict

### Modified: `hpm/dynamics/__init__.py` (if exists)

Re-export `PatternDensity` if there is an `__init__.py`.

## Computation Sequence in Agent.step()

```
1. epistemic_accs, e_affs computed (existing)
2. field_freqs, ext_freqs, freq_totals computed (existing)
3. e_socs computed (existing)
4. e_costs computed (existing)
5. NEW: densities computed using affective.last_capacity() + structural + field_freq
6. totals computed (existing)
7. MetaPatternRule.step(patterns, weights, totals, densities=densities)  ← densities added
8. Prune + persist (existing)
9. Field registration (existing)
```

Note: `last_capacity()` uses the capacity computed during step 1 (the `update()` call in the epistemic/affective loop), so the density reflects the current step's evaluator state. This is correct — density is computed from current observations, not lagged.

## New Config Parameters

| Field | Default | Meaning |
|---|---|---|
| `kappa_D` | 0.0 | Density bias weight. 0 = off (backward compatible) |
| `alpha_conn` | 0.33 | Weight of structural connectivity in D(h) |
| `alpha_sat` | 0.33 | Weight of evaluator saturation in D(h) |
| `alpha_amp` | 0.34 | Weight of field amplification in D(h) |

## Backward Compatibility

- `kappa_D = 0.0` default: density bias contributes 0 to weight updates
- `densities=None` default in MetaPatternRule.step(): behaves identically to current code
- `last_capacity()` returns 0.0 only if called for a pattern ID that has never been through `update()` — which cannot happen in `Agent.step()` since the affective loop runs before density computation. On the actual first step, `last_capacity()` returns 0.5 (sigmoid(0) = 0.5, capacity = 0.5).
- All 125 existing tests pass unchanged

## Testing

### Unit tests: `tests/dynamics/test_density.py`

- `test_density_zero_when_all_components_zero` — zero connectivity, zero saturation, zero field_freq → D = 0
- `test_density_one_when_all_components_maxed` — max connectivity, zero surprise, 100% field_freq → D ≈ 1
- `test_structural_component_uses_connectivity_and_compress` — verify formula
- `test_saturation_high_for_low_surprise_high_stability` — low epi_acc, high capacity → high saturation
- `test_saturation_low_for_high_surprise` — high epi_acc → low saturation regardless of capacity
- `test_field_freq_zero_when_no_field` — field_freq=0.0 contribution is zero

### Unit tests: `tests/evaluators/test_affective_capacity.py` (add to existing test file)

- `test_last_capacity_returns_zero_on_first_step` — unknown pattern_id → 0.0
- `test_last_capacity_reflects_current_step` — after one update, returns capacity for that step
- `test_capacity_is_one_minus_novelty` — verify relationship

### Integration tests: `tests/dynamics/test_meta_pattern_rule_density.py`

- `test_density_bias_increases_high_density_pattern_weight` — high-density pattern gains weight faster than low-density under same total
- `test_kappa_d_zero_unchanged_from_baseline` — MetaPatternRule with kappa_D=0 produces identical output to existing behaviour
- `test_agent_step_includes_density_mean` — agent with kappa_D > 0 returns `density_mean` in step dict
- `test_renormalisation_holds_after_density_bias` — sum of weights = 1.0 after step with density bias

## What This Is Not

- Not a pattern classification system (Level 1-5 is a separate sub-project)
- Not a recombination operator (separate sub-project)
- Not a change to the Pattern protocol — `GaussianPattern.connectivity()` and `compress()` already exist
