# Polymorphic Pattern Store — Phase 1: LaplacePattern

**Date:** 2026-03-22

## Goal

Add a second pattern type — `LaplacePattern` — to the HPM framework, establishing the minimal polymorphic protocol needed for multiple pattern types to coexist in the system. This is a proof-of-concept phase; a full mixed-type store (one agent holding multiple pattern types simultaneously) is deferred until this phase validates successfully.

## Scope

- **In scope:** `LaplacePattern` class, `pattern_type` AgentConfig field, pattern factory, ARC benchmark integration, unit + integration tests
- **Out of scope:** Mixed-type stores (one agent, multiple types), cross-type KL (closed-form), cross-type recombination, von Mises / Dirac / Categorical / Sparse Kernel types

---

## Section 1: LaplacePattern Class

**File:** `hpm/patterns/laplace.py`

The Laplace distribution is parameterised by location `mu` (ndarray, shape `[D]`) and scale `b` (ndarray, shape `[D]`, all values > 0). There is no full covariance matrix — scale is per-dimension only.

### `log_prob(x) -> float`

Returns NLL (negative log-likelihood), consistent with `GaussianPattern.log_prob` sign convention (lower = more probable):

```
NLL = sum(|x - mu| / b) + sum(log(2 * b))
```

### `update(x) -> LaplacePattern`

Online update returning a new instance (value-type semantics, same as GaussianPattern):
- `mu` updated via running mean (approximation to true median — acceptable for streaming)
- `b` updated via running mean of `|x - mu|`, floored at `1e-6` to prevent collapse
- `id` and `level` preserved on the new instance

### `sigma` attribute

LaplacePattern has **no `sigma` attribute** (not in `__dict__`, not a property). This is intentional: `MetaPatternRule.sym_kl_normalised` checks `hasattr(p, 'sigma')` to select the Gaussian Cholesky KL path. When this check fails, the Monte Carlo fallback fires automatically, using `p.log_prob(s)` — which LaplacePattern implements correctly.

### `recombine(other) -> LaplacePattern`

Averages `mu` and `b` element-wise. Raises `TypeError` if `other` is not a `LaplacePattern` (enforces homogeneous stores for this phase).

### Structural methods

Mirroring GaussianPattern, operating on `b` instead of `sigma`:
- `description_length()`: count of non-trivial `mu` entries + dimensions
- `connectivity()`: mean absolute correlation of scale values (proxy for cross-dimensional coupling)
- `compress()`: ratio of max `b` to sum of `b` values (analogous to top-eigenvalue ratio)
- `is_structurally_valid()`: returns True iff all `b > 0`

### Serialisation

`to_dict()` / `from_dict()` round-trip preserving `mu`, `b`, `id`, `level`, `source_id`, `_n_obs`.

---

## Section 2: AgentConfig + Pattern Factory

### AgentConfig (`hpm/config.py`)

One new field added to the existing dataclass:

```python
pattern_type: str = "gaussian"  # "gaussian" | "laplace"
```

Default is `"gaussian"` — all existing code paths are unaffected.

### Factory (`hpm/patterns/factory.py`)

New module with a single function:

```python
def make_pattern(mu, scale, pattern_type: str = "gaussian", **kwargs):
    if pattern_type == "gaussian":
        return GaussianPattern(mu, scale, **kwargs)
    elif pattern_type == "laplace":
        return LaplacePattern(mu, scale, **kwargs)  # scale interpreted as b
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type!r}")
```

When a scalar or `init_sigma` float is passed as `scale` for Laplace, it is broadcast to `np.ones(D) * scale`.

The Agent class reads `self.config.pattern_type` and calls the factory when initialising its pattern. No other Agent code changes.

### Orchestrator (`benchmarks/multi_agent_common.py`)

`make_orchestrator()` gains an optional `pattern_types` argument (list of strings, one per agent). Defaults to `["gaussian"] * n_agents`. Passes each entry to the corresponding agent's config.

---

## Section 3: Integration with Existing Systems

### MetaPatternRule

No code changes. The `sym_kl_normalised` Monte Carlo fallback already handles non-Gaussian patterns. LaplacePattern's absence of a `sigma` attribute routes it there automatically. Cost: ~200 samples per pair, but the existing `n > 20` guard suppresses pairwise KL for large stores.

### TieredStore / PatternStore

No changes. Stores hold pattern objects by agent_id and never inspect their type. Tier1→Tier2 promotion, negative partition, and `query()` all operate on any object with a `log_prob` method.

### PatternField

No changes. `pattern.recombine(other)` is called during cross-agent sharing. With homogeneous per-agent stores, both patterns are always the same type.

### ARC Benchmark

`make_arc_orchestrator()` in `benchmarks/multi_agent_arc.py` gains an optional `pattern_types` parameter. A new convenience function `make_arc_laplace_orchestrator()` sets `pattern_types=["laplace", "laplace"]`. `ensemble_score` calls `p.log_prob(vec)` — works identically for both types.

**Validation:** Run the multi-agent ARC benchmark with the Laplace orchestrator on the training split. Compare accuracy and mean-rank against the Gaussian baseline to confirm the pattern type has distinct and functional behaviour.

---

## Section 4: Testing Strategy

### `tests/patterns/test_laplace.py` (unit)

- `log_prob` returns lower NLL for `x = mu` than for `x` far from `mu`
- `log_prob` is always finite for valid inputs
- `update` increments `_n_obs`, moves `mu` toward observed `x`
- `b` converges toward `|x - mu|` after repeated identical observations
- `hasattr(pattern, 'sigma')` returns `False` (critical for MetaPatternRule routing)
- `recombine` with another `LaplacePattern` averages `mu` and `b`
- `recombine` with a `GaussianPattern` raises `TypeError`
- `to_dict` / `from_dict` round-trip preserves all fields
- `is_structurally_valid` returns `False` when any `b ≤ 0`

### `tests/patterns/test_factory.py` (unit)

- `make_pattern("gaussian", ...)` returns `GaussianPattern`
- `make_pattern("laplace", ...)` returns `LaplacePattern`
- Unknown `pattern_type` raises `ValueError`
- Scalar `scale` is broadcast correctly for both types

### `tests/integration/test_laplace_arc.py` (integration)

- Runs `make_arc_orchestrator(pattern_types=["laplace", "laplace"])` on 10 ARC tasks
- Asserts no exceptions and `mean_rank ≤ 5.0`
- Confirms `log_prob` outputs differ between Gaussian and Laplace on the same vector (validates distinct behaviour)

All existing test files remain unchanged; all current tests must pass after this phase.

---

## Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| No `sigma` attribute on LaplacePattern | Routes KL computation to existing Monte Carlo fallback without touching MetaPatternRule |
| Running mean for `mu` (not true median) | Median estimation online is non-trivial; running mean is a practical approximation sufficient for pattern matching |
| Homogeneous per-agent (not mixed store) | Defers cross-type recombination and KL problems until the Laplace type is validated |
| `b` floor at `1e-6` | Prevents degenerate collapse analogous to Gaussian's Cholesky fallback for near-singular sigma |
| Factory in separate file | Keeps `gaussian.py` and `laplace.py` independent; factory is the only place that knows about all types |
