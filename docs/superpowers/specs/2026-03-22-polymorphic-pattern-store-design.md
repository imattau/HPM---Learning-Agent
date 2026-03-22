# Polymorphic Pattern Store — Phase 1: LaplacePattern

**Date:** 2026-03-22

## Goal

Add a second pattern type — `LaplacePattern` — to the HPM framework, establishing the minimal polymorphic protocol needed for multiple pattern types to coexist in the system. This is a proof-of-concept phase; a full mixed-type store (one agent holding multiple pattern types simultaneously) is deferred until this phase validates successfully.

## Scope

- **In scope:** `LaplacePattern` class, `sample()` method on GaussianPattern + LaplacePattern, minimal MetaPatternRule change, `pattern_type` AgentConfig field, pattern factory with dict dispatcher, ARC benchmark integration, unit + integration tests
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

### `sample(n: int, rng) -> np.ndarray`

Returns `n` samples drawn from the Laplace distribution:

```python
return rng.laplace(loc=self.mu, scale=self.b, size=(n, len(self.mu)))
```

This method is added to **both** `LaplacePattern` and `GaussianPattern` (see Section 3).

### `update(x) -> LaplacePattern`

Online update returning a new instance (value-type semantics, same as GaussianPattern):

- Record `mu_old = self.mu` before updating
- `mu` updated via running mean: `mu_new = (mu_old * n_obs + x) / (n_obs + 1)`
- `b` updated via running mean of residual against `mu_old`: `b_new = (b_old * n_obs + |x - mu_old|) / (n_obs + 1)`, floored at `1e-6`
- Using `mu_old` for the residual avoids the downward bias that would occur if the already-shifted `mu_new` were used
- Note: unlike `GaussianPattern.update()` which keeps `sigma` fixed, `LaplacePattern.update()` actively updates `b` — this is intentional since `b` is a single-parameter scale that can be estimated online efficiently
- `id` and `level` preserved on the new instance

### `sigma` attribute

LaplacePattern has **no `sigma` attribute** — not in `__dict__`, not a property. This routes `sym_kl_normalised` to the Monte Carlo branch (which uses `sample()`, not `sigma` — see Section 3).

### `recombine(other) -> LaplacePattern`

Averages `mu` and `b` element-wise. Raises `TypeError` if `other` is not a `LaplacePattern` (enforces homogeneous stores for this phase).

**Invariant:** `PatternField` only broadcasts between agents configured with the same `pattern_type`. Since `make_orchestrator()` assigns one type per agent, and `PatternField` shares within-type by querying a single store per agent, cross-type recombination cannot occur in practice. `TypeError` is a safety net, not the primary guard.

### Structural methods

- `description_length()`: count of non-trivial `mu` entries + dimensions (same as GaussianPattern)
- `connectivity()`: always returns `0.0` — there is no off-diagonal structure in a diagonal-scale parameterisation (unlike Gaussian's full covariance matrix). Satisfies the protocol method signature with defined semantics.
- `compress()`: ratio of max `b` to sum of `b` values (analogous to top-eigenvalue ratio for Gaussian)
- `is_structurally_valid()`: returns `True` iff all `b > 0`

### Serialisation

`to_dict()` emits `'type': 'laplace'` and preserves `mu`, `b`, `id`, `level`, `source_id`, `_n_obs`. `from_dict()` is a classmethod on `LaplacePattern`. A top-level dispatcher `pattern_from_dict(d)` in `factory.py` routes on `d['type']` (see Section 2).

---

## Section 2: AgentConfig + Pattern Factory

### AgentConfig (`hpm/config.py`)

One new field added to the existing dataclass:

```python
pattern_type: str = "gaussian"  # "gaussian" | "laplace"
```

Default is `"gaussian"` — all existing code paths are unaffected.

### Factory (`hpm/patterns/factory.py`)

New module with two functions:

```python
def make_pattern(mu, scale, pattern_type: str = "gaussian", **kwargs):
    """Construct a pattern from (mu, scale) parameters.
    For Laplace, scale is interpreted as b (broadcast to vector if scalar).
    """
    if pattern_type == "gaussian":
        return GaussianPattern(mu, scale, **kwargs)
    elif pattern_type == "laplace":
        b = np.ones(len(mu)) * scale if np.isscalar(scale) else np.asarray(scale)
        return LaplacePattern(mu, b, **kwargs)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type!r}")


def pattern_from_dict(d: dict):
    """Deserialise a pattern from a dict produced by to_dict()."""
    t = d.get('type', 'gaussian')
    if t == 'gaussian':
        return GaussianPattern.from_dict(d)
    elif t == 'laplace':
        return LaplacePattern.from_dict(d)
    else:
        raise ValueError(f"Unknown pattern type in dict: {t!r}")
```

The Agent class reads `self.config.pattern_type` and calls `make_pattern` when initialising its pattern. Any code path that deserialises patterns from dicts uses `pattern_from_dict`.

### Orchestrator (`benchmarks/multi_agent_common.py`)

`make_orchestrator()` gains an optional `pattern_types` argument (list of strings, one per agent). Defaults to `["gaussian"] * n_agents`. Passes each entry to the corresponding agent's config.

---

## Section 3: Integration with Existing Systems

### MetaPatternRule (`hpm/dynamics/meta_pattern_rule.py`)

**Minimal change required.** The existing Monte Carlo fallback branch calls `rng.multivariate_normal(p.mu, p.sigma, n_samples)`, which requires `.sigma` — crashing for `LaplacePattern`. Fix: replace the two `rng.multivariate_normal(...)` calls with `p.sample(n_samples, rng)` and `q.sample(n_samples, rng)`:

```python
# Before (broken for non-Gaussian):
samples_p = rng.multivariate_normal(p.mu, p.sigma, n_samples)
samples_q = rng.multivariate_normal(q.mu, q.sigma, n_samples)

# After (works for any pattern with sample()):
samples_p = p.sample(n_samples, rng)
samples_q = q.sample(n_samples, rng)
```

`GaussianPattern` gains a `sample(n, rng)` method returning `rng.multivariate_normal(self.mu, self.sigma, n)`. `LaplacePattern` gains `sample(n, rng)` returning `rng.laplace(self.mu, self.b, size=(n, len(self.mu)))`. The `if hasattr(p, 'sigma')` guard remains unchanged — it still correctly routes Gaussian pairs through the fast Cholesky KL path.

### TieredStore / PatternStore

No changes. Stores hold pattern objects by agent_id and never inspect their type.

### PatternField

No changes. `pattern.recombine(other)` is called during cross-agent sharing. With homogeneous per-agent stores, both patterns are always the same type (see invariant in Section 1).

### ARC Benchmark (`benchmarks/multi_agent_arc.py`)

- `make_arc_orchestrator()` gains an optional `pattern_types` parameter (default: `["gaussian", "gaussian"]`)
- `evaluate_task()` gains an optional `pattern_types` parameter, threads it through to `make_arc_orchestrator()`
- New convenience function `make_arc_laplace_orchestrator()` calls `make_arc_orchestrator(pattern_types=["laplace", "laplace"])`
- `ensemble_score` calls `p.log_prob(vec)` — works identically for both types

**Validation:** Run the multi-agent ARC benchmark with `make_arc_laplace_orchestrator()` on the training split. Compare accuracy and mean-rank against the Gaussian baseline.

---

## Section 4: Testing Strategy

### `tests/patterns/test_laplace.py` (unit)

- `log_prob` returns lower NLL for `x = mu` than for `x` far from `mu`
- `log_prob` is always finite for valid inputs
- `update` increments `_n_obs`, moves `mu` toward observed `x`
- `b` converges toward `|x - mu_old|` after repeated identical observations
- `sample(n, rng)` returns array of shape `(n, D)` with values drawn from Laplace distribution
- `hasattr(pattern, 'sigma')` returns `False`
- `connectivity()` always returns `0.0`
- `recombine` with another `LaplacePattern` averages `mu` and `b`
- `recombine` with a `GaussianPattern` raises `TypeError`
- `to_dict` / `from_dict` round-trip preserves all fields; `to_dict()['type'] == 'laplace'`
- `is_structurally_valid` returns `False` when any `b ≤ 0`

### `tests/patterns/test_gaussian_sample.py` (unit, new)

- `GaussianPattern.sample(n, rng)` returns array of shape `(n, D)`
- Mean of samples is close to `mu` (statistical check, large n)

### `tests/patterns/test_factory.py` (unit)

- `make_pattern("gaussian", ...)` returns `GaussianPattern`
- `make_pattern("laplace", ...)` returns `LaplacePattern`
- Scalar `scale` broadcast correctly for Laplace
- `pattern_from_dict({'type': 'gaussian', ...})` returns `GaussianPattern`
- `pattern_from_dict({'type': 'laplace', ...})` returns `LaplacePattern`
- Unknown `pattern_type` raises `ValueError`

### `tests/integration/test_laplace_arc.py` (integration)

- Runs `evaluate_task(..., pattern_types=["laplace", "laplace"])` on 10 ARC tasks
- Asserts no exceptions and `mean_rank ≤ 5.0`
- Confirms `log_prob` outputs differ between Gaussian and Laplace on the same vector

All existing test files remain unchanged; all current tests must pass after this phase.

---

## Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| `sample()` method on patterns | Fixes Monte Carlo KL fallback cleanly; makes pattern protocol more complete; two-line change to MetaPatternRule |
| No `sigma` attribute on LaplacePattern | Routes to Monte Carlo branch in `sym_kl_normalised` automatically |
| `mu_old` for `b` residual | Avoids downward bias from computing residual against already-shifted mean |
| `b` updated online (unlike Gaussian's fixed `sigma`) | Single-parameter per-dimension scale is cheap to estimate; Gaussian sigma is kept fixed for stability |
| `connectivity()` returns 0.0 | Diagonal scale has no off-diagonal structure; honest over misleading proxy |
| `pattern_from_dict()` in factory | Single dispatch point for deserialisation; keeps pattern classes independent |
| Homogeneous per-agent (not mixed store) | Defers cross-type recombination and KL problems until Laplace is validated |
| `b` floor at `1e-6` | Prevents degenerate collapse analogous to Gaussian's Cholesky fallback |
