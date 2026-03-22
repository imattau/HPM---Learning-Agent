# CategoricalPattern — Polymorphic Pattern Store Phase 2

**Date:** 2026-03-22

## Goal

Add `CategoricalPattern` as a third pattern type in the HPM framework. Where `GaussianPattern` models a D-dimensional continuous space and `LaplacePattern` models the same space with heavier tails, `CategoricalPattern` models a D-dimensional **discrete** space: each dimension is an independent categorical distribution over a finite alphabet of K symbols.

This enables agents to represent discrete structural laws — colour mappings, symbol co-occurrences, positional invariants — without any knowledge of what those symbols mean. The alphabet is defined by a hyperparameter K; the agent's encoding layer handles substrate-specific discretisation.

**K=1 is not supported.** Raise `ValueError` at construction if `K < 2`.

---

## Section 1: CategoricalPattern Class

**File:** `hpm/patterns/categorical.py`

### Parameters

- `probs` — `np.ndarray` shape `(D, K)`. Each row is a probability distribution over K symbols. All values ≥ 1e-8 (floored at construction and after each update). Rows sum to 1.
- `K` — alphabet size (int, ≥ 2), stored as an instance attribute. Hyperparameter set at construction; not updated.
- No `sigma` attribute — absence routes `MetaPatternRule.sym_kl_normalised` to the Monte Carlo KL branch (same mechanism as `LaplacePattern`).

### `log_prob(x) -> float`

`x` is a D-length integer array with values in `{0…K-1}`. Out-of-range values are the caller's responsibility — the pattern does not validate them (consistent with how `GaussianPattern` does not validate float inputs). NumPy will raise `IndexError` on out-of-range values, which is the expected failure mode.

```
NLL = -sum(log(probs[d, x[d]]) for d in range(D))
```

Lower NLL = more probable. Consistent with `GaussianPattern` sign convention so `ensemble_score` works unchanged.

### `update(x) -> CategoricalPattern`

Online Bayesian count update. `_n_obs` is initialised to **K** (not 0), treating the initial uniform `probs` as K pseudo-observations — one per symbol per position. This prevents the first real observation from collapsing the posterior to a point mass.

```
probs_new[d, k] = (probs_old[d, k] * n_obs + one_hot(x[d], K)[k]) / (n_obs + 1)
```

Uses pre-update `probs_old` (consistent with `LaplacePattern`'s `mu_old` convention). Floor applied at 1e-8 after update. Returns a new instance (value-type semantics). `id` and `level` preserved. `_n_obs` incremented by 1.

### `recombine(other) -> CategoricalPattern`

`_n_obs`-weighted average of `probs` matrices:

```
total = n_obs_self + n_obs_other
if total == 0:
    probs_new = (probs_self + probs_other) / 2   # fallback: uniform average
else:
    probs_new = (probs_self * n_obs_self + probs_other * n_obs_other) / total
```

Renormalise rows after averaging (to correct any floating-point drift). Raises `TypeError` if `other` is not a `CategoricalPattern`. Both patterns must have the same D and K; if not, raise `ValueError`.

**Rationale for weighted average:** A pattern that has seen 100 observations carries more structural authority than one that has seen 5. Pooling raw counts is the statistically correct way to merge two Dirichlet posteriors. With `_n_obs` initialised to K, freshly constructed patterns both start at K, so the fallback (total == 0) is unreachable in practice; it is a safety net for direct construction with `_n_obs=0`.

### `sample(n, rng) -> np.ndarray`

Returns shape `(n, D)` **integer** array. Each column `d` drawn independently:

```python
samples[:, d] = rng.choice(K, size=n, p=probs[d])
```

**Note:** `sample()` is not part of the `Pattern` protocol in `base.py`, but it is required by `MetaPatternRule`'s Monte Carlo KL branch, which calls `p.sample(n, rng)` when `hasattr(p, 'sigma')` is False. If `sample()` is omitted, the MC KL branch will fail at runtime with `AttributeError`. It must be implemented.

### Structural Methods

- `description_length() -> float`: `float(count)` where `count` is the number of positions whose entropy `H(probs[d]) = -sum(p * log(p))` is below `log(K) * 0.5` — positions that have learned a definite preference (less than half maximum entropy). Returns `float` to match the protocol's declared return type.
- `connectivity() -> float`: Returns `0.0` — independence assumption across positions. Satisfies the protocol method signature with defined semantics; can be upgraded to average mutual information in a future phase.
- `compress() -> float`: `max_row_entropy / mean_row_entropy`. If `mean_row_entropy == 0` (all rows are point masses, e.g. K→1 limit), return `1.0`. Measures how concentrated the most peaked position is relative to average — high ratio = one very strong anchor found amid noise.
- `is_structurally_valid() -> bool`: Returns `True` iff all `probs >= 1e-8` (floor integrity — not `>= 0`, which is vacuous given the floor) and each row sums to 1 within 1e-6 tolerance.

### Serialisation

`to_dict()` emits:
```python
{
    'type': 'categorical',
    'probs': probs.tolist(),  # nested list
    'K': K,
    'id': self.id,
    'level': self.level,
    'source_id': self.source_id,
    'n_obs': self._n_obs,   # key is 'n_obs' (no underscore) — matches GaussianPattern/LaplacePattern convention
}
```

`from_dict()` is a classmethod on `CategoricalPattern` that reads `d['n_obs']` (not `d['_n_obs']`).
`pattern_from_dict()` in `factory.py` routes on `d['type'] == "categorical"`.

---

## Section 2: Factory + AgentConfig + Observation Protocol

### AgentConfig (`hpm/config.py`)

Two changes:

```python
alphabet_size: int = 10  # K for CategoricalPattern; ignored by Gaussian/Laplace
pattern_type: str = "gaussian"  # "gaussian" | "laplace" | "categorical"
```

Default `alphabet_size=10` covers the ARC colour palette (0–9). Meaningless for continuous pattern types — fully backward compatible. Update the `pattern_type` comment to include `"categorical"`.

### Factory (`hpm/patterns/factory.py`)

Add `alphabet_size: int = 10` as a named parameter to `make_pattern()`:

```python
def make_pattern(mu, scale, pattern_type: str = "gaussian", alphabet_size: int = 10, **kwargs):
```

New categorical branch:

```python
elif pattern_type == "categorical":
    K = kwargs.pop("K", alphabet_size)
    D = len(mu)
    probs = np.ones((D, K)) / K   # uniform = maximum entropy prior
    return CategoricalPattern(probs, K=K, **kwargs)
```

`mu` is used only for dimensionality (`D = len(mu)`); it is not stored in `CategoricalPattern`. `scale` is ignored for categorical patterns (consistent with how it is ignored for LaplacePattern's `b` when `scale` is a scalar).

Callers pass `alphabet_size=config.alphabet_size` when creating categorical patterns:

```python
make_pattern(mu=mu, scale=1.0, pattern_type="categorical", alphabet_size=config.alphabet_size)
```

Existing Gaussian/Laplace callers are unaffected — `alphabet_size` has a default and is ignored by those branches.

Extend `pattern_from_dict()`:

```python
elif t == "categorical":
    return CategoricalPattern.from_dict(d)
```

### Observation Protocol

`CategoricalPattern.log_prob(x)` expects integer vectors; `GaussianPattern` and `LaplacePattern` expect float vectors. These are incompatible in the same observation pipeline.

**Design decision: Separate encoding per agent (Substrate Honest).**

The benchmark routes integer-encoded observations to categorical agents and float-encoded observations to continuous agents. `orch.step({"arc_a": float_vec, "arc_b": int_vec})` already works — the orchestrator passes each agent its own keyed observation by `agent_id`.

This places discretisation responsibility on the Executive layer (benchmark / orchestrator loop), which in a full HPM deployment is the job of the `SubstrateBridge`. The pattern type remains substrate-agnostic; the bridge decides how to slice a signal into discrete symbols.

**In the ARC benchmark**, a categorical agent receives the raw output grid values flattened to a D-length integer vector (values 0–9, K=10). The delta encoding used for Gaussian/Laplace agents is not applicable here — colour identity, not colour change, is the discrete signal.

---

## Section 3: Specialist Integration

No changes required to any specialist class. The Pattern protocol is sufficient:

| Class | Interaction | Notes |
|---|---|---|
| `StructuralLawMonitor` | `connectivity()`, `compress()` | Both defined; entropy-based `compress()` gives richer signal than Gaussian's eigenvalue ratio |
| `MetaPatternRule` | `hasattr(p, 'sigma')` → MC KL → `p.sample(n, rng)` | `sample()` returns integer array; MC KL estimates divergence correctly |
| `RecombinationStrategist` | `p.recombine(other)` | `TypeError` on cross-type is the safety net |
| `PredictiveSynthesisAgent` | `p.log_prob(vec)` for fragility probing | Probe vector must be integer-typed for categorical agents; benchmark constructs probe accordingly |
| `DecisionalActor` | `ensemble_score` → `p.log_prob(vec)` | Works identically; lower NLL = stronger evidence |
| `TieredStore` / `ContextualPatternStore` | stores by id, never inspects type | Fully transparent |

**Ensemble competition:** Because `log_prob` sign convention is shared, `ensemble_score` naturally arbitrates between Gaussian, Laplace, and Categorical agents. Geometric tasks favour the Gaussian; colour-swap tasks favour the Categorical (its "peaky" `probs` produce low NLL for the correct symbol sequence). The weight update in `MetaPatternRule` amplifies whichever representation fits better.

---

## Section 4: Testing Strategy

### `tests/patterns/test_categorical.py` (unit)

- `log_prob` at a maximally probable vector is lower than at an improbable vector
- `log_prob` is always finite after construction (floor prevents `log(0)`)
- `update` increments `_n_obs`, shifts `probs` toward observed symbol
- After 100 identical updates, observed symbol probability > 0.95 at each position
- Initial `_n_obs == K` (pseudo-count for prior)
- `sample(n, rng)` returns shape `(n, D)` integer array with values in `{0…K-1}`
- `hasattr(pattern, 'sigma')` is `False`
- `recombine` with another `CategoricalPattern` uses `_n_obs`-weighted average
- `recombine` falls back to uniform average when both `_n_obs` contribute equally (same n_obs)
- `recombine` with `GaussianPattern` raises `TypeError`
- `is_structurally_valid()` — False if any entry < 1e-8 or any row sums outside [1-1e-6, 1+1e-6]
- `to_dict` / `from_dict` round-trip preserves `probs`, `K`, `id`, `level`, `n_obs` (no underscore in dict key)
- `compress()` returns `1.0` when all rows are point masses (zero denominator guard)
- `description_length()` returns a `float`
- `CategoricalPattern(probs, K=1)` raises `ValueError`

### `tests/integration/test_categorical_arc.py`

- A 3-agent ensemble (Gaussian + Laplace + Categorical) constructs without error
- Categorical agent receives integer-encoded observations; others receive delta-float
- `orch.step()` runs 10 steps without error
- `ensemble_score` returns finite values for both float-encoded and integer-encoded candidate vectors scored against their respective agents

---

## File Map

| File | Action |
|---|---|
| `hpm/patterns/categorical.py` | Create |
| `hpm/patterns/factory.py` | Extend `make_pattern()` signature + body; extend `pattern_from_dict()` |
| `hpm/config.py` | Add `alphabet_size: int = 10`; update `pattern_type` comment |
| `tests/patterns/test_categorical.py` | Create |
| `tests/integration/test_categorical_arc.py` | Create |
