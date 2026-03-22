# CategoricalPattern — Polymorphic Pattern Store Phase 2

**Date:** 2026-03-22

## Goal

Add `CategoricalPattern` as a third pattern type in the HPM framework. Where `GaussianPattern` models a D-dimensional continuous space and `LaplacePattern` models the same space with heavier tails, `CategoricalPattern` models a D-dimensional **discrete** space: each dimension is an independent categorical distribution over a finite alphabet of K symbols.

This enables agents to represent discrete structural laws — colour mappings, symbol co-occurrences, positional invariants — without any knowledge of what those symbols mean. The alphabet is defined by a hyperparameter K; the agent's encoding layer handles substrate-specific discretisation.

---

## Section 1: CategoricalPattern Class

**File:** `hpm/patterns/categorical.py`

### Parameters

- `probs` — `np.ndarray` shape `(D, K)`. Each row is a probability distribution over K symbols. All values ≥ 1e-8 (floored at construction and after each update). Rows sum to 1.
- `K` — alphabet size, stored as an instance attribute. Hyperparameter set at construction; not updated.
- No `sigma` attribute — absence routes `MetaPatternRule.sym_kl_normalised` to the Monte Carlo KL branch (same mechanism as `LaplacePattern`).

### `log_prob(x) -> float`

`x` is a D-length integer array with values in {0…K-1}:

```
NLL = -sum(log(probs[d, x[d]]) for d in range(D))
```

Lower NLL = more probable. Consistent with `GaussianPattern` sign convention so `ensemble_score` works unchanged across all pattern types.

### `update(x) -> CategoricalPattern`

Online Bayesian count update using pre-update probabilities (consistent with `LaplacePattern`'s `mu_old` convention):

```
probs_new[d, k] = (probs_old[d, k] * n_obs + one_hot(x[d], K)[k]) / (n_obs + 1)
```

Floor applied at 1e-8 after update. Returns a new instance (value-type semantics). `id` and `level` preserved.

### `recombine(other) -> CategoricalPattern`

`_n_obs`-weighted average of `probs` matrices:

```
probs_new = (probs_self * n_obs_self + probs_other * n_obs_other) / (n_obs_self + n_obs_other)
```

Raises `TypeError` if `other` is not a `CategoricalPattern`. This is a safety net; the primary guard is that `make_orchestrator()` assigns homogeneous types per agent so cross-type recombination cannot occur in normal operation.

**Rationale for weighted average:** A pattern that has seen 100 observations carries more structural authority than one that has seen 5. Pooling raw counts is the statistically correct way to merge two Dirichlet posteriors.

### `sample(n, rng) -> np.ndarray`

Returns shape `(n, D)` integer array. Each column `d` drawn independently:

```python
samples[:, d] = rng.choice(K, size=n, p=probs[d])
```

Required by `MetaPatternRule`'s Monte Carlo KL branch.

### Structural Methods

- `description_length()`: count of positions whose entropy `H(probs[d])` is below `log(K) * 0.5` — positions that have learned a definite preference (less than half maximum entropy). Analogous to GaussianPattern's count of non-trivial mu entries.
- `connectivity()`: `0.0` — independence assumption across positions. Satisfies the protocol; can be upgraded to average mutual information in a future phase.
- `compress()`: `max_row_entropy / mean_row_entropy` — measures how concentrated the most peaked position is relative to average. High ratio = one very strong anchor found amid noise.
- `is_structurally_valid()`: `True` iff all `probs ≥ 0` and each row sums to 1 (within 1e-6 tolerance).

### Serialisation

`to_dict()` emits `type: "categorical"`, `probs` as nested list, `K`, `id`, `level`, `source_id`, `_n_obs`.
`from_dict()` is a classmethod on `CategoricalPattern`.
`pattern_from_dict()` in `factory.py` routes on `d['type'] == "categorical"`.

---

## Section 2: Factory + AgentConfig + Observation Protocol

### AgentConfig (`hpm/config.py`)

One new field:

```python
alphabet_size: int = 10  # K for CategoricalPattern; ignored by Gaussian/Laplace
```

Default 10 covers the ARC colour palette (0–9). Meaningless for continuous pattern types — fully backward compatible.

### Factory (`hpm/patterns/factory.py`)

Extend `make_pattern()`:

```python
elif pattern_type == "categorical":
    K = kwargs.get("K", alphabet_size)  # alphabet_size from AgentConfig
    D = len(mu)  # mu used only for dimensionality; not stored in the pattern
    probs = np.ones((D, K)) / K          # uniform = maximum entropy prior
    return CategoricalPattern(probs, K=K, level=level)
```

Uniform initialisation is the maximum-entropy prior — the agent starts with no preference. Each `update()` call is a symmetry-breaking event, shifting probability mass toward observed symbols.

Extend `pattern_from_dict()`:

```python
elif t == "categorical":
    return CategoricalPattern.from_dict(d)
```

### Observation Protocol

`CategoricalPattern.log_prob(x)` expects integer vectors; `GaussianPattern` and `LaplacePattern` expect float vectors. These are incompatible in the same observation pipeline.

**Design decision: Separate encoding per agent (Option A — Substrate Honest).**

The benchmark routes integer-encoded observations to categorical agents and float-encoded observations to continuous agents. `orch.step({"arc_a": float_vec, "arc_b": int_vec})` already works — the orchestrator passes each agent its own keyed observation by `agent_id`. The benchmark adds an integer encoding branch for categorical agents; the specialist classes and store layers are unaffected.

This places discretisation responsibility on the Executive layer (benchmark / orchestrator loop), which in a full HPM deployment is the job of the `SubstrateBridge`. The pattern type remains substrate-agnostic; the bridge decides how to slice a signal into discrete symbols.

**In the ARC benchmark**, a categorical agent receives `encode_grid_categorical(output) - encode_grid_categorical(input)` represented as a D-length integer vector — the per-pixel colour change, discretised to {-9…+9} and re-mapped to {0…18} (K=19), or simply the raw output grid flattened to integer values (K=10). The exact encoding is a benchmark-level choice.

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
- `log_prob` is always finite (floor prevents `log(0)`)
- `update` increments `_n_obs`, shifts `probs` toward observed symbol
- After 100 identical updates, observed symbol probability > 0.95 at each position
- `sample(n, rng)` returns shape `(n, D)` integer array with values in `{0…K-1}`
- `hasattr(pattern, 'sigma')` is `False`
- `recombine` with another `CategoricalPattern` uses `_n_obs`-weighted average
- `recombine` with `GaussianPattern` raises `TypeError`
- `is_structurally_valid()` — False if any row sums outside [1-1e-6, 1+1e-6]
- `to_dict` / `from_dict` round-trip preserves `probs`, `K`, `id`, `level`, `_n_obs`
- `compress()` returns value in [0, 1] (or ≥ 1 for max/mean ratio — clamp not required)
- `description_length()` returns positive integer

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
| `hpm/patterns/factory.py` | Extend `make_pattern()` + `pattern_from_dict()` |
| `hpm/config.py` | Add `alphabet_size: int = 10` |
| `tests/patterns/test_categorical.py` | Create |
| `tests/integration/test_categorical_arc.py` | Create |
