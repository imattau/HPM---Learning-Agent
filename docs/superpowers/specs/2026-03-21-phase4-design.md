# Phase 4 Design Specification: Hierarchical Score + Sequence Domain

**Date:** 2026-03-21
**Status:** Draft v3 (post-review-2 fixes)

---

## Overview

Phase 4 completes the HPM agent framework with two remaining items from the original design (§8):

1. **Hierarchical total score (D7)** — adds a compression bonus term `beta_comp * Comp_i(t)` to the per-pattern total score, rewarding patterns that compactly encode their learned structure.
2. **Symbol sequence domain** — a Markov chain-based domain plugin producing one-hot observations, with clean deep/surface separation for §9.1 validation.

Recombination operator, pattern density/stability, and pattern level classification were completed in prior sessions.

---

## 1. Hierarchical Total Score (D7)

### 1.1 Formula

From §3.5 of the design spec:

```
Total_i_hier(t) = A_i(t) + beta_comp * Comp_i(t) + J_i(t)
```

Where:
- `A_i(t) = -L_i(t)` — epistemic accuracy (unchanged)
- `Comp_i(t) = pattern.compress()` — compression score; for GaussianPattern this is the fraction of total variance in the pattern covariance explained by the dominant principal component (largest eigenvalue of `sigma` divided by the sum of all eigenvalues). Returns float in [0, 1]. Higher values indicate a more compact, structured pattern.
- `J_i(t) = beta_aff * E_aff_i + gamma_soc * E_soc_i` — non-epistemic term (unchanged)
- `beta_comp: float` — compression weight, default 0.0 (backward compatible)

### 1.2 Changes

**`hpm/config.py`** — add `beta_comp` in the evaluator weights block (near `beta_aff`, `gamma_soc`, `delta_cost`), with a comment referencing D7:
```python
beta_comp: float = 0.0   # compression bonus weight in hierarchical total score (D7)
```

**`hpm/agents/agent.py`** — modify totals computation to include compression term. Only call `compress()` when `beta_comp != 0.0` to avoid unconditional O(d³) overhead on every pattern every step:
```python
totals = np.array([
    epi
    + (self.config.beta_comp * p.compress() if self.config.beta_comp != 0.0 else 0.0)
    + self.config.beta_aff * e_aff
    + self.config.gamma_soc * e_soc
    + self.config.delta_cost * e_cost
    for p, epi, e_aff, e_soc, e_cost in zip(patterns, epistemic_accs, e_affs, e_socs, e_costs)
])
```

Also add `compress_mean` to the step return dict, gated by the same `beta_comp != 0.0` guard to avoid unconditional O(d³) overhead in the default case:
```python
'compress_mean': (
    float(np.mean([p.compress() for p in report_patterns]))
    if (report_patterns and self.config.beta_comp != 0.0)
    else 0.0
),
```

### 1.3 Backward Compatibility

With `beta_comp=0.0` (default), `compress()` is never called — neither in the totals computation nor in `compress_mean`. There is zero performance impact and results are identical to prior behaviour. `compress_mean` will be 0.0 in the return dict when `beta_comp=0.0`.

`compress()` on `GaussianPattern` uses `np.linalg.eigvalsh(sigma)`, which is O(d³) in feature dimension.

---

## 2. Symbol Sequence Domain

### 2.1 Structure

`hpm/domains/sequence.py` — implements the `Domain` protocol.

**Constructor signature:**
```python
def __init__(
    self,
    vocab_size: int = 8,
    order: int = 1,
    seed: int | None = None,
    transition: np.ndarray | None = None,   # if None, sampled via _random_transition
    label_map: np.ndarray | None = None,    # if None, identity permutation range(V)
):
    if order != 1:
        raise ValueError("only order=1 is supported")
```

**Parameters:**
| Parameter | Type | Default | Role |
|-----------|------|---------|------|
| `vocab_size` | int | 8 | Alphabet size V; also `feature_dim()` |
| `order` | int | 1 | Markov order; raises `ValueError` if != 1 |
| `seed` | int \| None | None | RNG seed for reproducibility |
| `transition` | ndarray \| None | None | Pre-built transition matrix (V, V); if None, sampled randomly |
| `label_map` | ndarray \| None | None | Pre-built label permutation (V,); if None, identity `[0,1,...,V-1]` |

**Internal state:**
- `_transition: np.ndarray` shape `(V, V)` — row-stochastic transition matrix; deep structure
- `_label_map: np.ndarray` shape `(V,)` — permutation of `range(V)`; surface structure (symbol labels)
- `_current: int` — current symbol index (in internal, pre-mapping space)
- `_rng: np.random.Generator`

### 2.2 Domain Protocol Implementation

```python
def observe(self) -> np.ndarray:
    """Sample next symbol, apply label map, return one-hot of dim vocab_size."""
    next_sym = int(self._rng.choice(self._vocab_size, p=self._transition[self._current]))
    self._current = next_sym
    x = np.zeros(self._vocab_size)
    x[self._label_map[next_sym]] = 1.0
    return x

def feature_dim(self) -> int:
    return self._vocab_size

def deep_perturb(self) -> 'SequenceDomain':
    """Return new domain with randomised transition matrix. Same label map.
    Uses a derived RNG so self._rng is not mutated."""
    derived_rng = np.random.default_rng(self._rng.integers(2**32))
    new_transition = self._random_transition(derived_rng, self._vocab_size)
    return SequenceDomain(self._vocab_size, seed=None,
                          transition=new_transition, label_map=self._label_map.copy())

def surface_perturb(self) -> 'SequenceDomain':
    """Return new domain with shuffled label map. Same transition matrix.
    Uses a derived RNG so self._rng is not mutated.
    Guarantees the new label_map differs from self._label_map (non-trivial perturbation)."""
    derived_rng = np.random.default_rng(self._rng.integers(2**32))
    new_label_map = derived_rng.permutation(self._vocab_size)
    while np.array_equal(new_label_map, self._label_map):
        new_label_map = derived_rng.permutation(self._vocab_size)
    return SequenceDomain(self._vocab_size, seed=None,
                          transition=self._transition.copy(), label_map=new_label_map)

def transfer_probe(self, near: bool) -> list[tuple[np.ndarray, int]]:
    """
    Generate 200 labelled observations without mutating self._current.

    Procedure:
    1. Create a probe domain with same _transition.
       near=True:  probe label_map = self._label_map.copy() — identical surface to training domain.
       near=False: probe label_map = reshuffled permutation that differs from self._label_map
                   (retry until label_map != self._label_map; guarantees non-trivial shift).
    2. Sample 200 steps from the probe domain starting from internal state 0,
       using a fresh derived RNG (does not affect self._rng or self._current).
    3. Each observation x is a one-hot of dim vocab_size (post label_map).
    4. Label is the internal symbol index (pre label_map; surface-invariant ground truth).
       Downstream: used by HPM metrics (hpm_predictions.py) to evaluate whether the
       agent's pattern library recovered the true transition structure.

    §9.1 validation logic: a pattern library that learned transition structure (not surface labels)
    should achieve low loss on near probes (same surface) and still-low loss on far probes
    (different surface). A pattern library that overfit to surface labels degrades on far probes.
    """
```

### 2.3 Deep/Surface Split

The key HPM validation (§9.1) requires that patterns learning deep structure (transition probabilities) should survive `surface_perturb()` better than `deep_perturb()`:

- **Deep structure** = `_transition` matrix — which symbols follow which (statistically)
- **Surface structure** = `_label_map` — which observed one-hot index maps to which internal symbol

`surface_perturb()` rotates the one-hot basis by permuting labels, preserving the underlying statistical structure. `deep_perturb()` changes the actual transition dynamics. A pattern that learned the transition structure degrades less under `surface_perturb()` than `deep_perturb()`.

### 2.4 Transition Matrix Generation

Default: Dirichlet-sampled rows with concentration parameter `alpha=1.0` (uniform on the simplex). The concentration is configurable to produce sparse (low alpha) or uniform (high alpha) transitions.

```python
@staticmethod
def _random_transition(rng, vocab_size, alpha=1.0) -> np.ndarray:
    rows = rng.dirichlet(np.full(vocab_size, alpha), size=vocab_size)
    return rows  # shape (V, V), row-stochastic
```

### 2.5 File Location

```
hpm/domains/sequence.py     # SequenceDomain class
hpm/domains/__init__.py     # add: from .sequence import SequenceDomain
tests/domains/test_sequence.py  # protocol compliance + perturbation tests
```

---

## 3. Testing Strategy

### Hierarchical total score
- `test_beta_comp_zero_compress_never_called_in_totals`: with `beta_comp=0`, mock `compress()` and assert it is not called during totals computation; `compress_mean` in return dict is 0.0
- `test_beta_comp_nonzero_affects_totals`: with `beta_comp>0`, patterns with higher `compress()` gain relative advantage over lower-compress patterns
- `test_compress_mean_nonzero_when_beta_comp_set`: with `beta_comp>0`, `compress_mean` in return dict is in (0, 1]

### Sequence domain
- `test_observe_returns_one_hot`: output is one-hot of dim `vocab_size` (exactly one 1.0, rest 0.0)
- `test_observe_stationary_distribution`: 10000-step run converges to theoretical stationary distribution (within tolerance)
- `test_feature_dim`: matches `vocab_size`
- `test_deep_perturb_changes_transition`: perturbed domain has different `_transition`; same `_label_map`; self._rng state unchanged after call
- `test_surface_perturb_preserves_transition`: perturbed domain has same `_transition`; different `_label_map`; self._rng state unchanged after call
- `test_transfer_probe_near_uses_source_label_map`: `near=True` probe uses `self._label_map.copy()` (identical surface to training)
- `test_transfer_probe_far_differs_from_source_label_map`: `near=False` probe label_map differs from `self._label_map` (non-trivial surface shift)
- `test_transfer_probe_shapes`: both probes return 200 `(ndarray, int)` tuples; ndarray shape `(vocab_size,)`; label in `range(vocab_size)`
- `test_transfer_probe_does_not_mutate_state`: calling `transfer_probe()` does not change `self._current` or advance `self._rng`
- `test_order_not_one_raises`: `SequenceDomain(order=2)` raises `ValueError`
- `test_domain_protocol_compliance`: SequenceDomain has all required methods (`observe`, `feature_dim`, `deep_perturb`, `surface_perturb`, `transfer_probe`)
- `test_reproducible_with_seed`: two domains with same seed produce identical 50-step sequences

---

## 4. What Is NOT in Scope

- Higher-order Markov chains (order > 1) — deferred; `order` param reserved but only order=1 implemented
- Sequence prediction accuracy metrics — existing `hpm_predictions.py` suffices for §9.1 validation
- ExternalSubstrate integration for sequence domain — domain is self-contained
