# Phase 4 Design Specification: Hierarchical Score + Sequence Domain

**Date:** 2026-03-21
**Status:** Draft v1

---

## Overview

Phase 4 completes the HPM agent framework with two remaining items from the original design (§8):

1. **Hierarchical total score (D7)** — adds a compression bonus term `beta_comp * Comp_i(t)` to the per-pattern total score, rewarding patterns with high mutual information between latent layers.
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
- `Comp_i(t) = pattern.compress()` — compression score; for GaussianPattern this is variance explained by the abstract cluster centroid z^(2) relative to total variance (mutual information proxy), returns float in [0, 1]
- `J_i(t) = beta_aff * E_aff_i + gamma_soc * E_soc_i` — non-epistemic term (unchanged)
- `beta_comp: float` — compression weight, default 0.0 (backward compatible)

### 1.2 Changes

**`hpm/config.py`** — add one field after `delta_cost`:
```python
beta_comp: float = 0.0   # compression bonus weight in hierarchical total score (D7)
```

**`hpm/agents/agent.py`** — modify totals computation to include compression term:
```python
totals = np.array([
    epi + self.config.beta_comp * p.compress()
    + self.config.beta_aff * e_aff
    + self.config.gamma_soc * e_soc
    + self.config.delta_cost * e_cost
    for p, epi, e_aff, e_soc, e_cost in zip(patterns, epistemic_accs, e_affs, e_socs, e_costs)
])
```

Also add `compress_mean` to the step return dict:
```python
'compress_mean': float(np.mean([p.compress() for p in report_patterns])) if report_patterns else 0.0,
```

### 1.3 Backward Compatibility

With `beta_comp=0.0` (default), `compress()` is still called but multiplied by zero — results are identical to prior behaviour. No migration needed for existing configs or tests.

`compress()` on `GaussianPattern` is already implemented and O(d²) in feature dimension — cheap.

---

## 2. Symbol Sequence Domain

### 2.1 Structure

`hpm/domains/sequence.py` — implements the `Domain` protocol.

**Parameters:**
| Parameter | Type | Default | Role |
|-----------|------|---------|------|
| `vocab_size` | int | 8 | Alphabet size V; also `feature_dim()` |
| `order` | int | 1 | Markov order (currently only 1 supported) |
| `seed` | int \| None | None | RNG seed for reproducibility |

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
    """Return new domain with randomised transition matrix. Same label map."""
    new_transition = self._random_transition(self._rng)
    return SequenceDomain(self._vocab_size, seed=None,
                          transition=new_transition, label_map=self._label_map.copy())

def surface_perturb(self) -> 'SequenceDomain':
    """Return new domain with shuffled label map. Same transition matrix."""
    new_label_map = self._rng.permutation(self._vocab_size)
    return SequenceDomain(self._vocab_size, seed=None,
                          transition=self._transition.copy(), label_map=new_label_map)

def transfer_probe(self, near: bool) -> list[tuple[np.ndarray, int]]:
    """
    Generate 200 labelled observations for transfer evaluation.
    near=True: same transitions, minor surface variation (swap 2 labels).
    near=False: same transitions, fully reshuffled label map (far transfer).
    Label is the internal symbol index (ground truth, surface-invariant).
    """
```

### 2.3 Deep/Surface Split

The key HPM validation (§9.1) requires that patterns learning deep structure (transition probabilities) should survive `surface_perturb()` better than `deep_perturb()`:

- **Deep structure** = `_transition` matrix — which symbols follow which
- **Surface structure** = `_label_map` — which observed one-hot index maps to which internal symbol

A `GaussianPattern` that has learned the true transition structure has mu/Sigma aligned with the transition eigenvectors. Permuting labels (`surface_perturb`) rotates the one-hot basis but preserves the statistical structure; the pattern's log-prob on new observations degrades less than under `deep_perturb` which changes the actual transition dynamics.

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
hpm/domains/__init__.py     # add SequenceDomain to exports
tests/domains/test_sequence.py  # protocol compliance + perturbation tests
```

---

## 3. Testing Strategy

### Hierarchical total score
- `test_beta_comp_zero_unchanged`: with `beta_comp=0`, step output matches prior behaviour
- `test_beta_comp_nonzero_affects_totals`: with `beta_comp>0`, patterns with higher `compress()` gain relative advantage
- `test_compress_mean_in_return_dict`: verify key present and in [0,1]

### Sequence domain
- `test_observe_returns_one_hot`: output is one-hot of dim `vocab_size`
- `test_observe_stationary_distribution`: long run converges to theoretical stationary dist
- `test_feature_dim`: matches `vocab_size`
- `test_deep_perturb_changes_transition`: perturbed domain has different `_transition`
- `test_surface_perturb_preserves_transition`: perturbed domain has same `_transition`, different `_label_map`
- `test_transfer_probe_near_far`: near and far probes have correct shapes and label ranges
- `test_domain_protocol_compliance`: SequenceDomain satisfies Domain protocol (all methods present)
- `test_reproducible_with_seed`: same seed produces same sequence

---

## 4. What Is NOT in Scope

- Higher-order Markov chains (order > 1) — deferred; architecture supports it via `order` param but only order=1 implemented
- Sequence prediction metrics (accuracy of next-symbol prediction) — existing HPM metrics (`hpm_predictions.py`) suffice for §9.1 validation
- ExternalSubstrate integration for sequence domain — not needed; domain is self-contained
