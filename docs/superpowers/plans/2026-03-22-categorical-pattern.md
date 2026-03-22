# CategoricalPattern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CategoricalPattern` as a third pattern type — a D×K probability matrix modelling discrete symbol sequences — completing the polymorphic pattern store Phase 2.

**Architecture:** `CategoricalPattern` mirrors `LaplacePattern`'s interface: no `sigma` attribute (routes MetaPatternRule to Monte Carlo KL), value-type `update()`, `sample()` returning integer arrays. The factory gains an `alphabet_size` parameter; `AgentConfig` gains `alphabet_size: int = 10`. No specialist classes (TieredStore, StructuralLawMonitor, etc.) require changes.

**Tech Stack:** Python 3.11+, NumPy. No new dependencies.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `hpm/patterns/categorical.py` | Create | CategoricalPattern class |
| `hpm/patterns/factory.py` | Modify | Add categorical branch to `make_pattern()` and `pattern_from_dict()` |
| `hpm/config.py` | Modify | Add `alphabet_size: int = 10` field |
| `tests/patterns/test_categorical.py` | Create | Unit tests for CategoricalPattern |
| `tests/integration/test_categorical_arc.py` | Create | Integration test: 3-agent ensemble with categorical agent |

---

## Task 1: CategoricalPattern class

**Files:**
- Create: `hpm/patterns/categorical.py`
- Test: `tests/patterns/test_categorical.py`

**Background:** `LaplacePattern` in `hpm/patterns/laplace.py` is the reference. Follow its structure exactly: `__init__`, `log_prob`, `sample`, `update`, `recombine`, `description_length`, `connectivity`, `compress`, `is_structurally_valid`, `to_dict`, `from_dict`. Key differences:

- Parameters are `probs` (D×K float array) and `K` (int), not `mu`/`b`
- `_n_obs` initialised to `K` (pseudo-count for Dirichlet prior), not 0
- `log_prob(x)` takes an **integer** array (values 0…K-1), not float
- `sample()` returns an **integer** array, not float
- `recombine()` uses `_n_obs`-weighted average (not 0.5/0.5)
- `K < 2` raises `ValueError` at construction

- [ ] **Step 1: Write the failing tests**

Create `tests/patterns/test_categorical.py`:

```python
import numpy as np
import pytest
from hpm.patterns.categorical import CategoricalPattern
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def pattern():
    rng = np.random.default_rng(0)
    D, K = 4, 5
    probs = np.ones((D, K)) / K
    return CategoricalPattern(probs, K=K)


def test_log_prob_at_mode_lower_than_improbable(pattern):
    """log_prob at the most probable symbol should be lower than at a rare symbol."""
    # After updates, symbol 0 becomes dominant
    p = pattern
    for _ in range(50):
        p = p.update(np.zeros(4, dtype=int))
    # Symbol 0 is now most probable at every position
    best = np.zeros(4, dtype=int)
    worst = np.ones(4, dtype=int) * (5 - 1)
    assert p.log_prob(best) < p.log_prob(worst)


def test_log_prob_finite(pattern):
    x = np.zeros(4, dtype=int)
    assert np.isfinite(pattern.log_prob(x))


def test_no_sigma_attribute(pattern):
    """Critical: absence of sigma routes MetaPatternRule to MC fallback."""
    assert not hasattr(pattern, 'sigma')


def test_initial_n_obs_equals_K(pattern):
    """_n_obs starts at K (pseudo-count for Dirichlet prior)."""
    assert pattern._n_obs == 5  # K=5


def test_update_increments_n_obs(pattern):
    x = np.zeros(4, dtype=int)
    updated = pattern.update(x)
    assert updated._n_obs == pattern._n_obs + 1


def test_update_shifts_prob_toward_observation():
    D, K = 3, 4
    probs = np.ones((D, K)) / K
    p = CategoricalPattern(probs, K=K)
    x = np.zeros(D, dtype=int)  # always observe symbol 0
    for _ in range(100):
        p = p.update(x)
    # Symbol 0 should dominate at every position
    for d in range(D):
        assert p.probs[d, 0] > 0.95


def test_probs_floor_prevents_zero():
    D, K = 2, 3
    probs = np.ones((D, K)) / K
    p = CategoricalPattern(probs, K=K)
    x = np.zeros(D, dtype=int)
    for _ in range(1000):
        p = p.update(x)
    assert np.all(p.probs >= 1e-8)


def test_sample_shape(pattern):
    rng = np.random.default_rng(42)
    samples = pattern.sample(50, rng)
    assert samples.shape == (50, 4)


def test_sample_values_in_range(pattern):
    rng = np.random.default_rng(0)
    samples = pattern.sample(200, rng)
    assert np.all(samples >= 0)
    assert np.all(samples < 5)  # K=5


def test_sample_dtype_integer(pattern):
    rng = np.random.default_rng(0)
    samples = pattern.sample(10, rng)
    assert np.issubdtype(samples.dtype, np.integer)


def test_recombine_weighted_by_n_obs():
    D, K = 2, 3
    # p1: 100 observations of symbol 0; p2: 10 observations of symbol 2
    probs1 = np.ones((D, K)) / K
    p1 = CategoricalPattern(probs1, K=K)
    for _ in range(100):
        p1 = p1.update(np.zeros(D, dtype=int))

    probs2 = np.ones((D, K)) / K
    p2 = CategoricalPattern(probs2, K=K)
    for _ in range(10):
        p2 = p2.update(np.full(D, 2, dtype=int))

    child = p1.recombine(p2)
    # p1 has far more observations so symbol 0 should still dominate
    assert child.probs[0, 0] > child.probs[0, 2]


def test_recombine_with_gaussian_raises(pattern):
    g = GaussianPattern(np.zeros(4), np.eye(4))
    with pytest.raises(TypeError):
        pattern.recombine(g)


def test_is_structurally_valid(pattern):
    assert pattern.is_structurally_valid()


def test_is_structurally_invalid_when_floor_violated():
    D, K = 2, 3
    probs = np.ones((D, K)) / K
    p = CategoricalPattern.__new__(CategoricalPattern)
    p.id = "test"
    p.level = 1
    p.source_id = None
    p.K = K
    p._n_obs = K
    p.probs = probs.copy()
    p.probs[0, 0] = 0.0  # violates floor
    assert not p.is_structurally_valid()


def test_to_dict_type_field(pattern):
    d = pattern.to_dict()
    assert d['type'] == 'categorical'


def test_to_dict_n_obs_key_no_underscore(pattern):
    d = pattern.to_dict()
    assert 'n_obs' in d
    assert '_n_obs' not in d


def test_roundtrip_serialisation(pattern):
    d = pattern.to_dict()
    restored = CategoricalPattern.from_dict(d)
    assert restored.id == pattern.id
    assert np.allclose(restored.probs, pattern.probs)
    assert restored.K == pattern.K
    assert restored.level == pattern.level
    assert restored._n_obs == pattern._n_obs


def test_id_preserved_on_update(pattern):
    updated = pattern.update(np.zeros(4, dtype=int))
    assert updated.id == pattern.id


def test_description_length_returns_float(pattern):
    dl = pattern.description_length()
    assert isinstance(dl, float)


def test_description_length_positive(pattern):
    # After learning a peaked distribution, some positions have low entropy
    p = pattern
    for _ in range(100):
        p = p.update(np.zeros(4, dtype=int))
    assert p.description_length() > 0.0


def test_connectivity_zero(pattern):
    assert pattern.connectivity() == 0.0


def test_compress_between_zero_and_max(pattern):
    c = pattern.compress()
    assert np.isfinite(c)
    assert c >= 0.0


def test_compress_returns_one_when_all_point_masses():
    """compress() returns 1.0 when mean entropy is 0 (zero-denominator guard)."""
    D, K = 2, 3
    # Create a near-point-mass: symbol 0 gets all mass
    probs = np.full((D, K), 1e-8)
    probs[:, 0] = 1.0 - (K - 1) * 1e-8
    p = CategoricalPattern(probs, K=K)
    assert p.compress() == 1.0


def test_k_less_than_2_raises():
    probs = np.ones((3, 1))
    with pytest.raises(ValueError):
        CategoricalPattern(probs, K=1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_categorical.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'hpm.patterns.categorical'`

- [ ] **Step 3: Implement `hpm/patterns/categorical.py`**

```python
import uuid
import numpy as np


class CategoricalPattern:
    """
    Pattern h = (probs,) — a product-of-categoricals generative model over discrete feature space.
    probs is a (D, K) matrix; each row is a probability distribution over K symbols.
    All entries floored at 1e-8. Rows sum to 1.

    Value type: update() returns a new instance; id is preserved on update,
    fresh on recombination.

    log_prob returns NLL (lower = more probable), consistent with GaussianPattern.
    No sigma attribute: routes sym_kl_normalised to Monte Carlo fallback via sample().
    _n_obs initialised to K (pseudo-count treating uniform prior as K observations).
    K < 2 raises ValueError.
    """

    def __init__(self, probs: np.ndarray, K: int, id: str | None = None,
                 level: int = 1, source_id: str | None = None):
        if K < 2:
            raise ValueError(f"CategoricalPattern requires K >= 2, got K={K}")
        self.id = id or str(uuid.uuid4())
        self.K = K
        self.probs = np.maximum(np.array(probs, dtype=float), 1e-8)
        # Renormalise rows after floor application
        self.probs = self.probs / self.probs.sum(axis=1, keepdims=True)
        self.level = level
        self._n_obs: int = K  # pseudo-count: uniform prior = K observations
        self.source_id = source_id

    def log_prob(self, x: np.ndarray) -> float:
        """Return NLL: -sum(log(probs[d, x[d]])). Lower = more probable.
        x must be a D-length integer array with values in {0…K-1}.
        Out-of-range values are the caller's responsibility (NumPy raises IndexError).
        """
        x = np.asarray(x, dtype=int)
        D = self.probs.shape[0]
        nll = 0.0
        for d in range(D):
            nll -= np.log(self.probs[d, x[d]])
        return float(nll)

    def sample(self, n: int, rng) -> np.ndarray:
        """Return n samples, shape (n, D), integer dtype, values in {0…K-1}.
        Required by MetaPatternRule's Monte Carlo KL branch (which calls p.sample()
        when hasattr(p, 'sigma') is False).
        """
        D = self.probs.shape[0]
        samples = np.empty((n, D), dtype=int)
        for d in range(D):
            samples[:, d] = rng.choice(self.K, size=n, p=self.probs[d])
        return samples

    def update(self, x: np.ndarray) -> 'CategoricalPattern':
        """Online Bayesian count update. Returns new instance (value-type semantics).
        Uses pre-update probs (probs_old) for the increment — consistent with
        LaplacePattern's mu_old convention. id and level preserved.
        """
        x = np.asarray(x, dtype=int)
        D, K = self.probs.shape
        n = self._n_obs + 1
        # One-hot encode x for each position
        one_hot = np.zeros((D, K), dtype=float)
        for d in range(D):
            one_hot[d, x[d]] = 1.0
        new_probs = (self.probs * self._n_obs + one_hot) / n
        new_probs = np.maximum(new_probs, 1e-8)
        new_probs = new_probs / new_probs.sum(axis=1, keepdims=True)
        new_p = CategoricalPattern(new_probs, K=self.K, id=self.id,
                                   level=self.level, source_id=self.source_id)
        new_p._n_obs = n
        return new_p

    def recombine(self, other: 'CategoricalPattern') -> 'CategoricalPattern':
        """_n_obs-weighted average of probs matrices.
        Raises TypeError if other is not CategoricalPattern.
        Raises ValueError if D or K differ.
        Falls back to uniform (0.5/0.5) average if both _n_obs are 0.
        """
        if not isinstance(other, CategoricalPattern):
            raise TypeError(f"Cannot recombine CategoricalPattern with {type(other).__name__}")
        if self.probs.shape != other.probs.shape:
            raise ValueError(
                f"Cannot recombine CategoricalPattern with shapes "
                f"{self.probs.shape} and {other.probs.shape}"
            )
        total = self._n_obs + other._n_obs
        if total == 0:
            new_probs = 0.5 * self.probs + 0.5 * other.probs
        else:
            new_probs = (self.probs * self._n_obs + other.probs * other._n_obs) / total
        new_probs = np.maximum(new_probs, 1e-8)
        new_probs = new_probs / new_probs.sum(axis=1, keepdims=True)
        return CategoricalPattern(new_probs, K=self.K)

    def description_length(self) -> float:
        """Count of positions with entropy below half of maximum entropy (log K * 0.5).
        Low-entropy positions have learned a definite preference.
        Returns float to match the Pattern protocol return type.
        """
        D = self.probs.shape[0]
        max_half_entropy = np.log(self.K) * 0.5
        count = 0
        for d in range(D):
            h = -np.sum(self.probs[d] * np.log(self.probs[d]))
            if h < max_half_entropy:
                count += 1
        return float(count)

    def connectivity(self) -> float:
        """Always 0.0 — independence assumption across positions."""
        return 0.0

    def compress(self) -> float:
        """max_row_entropy / mean_row_entropy.
        Returns 1.0 when mean entropy is 0 (all rows are point masses).
        """
        entropies = np.array([
            -np.sum(self.probs[d] * np.log(self.probs[d]))
            for d in range(self.probs.shape[0])
        ])
        mean_h = float(entropies.mean())
        if mean_h == 0.0:
            return 1.0
        return float(entropies.max() / mean_h)

    def is_structurally_valid(self) -> bool:
        """True iff all probs >= 1e-8 (floor integrity) and rows sum to 1 within 1e-6."""
        if not np.all(self.probs >= 1e-8):
            return False
        row_sums = self.probs.sum(axis=1)
        return bool(np.all(np.abs(row_sums - 1.0) < 1e-6))

    def to_dict(self) -> dict:
        return {
            'type': 'categorical',
            'id': self.id,
            'probs': self.probs.tolist(),
            'K': self.K,
            'n_obs': self._n_obs,
            'level': self.level,
            'source_id': self.source_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CategoricalPattern':
        p = cls(np.array(d['probs']), K=d['K'], id=d['id'],
                level=d.get('level', 1), source_id=d.get('source_id', None))
        p._n_obs = d['n_obs']
        return p
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_categorical.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpm/patterns/categorical.py tests/patterns/test_categorical.py
git commit -m "feat: add CategoricalPattern (D×K probability matrix, discrete pattern type)"
```

---

## Task 2: Factory + AgentConfig extension

**Files:**
- Modify: `hpm/patterns/factory.py`
- Modify: `hpm/config.py`

**Background:** `factory.py` currently has `make_pattern(mu, scale, pattern_type, **kwargs)` and `pattern_from_dict()`. Both need a categorical branch. `config.py` needs `alphabet_size: int = 10`.

- [ ] **Step 1: Write the failing tests**

Create `tests/patterns/test_factory_categorical.py`:

```python
import numpy as np
import pytest
from hpm.patterns.factory import make_pattern, pattern_from_dict
from hpm.patterns.categorical import CategoricalPattern
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern


def test_make_pattern_categorical_returns_categorical():
    mu = np.zeros(4)
    p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=6)
    assert isinstance(p, CategoricalPattern)


def test_make_pattern_categorical_uses_mu_for_dimension():
    mu = np.zeros(8)
    p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=5)
    assert p.probs.shape == (8, 5)


def test_make_pattern_categorical_uniform_init():
    mu = np.zeros(3)
    p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=4)
    expected = np.ones((3, 4)) / 4
    assert np.allclose(p.probs, expected, atol=1e-6)


def test_make_pattern_categorical_default_alphabet_size():
    mu = np.zeros(3)
    p = make_pattern(mu, scale=1.0, pattern_type="categorical")
    assert p.K == 10  # default


def test_make_pattern_gaussian_unchanged():
    mu = np.zeros(4)
    scale = np.eye(4)
    p = make_pattern(mu, scale, pattern_type="gaussian")
    assert isinstance(p, GaussianPattern)


def test_make_pattern_laplace_unchanged():
    mu = np.zeros(4)
    p = make_pattern(mu, scale=1.0, pattern_type="laplace")
    assert isinstance(p, LaplacePattern)


def test_make_pattern_unknown_raises():
    with pytest.raises(ValueError, match="categorical"):
        make_pattern(np.zeros(3), 1.0, pattern_type="von_mises")


def test_pattern_from_dict_categorical():
    mu = np.zeros(3)
    p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=4)
    d = p.to_dict()
    restored = pattern_from_dict(d)
    assert isinstance(restored, CategoricalPattern)
    assert np.allclose(restored.probs, p.probs)
    assert restored.K == p.K


def test_pattern_from_dict_unknown_raises():
    with pytest.raises(ValueError):
        pattern_from_dict({'type': 'von_mises'})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_factory_categorical.py -v 2>&1 | head -20
```

Expected: ImportError or test failures on the categorical branches.

- [ ] **Step 3: Update `hpm/patterns/factory.py`**

Replace the entire file:

```python
import numpy as np
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern
from hpm.patterns.categorical import CategoricalPattern


def make_pattern(mu, scale, pattern_type: str = "gaussian", alphabet_size: int = 10, **kwargs):
    """Construct a pattern from (mu, scale) parameters.

    Args:
        mu: Location vector (ndarray or list). For categorical patterns, used only
            for dimensionality D = len(mu); not stored in the pattern.
        scale: For Gaussian: covariance matrix. For Laplace: scale vector b
               (scalar is broadcast to np.ones(len(mu)) * scalar).
               Ignored for categorical patterns.
        pattern_type: "gaussian", "laplace", or "categorical".
        alphabet_size: K for categorical patterns (ignored by gaussian/laplace).
        **kwargs: Passed to the pattern constructor (id, level, source_id, freeze_mu).
    """
    mu = np.asarray(mu, dtype=float)
    if pattern_type == "gaussian":
        return GaussianPattern(mu, scale, **kwargs)
    elif pattern_type == "laplace":
        b = np.ones(len(mu)) * scale if np.isscalar(scale) else np.asarray(scale, dtype=float)
        return LaplacePattern(mu, b, **kwargs)
    elif pattern_type == "categorical":
        K = kwargs.pop("K", alphabet_size)
        D = len(mu)
        probs = np.ones((D, K)) / K  # uniform = maximum entropy prior
        return CategoricalPattern(probs, K=K, **kwargs)
    else:
        raise ValueError(
            f"Unknown pattern_type: {pattern_type!r}. "
            f"Expected 'gaussian', 'laplace', or 'categorical'."
        )


def pattern_from_dict(d: dict):
    """Deserialise a pattern from a dict produced by to_dict().

    Dispatches on d['type']. Defaults to 'gaussian' if type key is absent
    (backward compatibility with pre-factory serialised patterns).
    """
    t = d.get('type', 'gaussian')
    if t == 'gaussian':
        return GaussianPattern.from_dict(d)
    elif t == 'laplace':
        return LaplacePattern.from_dict(d)
    elif t == 'categorical':
        return CategoricalPattern.from_dict(d)
    else:
        raise ValueError(
            f"Unknown pattern type in dict: {t!r}. "
            f"Expected 'gaussian', 'laplace', or 'categorical'."
        )
```

- [ ] **Step 4: Add `alphabet_size` to `hpm/config.py`**

Open `hpm/config.py`. After the `pattern_type` line (currently the last field), add:

```python
    alphabet_size: int = 10          # K for CategoricalPattern (ignored by gaussian/laplace)
```

The `pattern_type` comment is already updated to include `"categorical"` from the spec review fix. Verify it reads:
```python
    pattern_type: str = "gaussian"  # "gaussian" | "laplace" | "categorical"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_factory_categorical.py tests/patterns/test_categorical.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -q
```

Expected: All existing tests still pass (497 passed, 9 skipped).

- [ ] **Step 7: Commit**

```bash
git add hpm/patterns/factory.py hpm/config.py tests/patterns/test_factory_categorical.py
git commit -m "feat: extend factory and AgentConfig for CategoricalPattern (alphabet_size field)"
```

---

## Task 3: Integration test — 3-agent ensemble

**Files:**
- Create: `tests/integration/test_categorical_arc.py`

**Background:** This test verifies that a 3-agent orchestrator (Gaussian + Laplace + Categorical) runs correctly, with each agent receiving the appropriate type of observation. It does not test accuracy — just that the plumbing works end-to-end without errors.

The test uses `make_orchestrator` from `benchmarks/multi_agent_common.py` with `pattern_types=["gaussian", "laplace", "categorical"]`. The categorical agent receives integer-encoded observations (raw ARC grid colour values, flattened, padded to FEATURE_DIM=64 with zeros, clipped to {0…9}).

- [ ] **Step 1: Write the test**

Create `tests/integration/test_categorical_arc.py`:

```python
"""Integration test: 3-agent ensemble with Gaussian, Laplace, and Categorical patterns.

Verifies that:
- make_orchestrator accepts pattern_types=["gaussian", "laplace", "categorical"]
- Each agent receives correctly-typed observations without error
- ensemble_score returns finite values for all candidate vectors
- orch.step() runs 10 steps cleanly
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.multi_agent_common import make_orchestrator
from hpm.patterns.categorical import CategoricalPattern

FEATURE_DIM = 64
K = 10  # ARC colour alphabet: 0-9


def make_float_obs(rng):
    """Delta-encoded float observation for Gaussian/Laplace agents."""
    return rng.standard_normal(FEATURE_DIM)


def make_int_obs(rng):
    """Integer-encoded colour observation for Categorical agent.
    Simulates a flattened ARC grid padded to FEATURE_DIM, values 0-9.
    """
    return rng.integers(0, K, size=FEATURE_DIM)


def test_three_agent_ensemble_constructs():
    orch, agents, store = make_orchestrator(
        n_agents=3,
        feature_dim=FEATURE_DIM,
        agent_ids=["arc_gauss", "arc_laplace", "arc_cat"],
        pattern_types=["gaussian", "laplace", "categorical"],
        alphabet_size=K,
        beta_comp=0.0,
        gamma_soc=0.0,  # disable field sharing (cross-type recombination would TypeError)
    )
    assert len(agents) == 3
    from hpm.patterns.categorical import CategoricalPattern
    # Verify categorical agent has correct config
    cat_agent = next(a for a in agents if a.agent_id == "arc_cat")
    assert cat_agent.config.pattern_type == "categorical"
    assert cat_agent.config.alphabet_size == K


def test_three_agent_step_runs():
    orch, agents, store = make_orchestrator(
        n_agents=3,
        feature_dim=FEATURE_DIM,
        agent_ids=["arc_gauss", "arc_laplace", "arc_cat"],
        pattern_types=["gaussian", "laplace", "categorical"],
        alphabet_size=K,
        beta_comp=0.0,
        gamma_soc=0.0,
    )
    rng = np.random.default_rng(42)
    for _ in range(10):
        obs = {
            "arc_gauss": make_float_obs(rng),
            "arc_laplace": make_float_obs(rng),
            "arc_cat": make_int_obs(rng),
        }
        orch.step(obs)  # must not raise


def test_categorical_agent_log_prob_finite():
    orch, agents, store = make_orchestrator(
        n_agents=3,
        feature_dim=FEATURE_DIM,
        agent_ids=["arc_gauss", "arc_laplace", "arc_cat"],
        pattern_types=["gaussian", "laplace", "categorical"],
        alphabet_size=K,
        beta_comp=0.0,
        gamma_soc=0.0,
    )
    rng = np.random.default_rng(0)
    # Train categorical agent on some observations
    cat_agent = next(a for a in agents if a.agent_id == "arc_cat")
    for _ in range(5):
        obs = {
            "arc_gauss": make_float_obs(rng),
            "arc_laplace": make_float_obs(rng),
            "arc_cat": make_int_obs(rng),
        }
        orch.step(obs)

    # Score a candidate integer vector against categorical agent
    candidate_int = make_int_obs(rng)
    records = cat_agent.store.query("arc_cat")
    if records:
        for p, w in records:
            score = p.log_prob(candidate_int)
            assert np.isfinite(score), f"Non-finite log_prob: {score}"


def test_pattern_types_are_correct():
    """Each agent should seed patterns of the correct type."""
    orch, agents, store = make_orchestrator(
        n_agents=3,
        feature_dim=FEATURE_DIM,
        agent_ids=["arc_gauss", "arc_laplace", "arc_cat"],
        pattern_types=["gaussian", "laplace", "categorical"],
        alphabet_size=K,
        beta_comp=0.0,
        gamma_soc=0.0,
        agent_seeds=[1, 2, 3],  # force seeding so patterns exist
    )
    from hpm.patterns.gaussian import GaussianPattern
    from hpm.patterns.laplace import LaplacePattern

    for agent in agents:
        records = agent.store.query(agent.agent_id)
        if records:
            for p, _ in records:
                if agent.agent_id == "arc_gauss":
                    assert isinstance(p, GaussianPattern)
                elif agent.agent_id == "arc_laplace":
                    assert isinstance(p, LaplacePattern)
                elif agent.agent_id == "arc_cat":
                    assert isinstance(p, CategoricalPattern)
```

- [ ] **Step 2: Run tests to verify they fail (or error)**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/integration/test_categorical_arc.py -v 2>&1 | head -30
```

Expected: Failures related to `pattern_type="categorical"` not yet being handled in `make_orchestrator` or `Agent._seed_if_empty`. Note which errors appear.

- [ ] **Step 3: Check Agent._seed_if_empty handles categorical**

Open `hpm/agents/agent.py` and find `_seed_if_empty`. It calls `make_pattern(mu=..., scale=..., pattern_type=self.config.pattern_type)`. With the factory update from Task 2, this should work for categorical — but it also needs to pass `alphabet_size=self.config.alphabet_size`.

Check the current call site:

```bash
grep -n "make_pattern\|alphabet_size" hpm/agents/agent.py
```

If `alphabet_size` is not passed, add it. Find the `make_pattern(...)` call in `agent.py` and update it to:

```python
make_pattern(
    mu=...,
    scale=...,
    pattern_type=self.config.pattern_type,
    alphabet_size=self.config.alphabet_size,
)
```

Do the same for any other `make_pattern` call in `agent.py`.

- [ ] **Step 4: Run integration tests to verify they pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/integration/test_categorical_arc.py -v
```

Expected: All 4 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -q
```

Expected: All tests pass (at least 497 passed, 9 skipped, plus the new tests).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_categorical_arc.py hpm/agents/agent.py
git commit -m "feat: wire CategoricalPattern into Agent and integration tests; 3-agent ensemble verified"
```
