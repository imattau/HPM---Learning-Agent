# Polymorphic Pattern Store — Phase 1: LaplacePattern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `LaplacePattern` as a second pattern type, establishing the minimal polymorphic protocol so agents can be configured to use Laplace instead of Gaussian distributions.

**Architecture:** Each agent is configured with a `pattern_type` field in `AgentConfig`; a factory creates the right pattern type at initialisation. `LaplacePattern` exposes `sample()` instead of `sigma`, and a two-line fix to `MetaPatternRule` routes it through the existing Monte Carlo KL fallback. All stores, fields, and benchmarks work unchanged.

**Tech Stack:** Python 3.11+, NumPy. No new dependencies.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `hpm/patterns/laplace.py` | Create | LaplacePattern class |
| `hpm/patterns/factory.py` | Create | make_pattern() + pattern_from_dict() |
| `hpm/patterns/gaussian.py` | Modify | Add sample() method |
| `hpm/dynamics/meta_pattern_rule.py` | Modify | 2-line MC fallback fix |
| `hpm/config.py` | Modify | Add pattern_type field |
| `benchmarks/multi_agent_common.py` | Modify | pattern_types param in make_orchestrator() |
| `benchmarks/multi_agent_arc.py` | Modify | pattern_types in evaluate_task() + convenience fn |
| `tests/patterns/test_laplace.py` | Create | LaplacePattern unit tests |
| `tests/patterns/test_gaussian_sample.py` | Create | GaussianPattern.sample() unit test |
| `tests/patterns/test_factory.py` | Create | Factory unit tests |
| `tests/integration/test_laplace_arc.py` | Create | End-to-end integration test |

---

## Task 1: LaplacePattern class

**Files:**
- Create: `hpm/patterns/laplace.py`
- Test: `tests/patterns/test_laplace.py`

### Background

`GaussianPattern` in `hpm/patterns/gaussian.py` is the reference implementation. `LaplacePattern` mirrors its interface but uses a location `mu` + per-dimension scale `b` (no full covariance matrix). Key difference: **no `sigma` attribute** — this is what makes `MetaPatternRule.sym_kl_normalised` route to the Monte Carlo fallback instead of the Cholesky KL path.

The Pattern protocol (in `hpm/patterns/base.py`) requires: `id`, `level`, `log_prob`, `description_length`, `connectivity`, `compress`, `update`, `recombine`, `is_structurally_valid`, `to_dict`, `from_dict`.

Run tests with: `cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_laplace.py -v`

---

- [ ] **Step 1: Write the failing tests**

Create `tests/patterns/test_laplace.py`:

```python
import numpy as np
import pytest
from hpm.patterns.laplace import LaplacePattern
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def pattern():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(4)
    b = np.array([0.5, 1.0, 1.5, 2.0])
    return LaplacePattern(mu, b)


def test_log_prob_at_mu_is_minimum(pattern):
    """log_prob at mu should be lower (more probable) than at a distant point."""
    far = pattern.mu + 10.0
    assert pattern.log_prob(pattern.mu) < pattern.log_prob(far)


def test_log_prob_finite(pattern):
    x = np.ones(4)
    assert np.isfinite(pattern.log_prob(x))


def test_no_sigma_attribute(pattern):
    """Critical: absence of sigma routes MetaPatternRule to MC fallback."""
    assert not hasattr(pattern, 'sigma')


def test_update_increments_n_obs(pattern):
    x = np.ones(4)
    updated = pattern.update(x)
    assert updated._n_obs == pattern._n_obs + 1


def test_update_moves_mu_toward_observation(pattern):
    # After many updates toward a fixed point, mu should move toward it
    target = np.ones(4) * 5.0
    p = LaplacePattern(np.zeros(4), np.ones(4))
    for _ in range(100):
        p = p.update(target)
    assert np.allclose(p.mu, target, atol=0.1)


def test_b_updates_using_mu_old():
    """b converges toward |x - mu_old| after repeated identical observations."""
    # Start with mu=0, b=10 (large), observe x=3 repeatedly.
    # After convergence: mu → 3.0, mu_old → 3.0, so |x - mu_old| → 0, so b → 0 (floored at 1e-6).
    p = LaplacePattern(np.zeros(2), np.ones(2) * 10.0)
    x = np.array([3.0, 3.0])
    for _ in range(500):
        p = p.update(x)
    # b should have contracted sharply from the initial 10.0
    assert np.all(p.b < 0.1)


def test_b_floor_prevents_zero():
    p = LaplacePattern(np.zeros(2), np.ones(2) * 10.0)
    x = np.zeros(2)  # |x - mu| = 0 every step
    for _ in range(1000):
        p = p.update(x)
    assert np.all(p.b >= 1e-6)


def test_b_floor_on_construction():
    p = LaplacePattern(np.zeros(2), np.array([-1.0, 0.0]))
    assert np.all(p.b >= 1e-6)


def test_sample_shape(pattern):
    rng = np.random.default_rng(42)
    samples = pattern.sample(50, rng)
    assert samples.shape == (50, 4)


def test_sample_mean_close_to_mu():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -2.0, 3.0])
    b = np.ones(3) * 0.1
    p = LaplacePattern(mu, b)
    samples = p.sample(2000, rng)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)


def test_connectivity_always_zero(pattern):
    assert pattern.connectivity() == 0.0


def test_compress_ratio(pattern):
    c = pattern.compress()
    assert 0.0 <= c <= 1.0


def test_is_structurally_valid(pattern):
    assert pattern.is_structurally_valid()


def test_is_structurally_invalid_when_b_zero():
    p = LaplacePattern.__new__(LaplacePattern)
    p.mu = np.zeros(3)
    p.b = np.array([0.0, 1.0, 1.0])
    assert not p.is_structurally_valid()


def test_recombine_averages_mu_and_b():
    p1 = LaplacePattern(np.zeros(3), np.ones(3))
    p2 = LaplacePattern(np.ones(3) * 2.0, np.ones(3) * 3.0)
    child = p1.recombine(p2)
    assert np.allclose(child.mu, np.ones(3))
    assert np.allclose(child.b, np.ones(3) * 2.0)


def test_recombine_with_gaussian_raises():
    p = LaplacePattern(np.zeros(3), np.ones(3))
    g = GaussianPattern(np.zeros(3), np.eye(3))
    with pytest.raises(TypeError):
        p.recombine(g)


def test_to_dict_type_field(pattern):
    d = pattern.to_dict()
    assert d['type'] == 'laplace'


def test_roundtrip_serialisation(pattern):
    d = pattern.to_dict()
    restored = LaplacePattern.from_dict(d)
    assert restored.id == pattern.id
    assert np.allclose(restored.mu, pattern.mu)
    assert np.allclose(restored.b, pattern.b)
    assert restored.level == pattern.level
    assert restored._n_obs == pattern._n_obs


def test_id_preserved_on_update(pattern):
    updated = pattern.update(np.ones(4))
    assert updated.id == pattern.id


def test_description_length_positive(pattern):
    assert pattern.description_length() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_laplace.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError` for `hpm.patterns.laplace`

- [ ] **Step 3: Implement LaplacePattern**

Create `hpm/patterns/laplace.py`:

```python
import uuid
import numpy as np


class LaplacePattern:
    """
    Pattern h = (mu, b) — a Laplace generative model over feature space.
    b is a per-dimension scale vector (all entries > 0, floored at 1e-6).
    Value type: update() returns a new instance; id is preserved on update,
    fresh on recombination.

    log_prob returns NLL (lower = more probable), consistent with GaussianPattern.
    No sigma attribute: routes sym_kl_normalised to Monte Carlo fallback via sample().
    """

    def __init__(self, mu: np.ndarray, b: np.ndarray, id: str | None = None,
                 level: int = 1, source_id: str | None = None, freeze_mu: bool = False):
        self.id = id or str(uuid.uuid4())
        self.mu = np.array(mu, dtype=float)
        self.b = np.maximum(np.array(b, dtype=float), 1e-6)
        self.level = level
        self._n_obs: int = 0
        self.source_id = source_id
        self.freeze_mu = freeze_mu

    def log_prob(self, x: np.ndarray) -> float:
        """Return NLL: sum(|x - mu| / b) + sum(log(2b)). Lower = more probable."""
        diff = np.asarray(x, dtype=float) - self.mu
        return float(np.sum(np.abs(diff) / self.b) + np.sum(np.log(2.0 * self.b)))

    def sample(self, n: int, rng) -> np.ndarray:
        """Return n samples from the Laplace distribution, shape (n, D)."""
        return rng.laplace(loc=self.mu, scale=self.b, size=(n, len(self.mu)))

    def update(self, x: np.ndarray) -> 'LaplacePattern':
        mu_old = self.mu
        n = self._n_obs + 1
        new_mu = mu_old if self.freeze_mu else (mu_old * self._n_obs + np.asarray(x, dtype=float)) / n
        # Residual uses mu_old (before shift) to avoid downward bias on b
        new_b = np.maximum((self.b * self._n_obs + np.abs(np.asarray(x, dtype=float) - mu_old)) / n, 1e-6)
        new_p = LaplacePattern(new_mu, new_b, id=self.id, level=self.level,
                               source_id=self.source_id, freeze_mu=self.freeze_mu)
        new_p._n_obs = n
        return new_p

    def recombine(self, other: 'LaplacePattern') -> 'LaplacePattern':
        if not isinstance(other, LaplacePattern):
            raise TypeError(f"Cannot recombine LaplacePattern with {type(other).__name__}")
        return LaplacePattern(0.5 * self.mu + 0.5 * other.mu,
                              0.5 * self.b + 0.5 * other.b)

    def description_length(self) -> float:
        return float(np.sum(np.abs(self.mu) > 1e-6) + self.b.shape[0])

    def connectivity(self) -> float:
        """Always 0.0 — no off-diagonal structure in diagonal-scale parameterisation."""
        return 0.0

    def compress(self) -> float:
        total = self.b.sum()
        if total == 0.0:
            return 0.0
        return float(self.b.max() / total)

    def is_structurally_valid(self) -> bool:
        return bool(np.all(self.b > 0))

    def to_dict(self) -> dict:
        return {
            'type': 'laplace',
            'id': self.id,
            'mu': self.mu.tolist(),
            'b': self.b.tolist(),
            'n_obs': self._n_obs,
            'level': self.level,
            'source_id': self.source_id,
            'freeze_mu': self.freeze_mu,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LaplacePattern':
        p = cls(np.array(d['mu']), np.array(d['b']), id=d['id'],
                level=d.get('level', 1), source_id=d.get('source_id', None),
                freeze_mu=d.get('freeze_mu', False))
        p._n_obs = d['n_obs']
        return p
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_laplace.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add hpm/patterns/laplace.py tests/patterns/test_laplace.py
git commit -m "feat: add LaplacePattern with log_prob, sample, update, recombine"
```

---

## Task 2: GaussianPattern.sample() + MetaPatternRule fix

**Files:**
- Modify: `hpm/patterns/gaussian.py` (add `sample()` method after `log_prob`)
- Modify: `hpm/dynamics/meta_pattern_rule.py` (lines 36–39: replace `rng.multivariate_normal` with `p.sample()`)
- Test: `tests/patterns/test_gaussian_sample.py`

### Background

`meta_pattern_rule.py` has a Monte Carlo fallback in `sym_kl_normalised` (lines 33–42). The else-branch currently calls `rng.multivariate_normal(p.mu, p.sigma, n_samples)` — this crashes for `LaplacePattern` which has no `sigma`. Fix: replace with `p.sample(n_samples, rng)`. Both pattern types must implement `sample(n, rng) -> np.ndarray`.

Current broken code in `meta_pattern_rule.py` lines 34–39:
```python
        if rng is None:
            rng = np.random.default_rng()
        samples_p = rng.multivariate_normal(p.mu, p.sigma, n_samples)
        kl_pq = float(np.mean([q.log_prob(s) - p.log_prob(s) for s in samples_p]))
        samples_q = rng.multivariate_normal(q.mu, q.sigma, n_samples)
        kl_qp = float(np.mean([p.log_prob(s) - q.log_prob(s) for s in samples_q]))
```

Run tests with: `cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_gaussian_sample.py -v`

---

- [ ] **Step 1: Write failing test for GaussianPattern.sample()**

Create `tests/patterns/test_gaussian_sample.py`:

```python
import numpy as np
from hpm.patterns.gaussian import GaussianPattern


def test_sample_shape():
    rng = np.random.default_rng(0)
    p = GaussianPattern(np.zeros(3), np.eye(3))
    samples = p.sample(50, rng)
    assert samples.shape == (50, 3)


def test_sample_mean_close_to_mu():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -2.0, 3.0])
    p = GaussianPattern(mu, np.eye(3) * 0.01)
    samples = p.sample(2000, rng)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_gaussian_sample.py -v
```

Expected: `AttributeError: 'GaussianPattern' object has no attribute 'sample'`

- [ ] **Step 3: Add sample() to GaussianPattern**

In `hpm/patterns/gaussian.py`, add after `log_prob()` (after line 42):

```python
    def sample(self, n: int, rng) -> np.ndarray:
        """Return n samples from the Gaussian distribution, shape (n, D)."""
        return rng.multivariate_normal(self.mu, self.sigma, n)
```

- [ ] **Step 4: Run GaussianPattern.sample() tests**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_gaussian_sample.py -v
```

Expected: PASS

- [ ] **Step 5: Fix MetaPatternRule Monte Carlo fallback**

In `hpm/dynamics/meta_pattern_rule.py`, replace lines 36–39 (the two `rng.multivariate_normal` calls in the else-branch):

Old:
```python
        samples_p = rng.multivariate_normal(p.mu, p.sigma, n_samples)
        kl_pq = float(np.mean([q.log_prob(s) - p.log_prob(s) for s in samples_p]))
        samples_q = rng.multivariate_normal(q.mu, q.sigma, n_samples)
        kl_qp = float(np.mean([p.log_prob(s) - q.log_prob(s) for s in samples_q]))
```

New:
```python
        samples_p = p.sample(n_samples, rng)
        kl_pq = float(np.mean([q.log_prob(s) - p.log_prob(s) for s in samples_p]))
        samples_q = q.sample(n_samples, rng)
        kl_qp = float(np.mean([p.log_prob(s) - q.log_prob(s) for s in samples_q]))
```

- [ ] **Step 6: Run full existing test suite to confirm nothing broken**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -v --ignore=tests/integration 2>&1 | tail -20
```

Expected: All existing tests PASS (plus the 2 new ones)

- [ ] **Step 7: Commit**

```bash
git add hpm/patterns/gaussian.py hpm/dynamics/meta_pattern_rule.py tests/patterns/test_gaussian_sample.py
git commit -m "feat: add GaussianPattern.sample(), fix MetaPatternRule MC fallback for non-Gaussian patterns"
```

---

## Task 3: Pattern factory

**Files:**
- Create: `hpm/patterns/factory.py`
- Test: `tests/patterns/test_factory.py`

### Background

The factory is the single place that knows about all pattern types. Two functions: `make_pattern(mu, scale, pattern_type)` constructs from parameters; `pattern_from_dict(d)` deserialises by dispatching on `d['type']`. No other module should import both `GaussianPattern` and `LaplacePattern` — they should use the factory.

Run tests with: `cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_factory.py -v`

---

- [ ] **Step 1: Write failing tests**

Create `tests/patterns/test_factory.py`:

```python
import numpy as np
import pytest
from hpm.patterns.factory import make_pattern, pattern_from_dict
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern


def test_make_pattern_gaussian():
    p = make_pattern(np.zeros(3), np.eye(3), pattern_type="gaussian")
    assert isinstance(p, GaussianPattern)


def test_make_pattern_laplace():
    p = make_pattern(np.zeros(3), np.ones(3), pattern_type="laplace")
    assert isinstance(p, LaplacePattern)


def test_make_pattern_laplace_scalar_scale_broadcast():
    p = make_pattern(np.zeros(4), 2.0, pattern_type="laplace")
    assert isinstance(p, LaplacePattern)
    assert p.b.shape == (4,)
    assert np.allclose(p.b, 2.0)


def test_make_pattern_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pattern_type"):
        make_pattern(np.zeros(3), np.eye(3), pattern_type="von_mises")


def test_pattern_from_dict_gaussian():
    p = GaussianPattern(np.zeros(3), np.eye(3))
    d = p.to_dict()
    restored = pattern_from_dict(d)
    assert isinstance(restored, GaussianPattern)
    assert restored.id == p.id


def test_pattern_from_dict_laplace():
    p = LaplacePattern(np.zeros(3), np.ones(3))
    d = p.to_dict()
    restored = pattern_from_dict(d)
    assert isinstance(restored, LaplacePattern)
    assert restored.id == p.id


def test_pattern_from_dict_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pattern type in dict"):
        pattern_from_dict({'type': 'dirac', 'mu': [0.0]})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_factory.py -v
```

Expected: `ModuleNotFoundError` for `hpm.patterns.factory`

- [ ] **Step 3: Implement factory**

Create `hpm/patterns/factory.py`:

```python
import numpy as np
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern


def make_pattern(mu, scale, pattern_type: str = "gaussian", **kwargs):
    """Construct a pattern from (mu, scale) parameters.

    Args:
        mu: Location vector (ndarray or list).
        scale: For Gaussian: covariance matrix. For Laplace: scale vector b
               (scalar is broadcast to np.ones(len(mu)) * scalar).
        pattern_type: "gaussian" or "laplace".
        **kwargs: Passed to the pattern constructor (id, level, source_id, freeze_mu).
    """
    mu = np.asarray(mu, dtype=float)
    if pattern_type == "gaussian":
        return GaussianPattern(mu, scale, **kwargs)
    elif pattern_type == "laplace":
        b = np.ones(len(mu)) * scale if np.isscalar(scale) else np.asarray(scale, dtype=float)
        return LaplacePattern(mu, b, **kwargs)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type!r}. Expected 'gaussian' or 'laplace'.")


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
    else:
        raise ValueError(f"Unknown pattern type in dict: {t!r}. Expected 'gaussian' or 'laplace'.")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/patterns/test_factory.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add hpm/patterns/factory.py tests/patterns/test_factory.py
git commit -m "feat: add pattern factory with make_pattern() and pattern_from_dict()"
```

---

## Task 4: AgentConfig + orchestrator wiring

**Files:**
- Modify: `hpm/config.py` (add `pattern_type` field at end of dataclass)
- Modify: `benchmarks/multi_agent_common.py` (add `pattern_types` param to `make_orchestrator`)
- Modify: `benchmarks/multi_agent_arc.py` (add `pattern_types` to `evaluate_task` + new convenience function)

### Background

`AgentConfig` is a plain dataclass in `hpm/config.py`. Add `pattern_type: str = "gaussian"` as the last field. This default ensures all existing code is unaffected.

`make_orchestrator()` in `benchmarks/multi_agent_common.py` creates all agents from a shared `cfg_kwargs` dict. To support per-agent pattern types, it needs a `pattern_types: list[str] | None = None` parameter. When provided, each agent's config is created with the corresponding type.

`evaluate_task()` in `benchmarks/multi_agent_arc.py` calls `make_arc_orchestrator()` directly. Thread `pattern_types` through the call chain.

**Important:** The Agent class (`hpm/agents/agent.py`) creates its initial pattern — check how it does this and ensure it reads `self.config.pattern_type`. You may need to update the Agent's initialisation to use `make_pattern()` from the factory. Read `hpm/agents/agent.py` before implementing this task.

Run tests with: `cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -v 2>&1 | tail -20`

---

- [ ] **Step 1: Write failing tests for agent pattern_type wiring**

Create `tests/test_agent_pattern_type.py`:

```python
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.store.memory import InMemoryStore
from hpm.patterns.laplace import LaplacePattern
from hpm.patterns.gaussian import GaussianPattern


def _make_agent(pattern_type="gaussian"):
    cfg = AgentConfig(agent_id="test", feature_dim=4, pattern_type=pattern_type)
    return Agent(cfg, store=InMemoryStore(), field=None)


def test_agent_creates_gaussian_by_default():
    agent = _make_agent("gaussian")
    records = agent.store.query("test")
    assert len(records) >= 1
    p, _ = records[0]
    assert isinstance(p, GaussianPattern)


def test_agent_creates_laplace_pattern():
    agent = _make_agent("laplace")
    records = agent.store.query("test")
    assert len(records) >= 1
    p, _ = records[0]
    assert isinstance(p, LaplacePattern)


def test_agent_step_with_laplace_no_crash():
    agent = _make_agent("laplace")
    result = agent.step(np.ones(4))
    assert result.get("n_patterns", 0) >= 1
```

- [ ] **Step 2: Run failing test**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/test_agent_pattern_type.py -v
```

Expected: `test_agent_creates_laplace_pattern` and `test_agent_step_with_laplace_no_crash` FAIL (agent still creates GaussianPattern)

- [ ] **Step 3: Add pattern_type to AgentConfig**

In `hpm/config.py`, add as the last field before the closing of the dataclass:

```python
    # Pattern type selection
    pattern_type: str = "gaussian"  # "gaussian" | "laplace"
```

- [ ] **Step 4: Update Agent._seed_if_empty to use factory**

In `hpm/agents/agent.py`, replace the existing import and construction in `_seed_if_empty`:

```python
# At top of file, replace:
#   from ..patterns.gaussian import GaussianPattern
# with:
from ..patterns.gaussian import GaussianPattern  # keep for _share_pending (updated below)
from ..patterns.factory import make_pattern, pattern_from_dict
```

In `_seed_if_empty` (around line 143), replace:
```python
# Old:
init = GaussianPattern(
    mu=rng.normal(0, 1, self.config.feature_dim),
    sigma=np.eye(self.config.feature_dim) * self.config.init_sigma,
)

# New: pass sigma matrix for Gaussian, scalar init_sigma for Laplace
scale = (np.eye(self.config.feature_dim) * self.config.init_sigma
         if self.config.pattern_type == "gaussian"
         else self.config.init_sigma)
init = make_pattern(
    mu=rng.normal(0, 1, self.config.feature_dim),
    scale=scale,
    pattern_type=self.config.pattern_type,
)
```

- [ ] **Step 5: Fix _share_pending to support non-Gaussian patterns**

In `hpm/agents/agent.py`, `_share_pending` (around line 93) hardcodes `GaussianPattern`:

```python
# Old (crashes for LaplacePattern — no .sigma attribute):
shared_copy = GaussianPattern(
    mu=p.mu.copy(),
    sigma=p.sigma.copy(),
    source_id=p.id,
)

# New: type-agnostic clone via to_dict/from_dict
d = p.to_dict()
d['id'] = None          # drop id — fresh UUID on communication (preserves provenance via source_id)
d['source_id'] = p.id
shared_copy = pattern_from_dict(d)
```

Note: `pattern_from_dict` is already imported in Step 4. The `d['id'] = None` approach passes `None` as the id kwarg, which triggers `str(uuid.uuid4())` in the pattern constructor — same behaviour as before.

- [ ] **Step 6: Run agent pattern_type tests**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/test_agent_pattern_type.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 7: Verify existing tests still pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -v 2>&1 | tail -20
```

Expected: All existing tests PASS (pattern_type defaults to "gaussian")

- [ ] **Step 8: Add pattern_types to make_orchestrator()**

In `benchmarks/multi_agent_common.py`, add `pattern_types: list[str] | None = None` parameter to `make_orchestrator()`. Also add import at top:

```python
from hpm.patterns.factory import make_pattern
```

Replace the agents list comprehension:
```python
# Before:
agents = [
    Agent(AgentConfig(agent_id=aid, **cfg_kwargs), store=store, field=field)
    for aid in agent_ids
]

# After:
_pattern_types = pattern_types or ["gaussian"] * len(agent_ids)
agents = [
    Agent(AgentConfig(agent_id=aid, pattern_type=pt, **cfg_kwargs), store=store, field=field)
    for aid, pt in zip(agent_ids, _pattern_types)
]
```

In the `agent_seeds` block, replace the hardcoded `GaussianPattern` construction:
```python
# Old:
#   sigma = np.eye(feature_dim) * init_sigma
#   pattern = GaussianPattern(mu=mu, sigma=sigma)
# New:
scale = (np.eye(feature_dim) * init_sigma
         if agent.config.pattern_type == "gaussian"
         else init_sigma)
pattern = make_pattern(mu, scale, pattern_type=agent.config.pattern_type)
```

- [ ] **Step 9: Add pattern_types to make_arc_orchestrator() and evaluate_task()**

In `benchmarks/multi_agent_arc.py`:

1. Add `pattern_types` parameter to `make_arc_orchestrator()`:

```python
def make_arc_orchestrator(pattern_types=None):
    """Fresh 2-agent HPM orchestrator configured for ARC (per-task reset)."""
    return make_orchestrator(
        n_agents=2,
        feature_dim=FEATURE_DIM,
        agent_ids=["arc_a", "arc_b"],
        with_monitor=False,
        beta_comp=0.0,
        gamma_soc=0.5,
        T_recomb=5,
        recomb_cooldown=3,
        init_sigma=2.0,
        pattern_types=pattern_types or ["gaussian", "gaussian"],
    )
```

2. Add `pattern_types` parameter to `evaluate_task()`:

```python
def evaluate_task(task, all_tasks, task_idx, pattern_types=None):
    orch, agents, store = make_arc_orchestrator(pattern_types=pattern_types)
    # ... rest of function unchanged ...
```

3. Add convenience function after `make_arc_persistent_orchestrator()`:

```python
def make_arc_laplace_orchestrator():
    """Fresh 2-agent HPM orchestrator using LaplacePattern for ARC."""
    return make_arc_orchestrator(pattern_types=["laplace", "laplace"])
```

- [ ] **Step 10: Run full test suite**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -v 2>&1 | tail -30
```

Expected: All tests PASS

- [ ] **Step 11: Commit**

```bash
git add hpm/config.py hpm/agents/agent.py benchmarks/multi_agent_common.py benchmarks/multi_agent_arc.py tests/test_agent_pattern_type.py
git commit -m "feat: wire pattern_type through AgentConfig, Agent, make_orchestrator, and ARC benchmark"
```

---

## Task 5: Integration test

**Files:**
- Create: `tests/integration/test_laplace_arc.py`

### Background

This test runs a minimal end-to-end smoke test with LaplacePattern agents on 10 ARC tasks. It validates that the full stack (LaplacePattern → Agent → TieredStore/InMemoryStore → PatternField → MetaPatternRule → ensemble_score → evaluate_task) works without crashing. It also confirms LaplacePattern produces different NLL scores than GaussianPattern on the same input.

The test loads from the local fixtures rather than the HuggingFace dataset — use synthetic ARC-shaped data to avoid a network dependency. If the test environment has the dataset cached it can use it, otherwise generate synthetic tasks.

Run tests with: `cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/integration/test_laplace_arc.py -v`

---

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_laplace_arc.py`:

```python
"""
Integration smoke test: LaplacePattern agents run end-to-end through the ARC pipeline.

Uses synthetic tasks (no network dependency) to validate the full stack:
LaplacePattern → Agent → InMemoryStore → PatternField → MetaPatternRule → ensemble_score
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.multi_agent_arc import (
    make_arc_orchestrator, ensemble_score, encode_pair, FEATURE_DIM, N_DISTRACTORS
)
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern


def _make_synthetic_task(rng, n_train=3):
    """Generate a synthetic ARC-shaped task with random grids."""
    def rand_grid(size=5):
        return rng.integers(0, 9, size=(size, size)).tolist()

    return {
        "train": [{"input": rand_grid(), "output": rand_grid()} for _ in range(n_train)],
        "test": [{"input": rand_grid(), "output": rand_grid()}],
    }


def _make_synthetic_all_tasks(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [_make_synthetic_task(rng) for _ in range(n)]


@pytest.fixture(scope="module")
def all_tasks():
    return _make_synthetic_all_tasks(n=20)


def test_laplace_arc_no_crash(all_tasks):
    """Laplace agents run evaluate_task without raising exceptions."""
    from benchmarks.multi_agent_arc import evaluate_task
    correct, rank = evaluate_task(all_tasks[0], all_tasks, 0,
                                  pattern_types=["laplace", "laplace"])
    assert isinstance(correct, bool)
    assert 1 <= rank <= N_DISTRACTORS + 1


def test_laplace_mean_rank_in_bounds(all_tasks):
    """Run 10 tasks and confirm mean_rank is in [1, 5].

    Note: the upper bound (5.0) is the maximum possible rank with N_DISTRACTORS=4,
    so this is a pipeline smoke test — it validates no crash and correct rank range,
    not performance.
    """
    from benchmarks.multi_agent_arc import evaluate_task
    ranks = []
    for i in range(10):
        _, rank = evaluate_task(all_tasks[i], all_tasks, i,
                                pattern_types=["laplace", "laplace"])
        ranks.append(rank)
    mean_rank = np.mean(ranks)
    assert 1.0 <= mean_rank <= 5.0


def test_laplace_and_gaussian_produce_different_nll():
    """LaplacePattern and GaussianPattern produce different NLL for the same input."""
    rng = np.random.default_rng(42)
    mu = rng.standard_normal(FEATURE_DIM)
    x = rng.standard_normal(FEATURE_DIM)

    gauss = GaussianPattern(mu, np.eye(FEATURE_DIM))
    laplace = LaplacePattern(mu, np.ones(FEATURE_DIM))

    nll_g = gauss.log_prob(x)
    nll_l = laplace.log_prob(x)
    assert nll_g != nll_l, "Gaussian and Laplace should produce different NLL values"


def test_gaussian_orchestrator_still_works(all_tasks):
    """Confirm the default Gaussian orchestrator is unaffected by this change."""
    from benchmarks.multi_agent_arc import evaluate_task
    correct, rank = evaluate_task(all_tasks[0], all_tasks, 0)
    assert 1 <= rank <= N_DISTRACTORS + 1
```

- [ ] **Step 2: Run test to verify it fails (before Task 4 is complete)**

If running after Task 4 is complete, skip this step.

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/integration/test_laplace_arc.py -v
```

Expected: PASS after Task 4 is complete (all tasks implemented)

- [ ] **Step 3: Run full test suite**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -v 2>&1 | tail -30
```

Expected: All tests PASS including the 4 new integration tests

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_laplace_arc.py
git commit -m "test: add LaplacePattern ARC integration smoke test"
```

---

## Final Verification

After all tasks complete:

- [ ] **Run complete test suite**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -m pytest tests/ -v 2>&1 | tail -40
```

Expected: All tests PASS

- [ ] **Quick benchmark smoke test** (optional, validates end-to-end performance)

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python3 -c "
from benchmarks.multi_agent_arc import make_arc_laplace_orchestrator
orch, agents, store = make_arc_laplace_orchestrator()
print('Laplace orchestrator created OK')
for a in agents:
    print(f'  {a.agent_id}: pattern_type={a.config.pattern_type}')
"
```

Expected: No exceptions, both agents show `pattern_type=laplace`
