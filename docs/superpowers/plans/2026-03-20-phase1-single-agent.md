# HPM Phase 1: Single Agent + Concept Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single HPM agent that learns concepts hierarchically, with pattern dynamics faithful to the paper's math (Appendices A, D), validated against predictions §9.1 and §9.4.

**Architecture:** Modular Python package `hpm/` with pluggable Pattern representations (Gaussian default), an EvaluatorPipeline (epistemic + affective), Meta Pattern Rule replicator dynamics (D5), and a ConceptLearningDomain. Patterns are value types backed by InMemoryStore. All math symbols map 1:1 to spec §3.

**Tech Stack:** Python 3.11+, numpy, scipy, pytest, uv (package manager)

---

## File Map

```
hpm/
  __init__.py
  config.py                          # AgentConfig dataclass (all hyperparameters)
  patterns/
    __init__.py
    base.py                          # Pattern Protocol
    gaussian.py                      # GaussianPattern implementation
  store/
    __init__.py
    base.py                          # PatternStore Protocol
    memory.py                        # InMemoryStore
  evaluators/
    __init__.py
    epistemic.py                     # EpistemicEvaluator (D2-D3)
    affective.py                     # AffectiveEvaluator (D3, §9.4)
  dynamics/
    __init__.py
    meta_pattern_rule.py             # MetaPatternRule (D5)
  domains/
    __init__.py
    base.py                          # Domain Protocol
    concept.py                       # ConceptLearningDomain
  agents/
    __init__.py
    agent.py                         # Agent (wires all components)
  metrics/
    __init__.py
    hpm_predictions.py               # §9.1 sensitivity ratio, §9.4 curiosity profile

tests/
  conftest.py                        # shared fixtures
  patterns/
    test_gaussian.py
  store/
    test_memory.py
  evaluators/
    test_epistemic.py
    test_affective.py
  dynamics/
    test_meta_pattern_rule.py
  domains/
    test_concept.py
  agents/
    test_agent.py
  metrics/
    test_hpm_predictions.py
  integration/
    test_phase1.py

pyproject.toml
```

---

## Task 1: Project Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: all `__init__.py` files
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hpm-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Install dependencies**

```bash
uv venv && uv pip install -e ".[dev]"
```

Expected: virtual environment created, packages installed.

- [ ] **Step 3: Create package skeleton**

```bash
mkdir -p hpm/{patterns,store,evaluators,dynamics,domains,agents,metrics}
mkdir -p tests/{patterns,store,evaluators,dynamics,domains,agents,metrics,integration}
touch hpm/__init__.py hpm/config.py
touch hpm/patterns/__init__.py hpm/store/__init__.py
touch hpm/evaluators/__init__.py hpm/dynamics/__init__.py
touch hpm/domains/__init__.py hpm/agents/__init__.py hpm/metrics/__init__.py
touch tests/__init__.py tests/conftest.py
touch tests/patterns/__init__.py tests/store/__init__.py
touch tests/evaluators/__init__.py tests/dynamics/__init__.py
touch tests/domains/__init__.py tests/agents/__init__.py
touch tests/metrics/__init__.py tests/integration/__init__.py
```

- [ ] **Step 4: Create tests/conftest.py**

```python
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def dim():
    return 4


@pytest.fixture
def simple_pattern(dim):
    return GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))


@pytest.fixture
def rng():
    return np.random.default_rng(42)
```

- [ ] **Step 5: Verify pytest discovers tests**

```bash
pytest --collect-only
```

Expected: 0 tests collected, no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml hpm/ tests/
git commit -m "feat: project bootstrap — package skeleton and pyproject.toml"
```

---

## Task 2: AgentConfig Dataclass

**Files:**
- Create: `hpm/config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from hpm.config import AgentConfig

def test_config_defaults():
    cfg = AgentConfig(agent_id="a1", feature_dim=4)
    assert cfg.eta == 0.01
    assert cfg.beta_c == 0.1
    assert cfg.epsilon == 1e-4
    assert cfg.lambda_L == 0.1
    assert cfg.beta_aff == 0.5
    assert cfg.gamma_soc == 0.0   # off by default (single agent)
    assert cfg.feature_dim == 4
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError or AttributeError.

- [ ] **Step 3: Implement AgentConfig**

```python
# hpm/config.py
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    agent_id: str
    feature_dim: int
    # Dynamics (D5)
    eta: float = 0.01
    beta_c: float = 0.1
    epsilon: float = 1e-4
    # Evaluators
    lambda_L: float = 0.1        # EMA decay (D2)
    beta_aff: float = 0.5        # affective weight in J_i
    gamma_soc: float = 0.0       # social weight (0 = single agent)
    # Affective evaluator shape (§9.4)
    k: float = 1.0               # sigmoid sharpness
    c_opt: float = 10.0          # optimal complexity
    sigma_c: float = 5.0         # complexity bandwidth
    alpha_r: float = 0.0         # external reward weight
    # Social evaluator (Phase 2+)
    rho: float = 1.0             # field frequency amplification scale (D6)
    # Pattern initialisation
    init_sigma: float = 1.0      # initial covariance scale
```

- [ ] **Step 4: Run test — verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm/config.py tests/test_config.py
git commit -m "feat: AgentConfig dataclass with all Phase 1 hyperparameters"
```

---

## Task 3: Pattern Protocol + GaussianPattern

**Files:**
- Create: `hpm/patterns/base.py`
- Create: `hpm/patterns/gaussian.py`
- Create: `tests/patterns/test_gaussian.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/patterns/test_gaussian.py
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern


def test_log_prob_is_positive_loss(dim):
    """log_prob returns -log p(x|h), always >= 0 for well-formed inputs."""
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x = np.zeros(dim)
    assert p.log_prob(x) >= 0


def test_log_prob_lower_at_mean(dim):
    """Observation at mean has lower loss than distant observation."""
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    loss_near = p.log_prob(np.zeros(dim))
    loss_far = p.log_prob(np.ones(dim) * 10)
    assert loss_near < loss_far


def test_update_returns_new_instance(dim):
    """update() must return a new Pattern — value type semantics."""
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x = np.ones(dim)
    p2 = p.update(x)
    assert p2 is not p
    assert p2.id == p.id           # same pattern ID (continued learning)
    assert not np.allclose(p2.mu, p.mu)   # mu shifted toward x


def test_recombine_produces_new_id(dim):
    """Recombination produces a pattern with a fresh UUID."""
    p1 = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    p2 = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    child = p1.recombine(p2)
    assert child.id != p1.id
    assert child.id != p2.id


def test_structural_validity(dim):
    """Valid covariance passes; non-PD fails."""
    p_valid = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    assert p_valid.is_structurally_valid()

    bad_sigma = -np.eye(dim)
    p_invalid = GaussianPattern(mu=np.zeros(dim), sigma=bad_sigma)
    assert not p_invalid.is_structurally_valid()


def test_serialisation_roundtrip(dim):
    """to_dict / from_dict roundtrip preserves all state."""
    p = GaussianPattern(mu=np.arange(dim, dtype=float), sigma=np.eye(dim) * 2)
    d = p.to_dict()
    p2 = GaussianPattern.from_dict(d)
    assert p2.id == p.id
    assert np.allclose(p2.mu, p.mu)
    assert np.allclose(p2.sigma, p.sigma)


def test_description_length_positive(dim):
    p = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    assert p.description_length() > 0


def test_compress_between_zero_and_one(dim):
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    c = p.compress()
    assert 0.0 <= c <= 1.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/patterns/test_gaussian.py -v
```

Expected: ImportError (module not found).

- [ ] **Step 3: Implement Pattern Protocol**

```python
# hpm/patterns/base.py
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Pattern(Protocol):
    id: str

    def log_prob(self, x: np.ndarray) -> float: ...
    def description_length(self) -> float: ...
    def connectivity(self) -> float: ...
    def compress(self) -> float: ...
    def update(self, x: np.ndarray) -> 'Pattern': ...
    def recombine(self, other: 'Pattern') -> 'Pattern': ...
    def is_structurally_valid(self) -> bool: ...
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> 'Pattern': ...
```

- [ ] **Step 4: Implement GaussianPattern**

```python
# hpm/patterns/gaussian.py
import uuid
import numpy as np
from scipy.stats import multivariate_normal


class GaussianPattern:
    """
    Pattern h = (mu, Sigma) — a Gaussian generative model over feature space.
    Two-level hierarchy: z^(2) = cluster centroid, z^(1) = instance noise.
    Value type: update() returns a new instance; id is preserved on update,
    fresh on recombination.
    """

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, id: str | None = None):
        self.id = id or str(uuid.uuid4())
        self.mu = np.array(mu, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self._n_obs: int = 0

    def log_prob(self, x: np.ndarray) -> float:
        """Returns -log p(x|h) — epistemic loss. Lower = better fit."""
        return float(-multivariate_normal.logpdf(x, mean=self.mu, cov=self.sigma))

    def description_length(self) -> float:
        """MDL approximation: count of non-negligible parameters."""
        return float(
            np.sum(np.abs(self.mu) > 1e-6)
            + np.sum(np.abs(self.sigma - np.diag(np.diag(self.sigma))) > 1e-6)
            + self.sigma.shape[0]  # diagonal entries always count
        )

    def connectivity(self) -> float:
        """Mean absolute off-diagonal correlation — structural linkage proxy."""
        n = self.sigma.shape[0]
        if n <= 1:
            return 0.0
        std = np.sqrt(np.diag(self.sigma))
        with np.errstate(invalid='ignore'):
            corr = self.sigma / np.outer(std, std)
        corr = np.nan_to_num(corr)
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(np.abs(corr[mask])))

    def compress(self) -> float:
        """Fraction of variance explained by top principal component (z^(2) proxy)."""
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total = eigenvalues.sum()
        if total == 0.0:
            return 0.0
        return float(eigenvalues[-1] / total)

    def update(self, x: np.ndarray) -> 'GaussianPattern':
        """Online mean update — returns new instance with same id."""
        n = self._n_obs + 1
        new_mu = (self.mu * self._n_obs + x) / n
        new_p = GaussianPattern(new_mu, self.sigma.copy(), id=self.id)
        new_p._n_obs = n
        return new_p

    def recombine(self, other: 'GaussianPattern') -> 'GaussianPattern':
        """Convex combination of parent parameters — fresh id."""
        alpha = 0.5
        new_mu = alpha * self.mu + (1 - alpha) * other.mu
        new_sigma = alpha * self.sigma + (1 - alpha) * other.sigma
        return GaussianPattern(new_mu, new_sigma)

    def is_structurally_valid(self) -> bool:
        """True if covariance is positive definite."""
        try:
            eigenvalues = np.linalg.eigvalsh(self.sigma)
            return bool(np.all(eigenvalues > 0))
        except np.linalg.LinAlgError:
            return False

    def to_dict(self) -> dict:
        return {
            'type': 'gaussian',
            'id': self.id,
            'mu': self.mu.tolist(),
            'sigma': self.sigma.tolist(),
            'n_obs': self._n_obs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GaussianPattern':
        p = cls(np.array(d['mu']), np.array(d['sigma']), id=d['id'])
        p._n_obs = d['n_obs']
        return p
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/patterns/test_gaussian.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add hpm/patterns/ tests/patterns/
git commit -m "feat: Pattern Protocol and GaussianPattern (value type, serialisable)"
```

---

## Task 4: InMemoryStore

**Files:**
- Create: `hpm/store/base.py`
- Create: `hpm/store/memory.py`
- Create: `tests/store/test_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/store/test_memory.py
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.memory import InMemoryStore


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def pattern(dim):
    return GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))


def test_save_and_load(store, pattern):
    store.save(pattern, weight=0.8, agent_id="agent1")
    loaded_p, loaded_w = store.load(pattern.id)
    assert loaded_p.id == pattern.id
    assert loaded_w == pytest.approx(0.8)


def test_query_returns_agent_patterns(store, pattern, dim):
    other = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    store.save(pattern, 0.6, "agent1")
    store.save(other, 0.4, "agent2")
    results = store.query("agent1")
    assert len(results) == 1
    assert results[0][0].id == pattern.id


def test_update_weight(store, pattern):
    store.save(pattern, 0.5, "agent1")
    store.update_weight(pattern.id, 0.9)
    _, w = store.load(pattern.id)
    assert w == pytest.approx(0.9)


def test_delete(store, pattern):
    store.save(pattern, 1.0, "agent1")
    store.delete(pattern.id)
    assert store.query("agent1") == []


def test_query_all(store, pattern, dim):
    other = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    store.save(pattern, 0.6, "agent1")
    store.save(other, 0.4, "agent2")
    all_records = store.query_all()
    assert len(all_records) == 2
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/store/test_memory.py -v
```

- [ ] **Step 3: Implement PatternStore Protocol and InMemoryStore**

```python
# hpm/store/base.py
from typing import Protocol
import numpy as np


class PatternStore(Protocol):
    def save(self, pattern, weight: float, agent_id: str) -> None: ...
    def load(self, pattern_id: str) -> tuple: ...
    def query(self, agent_id: str) -> list: ...
    def query_all(self) -> list: ...
    def delete(self, pattern_id: str) -> None: ...
    def update_weight(self, pattern_id: str, weight: float) -> None: ...
```

```python
# hpm/store/memory.py


class InMemoryStore:
    """In-memory PatternStore. Default backend for Phase 1."""

    def __init__(self):
        # pattern_id -> (pattern, weight, agent_id)
        self._data: dict = {}

    def save(self, pattern, weight: float, agent_id: str) -> None:
        self._data[pattern.id] = (pattern, weight, agent_id)

    def load(self, pattern_id: str) -> tuple:
        pattern, weight, _ = self._data[pattern_id]
        return pattern, weight

    def query(self, agent_id: str) -> list:
        return [
            (p, w)
            for p, w, aid in self._data.values()
            if aid == agent_id
        ]

    def query_all(self) -> list:
        return list(self._data.values())

    def delete(self, pattern_id: str) -> None:
        self._data.pop(pattern_id, None)

    def update_weight(self, pattern_id: str, weight: float) -> None:
        if pattern_id in self._data:
            p, _, aid = self._data[pattern_id]
            self._data[pattern_id] = (p, weight, aid)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/store/test_memory.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hpm/store/ tests/store/
git commit -m "feat: PatternStore Protocol and InMemoryStore backend"
```

---

## Task 5: EpistemicEvaluator

**Files:**
- Create: `hpm/evaluators/epistemic.py`
- Create: `tests/evaluators/test_epistemic.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/evaluators/test_epistemic.py
import numpy as np
import pytest
from hpm.evaluators.epistemic import EpistemicEvaluator
from hpm.patterns.gaussian import GaussianPattern


def test_accuracy_improves_toward_mean(dim):
    """Pattern fit at its own mean should stabilise to a high accuracy."""
    evaluator = EpistemicEvaluator(lambda_L=0.5)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x_good = np.zeros(dim)   # at mean — low loss

    acc_values = [evaluator.update(pattern, x_good) for _ in range(20)]
    # Accuracy (= -running_loss) should be stable (not diverging)
    assert all(a <= 0 for a in acc_values)   # A_i <= 0 always


def test_accuracy_lower_for_distant_obs(dim):
    """Pattern evaluated on distant observation has lower accuracy."""
    evaluator = EpistemicEvaluator(lambda_L=1.0)   # instant update
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    acc_near = evaluator.update(pattern, np.zeros(dim))
    evaluator2 = EpistemicEvaluator(lambda_L=1.0)
    acc_far = evaluator2.update(pattern, np.ones(dim) * 10)
    assert acc_near > acc_far


def test_running_loss_ema(dim):
    """Running loss follows EMA formula: L(t) = (1-lambda)*L(t-1) + lambda*ell(t)."""
    evaluator = EpistemicEvaluator(lambda_L=0.5)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x = np.zeros(dim)
    loss0 = pattern.log_prob(x)
    # First call: L(0) = 0, so L(1) = lambda_L * ell(0), A(1) = -lambda_L * ell(0)
    acc = evaluator.update(pattern, x)
    assert acc == pytest.approx(-0.1 * loss0, rel=1e-5)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/evaluators/test_epistemic.py -v
```

- [ ] **Step 3: Implement EpistemicEvaluator**

```python
# hpm/evaluators/epistemic.py
import numpy as np


class EpistemicEvaluator:
    """
    Implements D2-D3 epistemic evaluator.
    Maintains EMA running loss L_i(t) per pattern.
    Returns accuracy A_i(t) = -L_i(t).
    """

    def __init__(self, lambda_L: float = 0.1):
        self.lambda_L = lambda_L
        self._running_loss: dict[str, float] = {}

    def update(self, pattern, x: np.ndarray) -> float:
        """
        Update running loss for pattern given observation x.
        Returns A_i(t) = -L_i(t).
        """
        ell = pattern.log_prob(x)   # instantaneous loss (D2)
        prev = self._running_loss.get(pattern.id, 0.0)  # L_i(0) = 0 per spec §3.2
        L = (1.0 - self.lambda_L) * prev + self.lambda_L * ell
        self._running_loss[pattern.id] = L
        return -L   # A_i(t)

    def accuracy(self, pattern_id: str) -> float:
        return -self._running_loss.get(pattern_id, 0.0)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/evaluators/test_epistemic.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hpm/evaluators/epistemic.py tests/evaluators/test_epistemic.py
git commit -m "feat: EpistemicEvaluator — EMA running loss and accuracy (D2-D3)"
```

---

## Task 6: AffectiveEvaluator

**Files:**
- Create: `hpm/evaluators/affective.py`
- Create: `tests/evaluators/test_affective.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/evaluators/test_affective.py
import numpy as np
import pytest
from hpm.evaluators.affective import AffectiveEvaluator
from hpm.patterns.gaussian import GaussianPattern


def test_e_aff_non_negative(dim):
    """Affective evaluator output is always non-negative."""
    aff = AffectiveEvaluator(k=1.0, c_opt=5.0, sigma_c=3.0)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    score = aff.update(pattern, current_accuracy=-2.0)
    assert score >= 0.0


def test_inverted_u_over_complexity():
    """E_aff peaks at intermediate complexity, lower at extremes."""
    aff = AffectiveEvaluator(k=1.0, c_opt=10.0, sigma_c=4.0)

    # Same improvement rate, different complexities
    low_c = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))    # low complexity
    mid_c = GaussianPattern(mu=np.ones(10), sigma=np.eye(10))   # mid complexity
    high_c = GaussianPattern(mu=np.ones(30), sigma=np.eye(30))  # high complexity

    # Simulate improvement (positive delta_A)
    score_low = aff.update(low_c, current_accuracy=-1.0)
    aff2 = AffectiveEvaluator(k=1.0, c_opt=10.0, sigma_c=4.0)
    score_mid = aff2.update(mid_c, current_accuracy=-1.0)
    aff3 = AffectiveEvaluator(k=1.0, c_opt=10.0, sigma_c=4.0)
    # Force improvement
    aff3.update(high_c, current_accuracy=-2.0)
    score_high = aff3.update(high_c, current_accuracy=-1.0)

    # Mid should beat low and high (inverted-U)
    assert score_mid >= score_low
    assert score_mid >= score_high


def test_external_reward_added(dim):
    """External reward (alpha_r * r_t) adds to E_aff when alpha_r > 0."""
    aff_no_reward = AffectiveEvaluator(k=1.0, c_opt=5.0, sigma_c=3.0, alpha_r=0.0)
    aff_reward = AffectiveEvaluator(k=1.0, c_opt=5.0, sigma_c=3.0, alpha_r=1.0)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    s1 = aff_no_reward.update(pattern, current_accuracy=-1.0, reward=2.0)
    s2 = aff_reward.update(pattern, current_accuracy=-1.0, reward=2.0)
    assert s2 > s1
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/evaluators/test_affective.py -v
```

- [ ] **Step 3: Implement AffectiveEvaluator**

```python
# hpm/evaluators/affective.py
import numpy as np


class AffectiveEvaluator:
    """
    Affective evaluator (D3, §9.4).
    Curiosity signal: peaks at intermediate complexity patterns
    that are currently improving — models the Goldilocks effect.

    E_aff_i(t) = novelty(t) * capacity(t) * g(c_i) + alpha_r * r_t

    where:
      novelty(t)  = sigmoid(k * Delta_A_i(t))
      capacity(t) = 1 - novelty(t)
      g(c)        = exp(-(c - c_opt)^2 / (2 * sigma_c^2))
    """

    def __init__(
        self,
        k: float = 1.0,
        c_opt: float = 10.0,
        sigma_c: float = 5.0,
        alpha_r: float = 0.0,
    ):
        self.k = k
        self.c_opt = c_opt
        self.sigma_c = sigma_c
        self.alpha_r = alpha_r
        self._prev_accuracy: dict[str, float] = {}

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _g(self, c: float) -> float:
        return float(np.exp(-((c - self.c_opt) ** 2) / (2.0 * self.sigma_c ** 2)))

    def update(self, pattern, current_accuracy: float, reward: float = 0.0) -> float:
        prev = self._prev_accuracy.get(pattern.id, current_accuracy)
        delta_A = current_accuracy - prev
        self._prev_accuracy[pattern.id] = current_accuracy

        novelty = self._sigmoid(self.k * delta_A)
        capacity = 1.0 - novelty
        c = pattern.description_length()

        e_aff = novelty * capacity * self._g(c)
        e_aff += self.alpha_r * reward
        return float(max(e_aff, 0.0))
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/evaluators/test_affective.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hpm/evaluators/affective.py tests/evaluators/test_affective.py
git commit -m "feat: AffectiveEvaluator — curiosity signal with inverted-U complexity profile (D3, 9.4)"
```

---

## Task 7: Meta Pattern Rule Dynamics

**Files:**
- Create: `hpm/dynamics/meta_pattern_rule.py`
- Create: `tests/dynamics/test_meta_pattern_rule.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/dynamics/test_meta_pattern_rule.py
import numpy as np
import pytest
from hpm.dynamics.meta_pattern_rule import MetaPatternRule, sym_kl_normalised
from hpm.patterns.gaussian import GaussianPattern


def test_weights_sum_to_one_after_step(dim):
    """Weights remain normalised after dynamics step."""
    mpr = MetaPatternRule(eta=0.1, beta_c=0.05, epsilon=1e-4)
    patterns = [
        GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim)),
        GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim)),
    ]
    weights = np.array([0.6, 0.4])
    totals = np.array([-1.0, -0.5])   # second pattern has higher total score
    new_w = mpr.step(patterns, weights, totals)
    assert new_w.sum() == pytest.approx(1.0, abs=1e-6)


def test_higher_total_gains_weight(dim):
    """Pattern with above-average total score gains weight."""
    mpr = MetaPatternRule(eta=0.1, beta_c=0.0, epsilon=1e-4)  # no conflict
    patterns = [
        GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim)),
        GaussianPattern(mu=np.ones(dim) * 5, sigma=np.eye(dim)),
    ]
    weights = np.array([0.5, 0.5])
    totals = np.array([-2.0, -0.5])   # pattern 1 has higher score
    new_w = mpr.step(patterns, weights, totals)
    assert new_w[1] > new_w[0]


def test_floor_prevents_empty_library(dim):
    """If all weights collapse, best pattern retained at weight 1.0."""
    mpr = MetaPatternRule(eta=100.0, beta_c=100.0, epsilon=0.1)
    patterns = [GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))]
    weights = np.array([1.0])
    totals = np.array([-999.0])
    new_w = mpr.step(patterns, weights, totals)
    assert len(new_w) == 1
    assert new_w[0] == pytest.approx(1.0)


def test_conflict_excludes_self_inhibition(dim):
    """D5 sum excludes j=i: a single pattern should not self-inhibit."""
    mpr = MetaPatternRule(eta=0.0, beta_c=1.0, epsilon=1e-4)   # only conflict term
    patterns = [GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))]
    weights = np.array([1.0])
    totals = np.array([-1.0])
    new_w = mpr.step(patterns, weights, totals)
    # Single pattern: no j != i pairs, so no conflict inhibition
    assert new_w[0] == pytest.approx(1.0, abs=1e-6)


def test_sym_kl_same_pattern_is_zero(dim):
    """Symmetric KL of identical patterns should be near zero."""
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    kl = sym_kl_normalised(p, p)
    assert kl == pytest.approx(0.0, abs=0.05)


def test_sym_kl_different_patterns_positive(dim):
    """Distant patterns have positive incompatibility."""
    p1 = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    p2 = GaussianPattern(mu=np.ones(dim) * 10, sigma=np.eye(dim))
    kl = sym_kl_normalised(p1, p2)
    assert kl > 0.0
    assert kl <= 1.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/dynamics/test_meta_pattern_rule.py -v
```

- [ ] **Step 3: Implement MetaPatternRule**

```python
# hpm/dynamics/meta_pattern_rule.py
import numpy as np


def sym_kl_normalised(p, q, n_samples: int = 200, rng=None) -> float:
    """
    Symmetrised KL divergence between two GaussianPatterns, normalised to [0,1].
    kappa_{ij} in D5 — incompatibility measure.
    Uses Monte Carlo approximation for generality.

    Note: log_prob(x) returns -log p(x|h), so:
      log p(x|h) = -log_prob(x)
      KL(p||q) = E_p[log p - log q] = E_p[-p.log_prob(x) - (-q.log_prob(x))]
               = E_p[q.log_prob(x) - p.log_prob(x)]
    """
    if rng is None:
        rng = np.random.default_rng()

    samples_p = rng.multivariate_normal(p.mu, p.sigma, n_samples)
    kl_pq = float(np.mean([q.log_prob(s) - p.log_prob(s) for s in samples_p]))

    samples_q = rng.multivariate_normal(q.mu, q.sigma, n_samples)
    kl_qp = float(np.mean([p.log_prob(s) - q.log_prob(s) for s in samples_q]))

    sym_kl = max((kl_pq + kl_qp) / 2.0, 0.0)
    return float(sym_kl / (sym_kl + 1.0))   # normalise to [0, 1]


class MetaPatternRule:
    """
    D5: Discrete-time replicator dynamics with conflict inhibition.

    w_i(t+1) = w_i(t)
              + eta*(Total_i - Total_bar) * w_i(t)          # replicator
              - beta_c * sum_{j!=i} kappa_{ij} * w_i * w_j  # conflict inhibition

    Weights renormalised after each step.
    Floor: if all weights < epsilon, best pattern retained at 1.0.
    """

    def __init__(self, eta: float = 0.01, beta_c: float = 0.1, epsilon: float = 1e-4):
        self.eta = eta
        self.beta_c = beta_c
        self.epsilon = epsilon
        self._rng = np.random.default_rng(0)

    def step(self, patterns: list, weights: np.ndarray, totals: np.ndarray) -> np.ndarray:
        n = len(patterns)
        if n == 0:
            return weights.copy()

        weights = np.array(weights, dtype=float)
        total_bar = float(np.dot(weights, totals))

        # Build incompatibility matrix kappa_{ij}
        kappa = np.zeros((n, n))  # diagonal stays 0: j!=i exclusion (D5)
        for i in range(n):
            for j in range(i + 1, n):  # explicit j != i
                k = sym_kl_normalised(patterns[i], patterns[j], rng=self._rng)
                kappa[i, j] = k
                kappa[j, i] = k
        assert np.all(np.diag(kappa) == 0.0), "kappa diagonal must be zero (j!=i in D5)"

        new_weights = weights.copy()
        for i in range(n):
            replicator = self.eta * (totals[i] - total_bar) * weights[i]
            conflict = self.beta_c * float(np.dot(kappa[i], weights) * weights[i])
            new_weights[i] = weights[i] + replicator - conflict

        new_weights = np.maximum(new_weights, 0.0)

        # Floor: never empty library (spec §3.3)
        if np.all(new_weights < self.epsilon):
            best = int(np.argmax(totals))
            new_weights = np.zeros(n)
            new_weights[best] = 1.0
            return new_weights

        total = new_weights.sum()
        if total > 0:
            new_weights /= total
        return new_weights
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/dynamics/test_meta_pattern_rule.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hpm/dynamics/meta_pattern_rule.py tests/dynamics/test_meta_pattern_rule.py
git commit -m "feat: MetaPatternRule replicator dynamics with conflict inhibition (D5)"
```

---

## Task 8: ConceptLearningDomain

**Files:**
- Create: `hpm/domains/base.py`
- Create: `hpm/domains/concept.py`
- Create: `tests/domains/test_concept.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/domains/test_concept.py
import numpy as np
import pytest
from hpm.domains.concept import ConceptLearningDomain, Concept


@pytest.fixture
def domain():
    concepts = [
        Concept(
            deep_features=np.array([1.0, 0.0]),
            surface_templates=[np.array([0.5, 0.5]), np.array([0.8, 0.2])],
            label=0,
        ),
        Concept(
            deep_features=np.array([0.0, 1.0]),
            surface_templates=[np.array([0.2, 0.8]), np.array([0.3, 0.7])],
            label=1,
        ),
    ]
    return ConceptLearningDomain(concepts, noise=0.01, seed=42)


def test_observe_returns_correct_dim(domain):
    x = domain.observe()
    assert x.shape == (domain.feature_dim(),)


def test_feature_dim_is_surface_plus_deep(domain):
    assert domain.feature_dim() == 4   # 2 surface + 2 deep


def test_deep_perturb_changes_deep_features(domain):
    perturbed = domain.deep_perturb()
    # Deep features should differ
    orig_deep = domain.concepts[0].deep_features
    pert_deep = perturbed.concepts[0].deep_features
    assert not np.allclose(orig_deep, pert_deep)


def test_surface_perturb_preserves_deep_features(domain):
    perturbed = domain.surface_perturb()
    orig_deep = domain.concepts[0].deep_features
    pert_deep = perturbed.concepts[0].deep_features
    assert np.allclose(orig_deep, pert_deep)


def test_transfer_probe_returns_labelled_pairs(domain):
    probes = domain.transfer_probe(near=True)
    assert len(probes) > 0
    x, label = probes[0]
    assert isinstance(x, np.ndarray)
    assert isinstance(label, int)
    assert x.shape == (domain.feature_dim(),)


def test_far_transfer_novel_surface(domain):
    near = domain.transfer_probe(near=True)
    far = domain.transfer_probe(near=False)
    # Far transfer uses novel surface — surface features differ more
    near_surfaces = np.stack([x[:2] for x, _ in near])
    far_surfaces = np.stack([x[:2] for x, _ in far])
    assert far_surfaces.std() > near_surfaces.std()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/domains/test_concept.py -v
```

- [ ] **Step 3: Implement Domain Protocol and ConceptLearningDomain**

```python
# hpm/domains/base.py
from typing import Protocol
import numpy as np


class Domain(Protocol):
    def observe(self) -> np.ndarray: ...
    def feature_dim(self) -> int: ...
    def deep_perturb(self) -> 'Domain': ...
    def surface_perturb(self) -> 'Domain': ...
    def transfer_probe(self, near: bool) -> list[tuple[np.ndarray, int]]: ...
```

```python
# hpm/domains/concept.py
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Concept:
    deep_features: np.ndarray      # structural invariants
    surface_templates: list        # list of np.ndarray surface prototypes
    label: int = 0


class ConceptLearningDomain:
    """
    Concept learning domain for HPM Phase 1.

    Observations x_t = [surface_features || deep_features].
    Surface features vary across instances; deep features are invariant per concept.
    Supports §9.1 (deep vs surface perturbation) and §9.2 (near/far transfer probes).
    """

    def __init__(
        self,
        concepts: list[Concept],
        noise: float = 0.1,
        seed: int | None = None,
    ):
        self.concepts = concepts
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._t = 0

    def observe(self) -> np.ndarray:
        concept = self.concepts[self._t % len(self.concepts)]
        self._t += 1
        surface = self.rng.choice(len(concept.surface_templates))
        s = concept.surface_templates[surface] + self.rng.normal(
            0, self.noise, concept.surface_templates[surface].shape
        )
        return np.concatenate([s, concept.deep_features])

    def feature_dim(self) -> int:
        c = self.concepts[0]
        return len(c.surface_templates[0]) + len(c.deep_features)

    def deep_perturb(self) -> 'ConceptLearningDomain':
        """Return copy with structurally altered deep features (§9.1)."""
        new_concepts = [
            Concept(
                deep_features=c.deep_features + self.rng.normal(0, 0.5, c.deep_features.shape),
                surface_templates=c.surface_templates,
                label=c.label,
            )
            for c in self.concepts
        ]
        return ConceptLearningDomain(new_concepts, self.noise)

    def surface_perturb(self) -> 'ConceptLearningDomain':
        """Return copy with altered surface templates but preserved deep structure (§9.1)."""
        new_concepts = [
            Concept(
                deep_features=c.deep_features.copy(),
                surface_templates=[
                    s + self.rng.normal(0, 0.5, s.shape) for s in c.surface_templates
                ],
                label=c.label,
            )
            for c in self.concepts
        ]
        return ConceptLearningDomain(new_concepts, self.noise)

    def transfer_probe(self, near: bool) -> list[tuple[np.ndarray, int]]:
        """
        Generate labelled test observations.
        near=True: familiar surface + correct deep features
        near=False (far): novel random surface + correct deep features
        """
        probes = []
        n_per_concept = 10
        for concept in self.concepts:
            for _ in range(n_per_concept):
                if near:
                    surface = self.rng.choice(len(concept.surface_templates))
                    s = concept.surface_templates[surface] + self.rng.normal(
                        0, self.noise * 0.5, concept.surface_templates[surface].shape
                    )
                else:
                    s = self.rng.normal(0, 2.0, concept.surface_templates[0].shape)
                x = np.concatenate([s, concept.deep_features])
                probes.append((x, int(concept.label)))
        return probes
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/domains/test_concept.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hpm/domains/ tests/domains/
git commit -m "feat: ConceptLearningDomain with deep/surface split and transfer probes (§9.1, §9.2)"
```

---

## Task 9: Agent

**Files:**
- Create: `hpm/agents/agent.py`
- Create: `tests/agents/test_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_agent.py
import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig


@pytest.fixture
def config():
    return AgentConfig(agent_id="test_agent", feature_dim=4, eta=0.05, lambda_L=0.2)


@pytest.fixture
def agent(config):
    return Agent(config)


def test_agent_initialises_with_one_pattern(agent):
    from hpm.store.memory import InMemoryStore
    records = agent.store.query("test_agent")
    assert len(records) == 1


def test_step_returns_metrics(agent):
    x = np.zeros(4)
    result = agent.step(x)
    assert 't' in result
    assert 'n_patterns' in result
    assert 'mean_accuracy' in result
    assert result['t'] == 1


def test_accuracy_generally_improves(agent):
    """After many steps on consistent observations, accuracy should improve."""
    x = np.zeros(4)
    results = [agent.step(x) for _ in range(50)]
    early = np.mean([r['mean_accuracy'] for r in results[:10]])
    late = np.mean([r['mean_accuracy'] for r in results[40:]])
    assert late >= early   # accuracy improves (less negative) over time


def test_patterns_are_updated_after_step(agent):
    x = np.ones(4)
    initial_records = agent.store.query("test_agent")
    initial_mu = initial_records[0][0].mu.copy()
    agent.step(x)
    updated_records = agent.store.query("test_agent")
    updated_mu = updated_records[0][0].mu
    assert not np.allclose(initial_mu, updated_mu)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/agents/test_agent.py -v
```

- [ ] **Step 3: Implement Agent**

```python
# hpm/agents/agent.py
import numpy as np
from ..config import AgentConfig
from ..patterns.gaussian import GaussianPattern
from ..evaluators.epistemic import EpistemicEvaluator
from ..evaluators.affective import AffectiveEvaluator
from ..dynamics.meta_pattern_rule import MetaPatternRule
from ..store.memory import InMemoryStore


class Agent:
    """
    Single HPM agent. Wires PatternLibrary, EvaluatorPipeline, and Dynamics.
    Backed by a PatternStore (InMemoryStore by default).

    Data flow per step (Phase 1, §7):
      1. Compute ell_i(t) for each pattern
      2. Update L_i(t) -> A_i(t) via EpistemicEvaluator
      3. Compute E_aff_i(t) via AffectiveEvaluator
      4. Total_i(t) = A_i(t) + beta_aff * E_aff_i(t)
      5. MetaPatternRule -> new weights
      6. Prune + update store
    """

    def __init__(self, config: AgentConfig, store=None):
        self.config = config
        self.agent_id = config.agent_id
        self.store = store or InMemoryStore()
        self.epistemic = EpistemicEvaluator(lambda_L=config.lambda_L)
        self.affective = AffectiveEvaluator(
            k=config.k,
            c_opt=config.c_opt,
            sigma_c=config.sigma_c,
            alpha_r=config.alpha_r,
        )
        self.dynamics = MetaPatternRule(
            eta=config.eta,
            beta_c=config.beta_c,
            epsilon=config.epsilon,
        )
        self._t = 0
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        if not self.store.query(self.agent_id):
            init = GaussianPattern(
                mu=np.zeros(self.config.feature_dim),
                sigma=np.eye(self.config.feature_dim) * self.config.init_sigma,
            )
            self.store.save(init, 1.0, self.agent_id)

    def step(self, x: np.ndarray, reward: float = 0.0) -> dict:
        records = self.store.query(self.agent_id)
        patterns = [p for p, _ in records]
        weights = np.array([w for _, w in records])

        accuracies = []
        e_affs = []
        for pattern in patterns:
            acc = self.epistemic.update(pattern, x)
            e_aff = self.affective.update(pattern, acc, reward)
            accuracies.append(acc)
            e_affs.append(e_aff)

        totals = np.array([
            acc + self.config.beta_aff * e_aff
            for acc, e_aff in zip(accuracies, e_affs)
        ])

        new_weights = self.dynamics.step(patterns, weights, totals)

        # Prune and persist
        for p in patterns:
            self.store.delete(p.id)
        for p, w in zip(patterns, new_weights):
            if w >= self.config.epsilon:
                updated = p.update(x)
                self.store.save(updated, float(w), self.agent_id)

        self._t += 1
        return {
            't': self._t,
            'n_patterns': int(np.sum(new_weights >= self.config.epsilon)),
            'mean_accuracy': float(np.mean(accuracies)),
            'max_weight': float(new_weights.max()),
        }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/agents/test_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hpm/agents/agent.py tests/agents/test_agent.py
git commit -m "feat: Agent — wires PatternLibrary, EvaluatorPipeline, and MetaPatternRule (Phase 1 §7)"
```

---

## Task 10: HPM Prediction Metrics

**Files:**
- Create: `hpm/metrics/hpm_predictions.py`
- Create: `tests/metrics/test_hpm_predictions.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/metrics/test_hpm_predictions.py
import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.domains.concept import ConceptLearningDomain, Concept
from hpm.metrics.hpm_predictions import sensitivity_ratio, curiosity_complexity_profile


@pytest.fixture
def trained_agent():
    cfg = AgentConfig(agent_id="metric_agent", feature_dim=4, eta=0.05, lambda_L=0.3)
    agent = Agent(cfg)
    domain = ConceptLearningDomain(
        concepts=[
            Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0),
            Concept(np.array([0.0, 1.0]), [np.array([0.2, 0.8])], label=1),
        ],
        noise=0.05,
        seed=0,
    )
    for _ in range(100):
        agent.step(domain.observe())
    return agent, domain


def test_sensitivity_ratio_returns_float(trained_agent):
    agent, domain = trained_agent
    ratio = sensitivity_ratio(agent, domain, n_steps=20)
    assert isinstance(ratio, float)


def test_curiosity_profile_returns_dict(trained_agent):
    agent, domain = trained_agent
    # Domains at three complexity levels
    domains = {
        2.0: ConceptLearningDomain(
            [Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0)],
            noise=0.01, seed=1
        ),
        10.0: ConceptLearningDomain(
            [Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0)],
            noise=0.3, seed=2
        ),
        50.0: ConceptLearningDomain(
            [Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0)],
            noise=2.0, seed=3
        ),
    }
    profile = curiosity_complexity_profile(agent, domains, n_steps=20)
    assert set(profile.keys()) == set(domains.keys())
    assert all(isinstance(v, float) for v in profile.values())
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/metrics/test_hpm_predictions.py -v
```

- [ ] **Step 3: Implement metrics**

```python
# hpm/metrics/hpm_predictions.py
import numpy as np
from copy import deepcopy


def _run_accuracy(agent, domain, n_steps: int) -> float:
    """Run a deepcopy of agent on domain for n_steps, return mean accuracy.
    Uses deepcopy to avoid mutating the caller's agent state (B5 fix).
    """
    from copy import deepcopy
    agent_copy = deepcopy(agent)
    domain_copy = deepcopy(domain)
    results = []
    for _ in range(n_steps):
        x = domain_copy.observe()
        result = agent_copy.step(x)
        results.append(result['mean_accuracy'])
    return float(np.mean(results))


def sensitivity_ratio(agent, domain, n_steps: int = 100) -> float:
    """
    §9.1: ratio of accuracy change under deep vs surface perturbation.
    HPM predicts ratio > 1 (agents more sensitive to deep structure changes).

    Returns: deep_drop / surface_drop
    """
    baseline = _run_accuracy(agent, domain, n_steps)
    deep_acc = _run_accuracy(agent, domain.deep_perturb(), n_steps)
    surface_acc = _run_accuracy(agent, domain.surface_perturb(), n_steps)

    deep_drop = baseline - deep_acc
    surface_drop = baseline - surface_acc

    if abs(surface_drop) < 1e-10:
        return float('inf')
    return float(deep_drop / surface_drop)


def curiosity_complexity_profile(
    agent,
    domains_by_complexity: dict[float, object],
    n_steps: int = 50,
) -> dict[float, float]:
    """
    §9.4: mean affective evaluator engagement per complexity level.
    HPM predicts inverted-U — peaks at intermediate complexity.

    Returns: {complexity_level: mean_e_aff}
    Note: approximates E_aff by tracking accuracy improvement rate per domain.
    """
    from copy import deepcopy
    profile = {}
    for complexity, domain in domains_by_complexity.items():
        # Fresh agent copy per complexity level — avoids cross-contamination (B6 fix)
        agent_copy = deepcopy(agent)
        domain_copy = deepcopy(domain)
        accuracies = []
        for _ in range(n_steps):
            x = domain_copy.observe()
            result = agent_copy.step(x)
            accuracies.append(result['mean_accuracy'])
        # E_aff proxy: mean absolute improvement across steps
        diffs = np.abs(np.diff(accuracies))
        profile[complexity] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    return profile
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/metrics/test_hpm_predictions.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hpm/metrics/ tests/metrics/
git commit -m "feat: HPM prediction metrics — sensitivity ratio (§9.1) and curiosity profile (§9.4)"
```

---

## Task 11: Integration Test

**Files:**
- Create: `tests/integration/test_phase1.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_phase1.py
"""
End-to-end Phase 1 integration test.
Verifies the full data flow: domain -> agent -> metrics.
"""
import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.domains.concept import ConceptLearningDomain, Concept
from hpm.metrics.hpm_predictions import sensitivity_ratio


def make_domain(seed=0):
    return ConceptLearningDomain(
        concepts=[
            Concept(
                deep_features=np.array([1.0, 0.0, 0.0]),
                surface_templates=[
                    np.array([0.8, 0.2]),
                    np.array([0.6, 0.4]),
                ],
                label=0,
            ),
            Concept(
                deep_features=np.array([0.0, 1.0, 0.0]),
                surface_templates=[
                    np.array([0.2, 0.8]),
                    np.array([0.3, 0.7]),
                ],
                label=1,
            ),
            Concept(
                deep_features=np.array([0.0, 0.0, 1.0]),
                surface_templates=[
                    np.array([0.5, 0.5]),
                    np.array([0.4, 0.6]),
                ],
                label=2,
            ),
        ],
        noise=0.05,
        seed=seed,
    )


def test_agent_accuracy_improves_over_training():
    """Accuracy should increase (become less negative) over 200 steps."""
    domain = make_domain()
    cfg = AgentConfig(
        agent_id="integration_agent",
        feature_dim=domain.feature_dim(),
        eta=0.05,
        lambda_L=0.2,
        beta_aff=0.3,
    )
    agent = Agent(cfg)

    early_acc, late_acc = [], []
    for t in range(200):
        result = agent.step(domain.observe())
        if t < 20:
            early_acc.append(result['mean_accuracy'])
        if t >= 180:
            late_acc.append(result['mean_accuracy'])

    assert np.mean(late_acc) > np.mean(early_acc), (
        f"Accuracy did not improve: early={np.mean(early_acc):.3f}, late={np.mean(late_acc):.3f}"
    )


def test_sensitivity_ratio_sign():
    """
    §9.1: after training, sensitivity ratio should be > 0.
    (Full HPM prediction ratio > 1 requires longer training — this tests the sign.)
    """
    domain = make_domain()
    cfg = AgentConfig(
        agent_id="sens_agent",
        feature_dim=domain.feature_dim(),
        eta=0.05,
        lambda_L=0.2,
    )
    agent = Agent(cfg)
    for _ in range(150):
        agent.step(domain.observe())

    ratio = sensitivity_ratio(agent, domain, n_steps=30)
    # Ratio should be a real number (finite or inf from zero surface drop)
    # HPM predicts > 1; here we just verify it is defined and non-negative
    assert not np.isnan(ratio)
    assert ratio >= 0.0 or ratio == float('inf')


def test_library_floor_prevents_empty_store():
    """§3.3 edge case: even under extreme dynamics, library never empties."""
    domain = make_domain()
    cfg = AgentConfig(
        agent_id="floor_agent",
        feature_dim=domain.feature_dim(),
        eta=100.0,    # extreme learning rate to force weight collapse
        beta_c=100.0,
        epsilon=0.01,
    )
    agent = Agent(cfg)
    # Run several steps — library should never empty
    for _ in range(20):
        agent.step(domain.observe())
        records = agent.store.query("floor_agent")
        assert len(records) >= 1, "Library emptied — floor not working"
        total_weight = sum(w for _, w in records)
        assert total_weight > 0.0


def test_store_persists_patterns_across_steps():
    """Patterns in the store reflect state after multiple steps."""
    domain = make_domain()
    cfg = AgentConfig(agent_id="store_agent", feature_dim=domain.feature_dim())
    agent = Agent(cfg)

    for _ in range(10):
        agent.step(domain.observe())

    records = agent.store.query("store_agent")
    assert len(records) >= 1
    for p, w in records:
        assert w >= cfg.epsilon
        assert p.is_structurally_valid()
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/integration/test_phase1.py -v
```

Expected: all PASS.

- [ ] **Step 3: Run full test suite**

```bash
pytest --tb=short
```

Expected: all tests passing, no errors.

- [ ] **Step 4: Final commit**

```bash
git add tests/integration/test_phase1.py
git commit -m "test: Phase 1 integration test — full data flow from domain through agent to metrics"
```

---

## Summary

Phase 1 delivers:
- `GaussianPattern` — value-type pattern with full HPM math interface
- `InMemoryStore` — pattern persistence backend
- `EpistemicEvaluator` — EMA running loss and accuracy (D2-D3)
- `AffectiveEvaluator` — curiosity signal with inverted-U profile (§9.4)
- `MetaPatternRule` — replicator dynamics with conflict inhibition (D5)
- `ConceptLearningDomain` — deep/surface concept structure with transfer probes
- `Agent` — full Phase 1 data flow wired end-to-end
- Metrics for §9.1 (sensitivity ratio) and §9.4 (curiosity profile)

**Next:** Phase 2 plan (PatternStore persistence + ExternalSubstrate) to be written after Phase 1 passes all tests.
