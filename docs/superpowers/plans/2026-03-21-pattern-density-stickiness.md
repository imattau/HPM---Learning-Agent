# Pattern Density + Stickiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pattern density D(h_i) as a stickiness bias in the weight update, so structurally coherent, evaluator-confirmed, and socially amplified patterns resist decay even when epistemically below average.

**Architecture:** Three components feed a `PatternDensity` class (structural connectivity, evaluator saturation, field amplification), which outputs a scalar in [0,1]. The scalar is applied as a `kappa_D * D(h_i) * w_i` bias inside `MetaPatternRule.step()` before the existing renormalisation. `kappa_D = 0.0` by default — existing behaviour is fully preserved.

**Tech Stack:** Python, numpy. No new dependencies.

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `hpm/dynamics/density.py` | `PatternDensity` class — computes D(h_i) |
| Modify | `hpm/evaluators/affective.py` | Add `_last_capacity` store + `last_capacity()` accessor |
| Modify | `hpm/dynamics/meta_pattern_rule.py` | Add `kappa_D` to `__init__`, `densities=None` to `step()` |
| Modify | `hpm/config.py` | Add `kappa_D`, `alpha_conn`, `alpha_sat`, `alpha_amp` |
| Modify | `hpm/agents/agent.py` | Instantiate PatternDensity, compute densities, wire to dynamics |
| Modify | `hpm/dynamics/__init__.py` | Re-export `PatternDensity` |
| Create | `tests/dynamics/test_density.py` | Unit tests for PatternDensity |
| Modify | `tests/evaluators/test_affective.py` | Add last_capacity tests |
| Create | `tests/dynamics/test_meta_pattern_rule_density.py` | Integration tests for density bias |

---

### Task 1: PatternDensity class

**Files:**
- Create: `hpm/dynamics/density.py`
- Create: `tests/dynamics/test_density.py`

**Background for implementer:**
- `GaussianPattern.connectivity()` — mean absolute off-diagonal correlation of sigma, returns float in [0,1]. For 1-D patterns returns 0.0 (no off-diagonal elements).
- `GaussianPattern.compress()` — ratio of largest eigenvalue to trace(sigma), returns float in [0,1]. For a valid 1-D pattern returns 1.0.
- The `loss` parameter is the running loss L_i (non-negative). The caller passes `-epi_acc` where `epi_acc = EpistemicEvaluator.update()` return value (which is `A_i = -L_i`, so negating it gives L_i ≥ 0).
- Formula: `D(h_i) = alpha_conn * structural + alpha_sat * saturation + alpha_amp * field_freq`
- `structural = (connectivity + compress) / 2`
- `saturation = (1 - loss/(1+loss)) * capacity`
- Clamp `loss = max(loss, 0.0)` defensively. Clamp final D to [0,1].

- [ ] **Step 1: Write the failing tests**

```python
# tests/dynamics/test_density.py
import numpy as np
import pytest
from hpm.dynamics.density import PatternDensity
from hpm.patterns.gaussian import GaussianPattern


def _pattern(dim=4, diag_only=True):
    """Helper: GaussianPattern with identity covariance (connectivity=0 for dim>1)."""
    return GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))


def test_density_low_when_saturation_and_field_zero():
    """With zero saturation (high loss) and zero field_freq, D is driven only by structural."""
    pd = PatternDensity(alpha_conn=0.33, alpha_sat=0.33, alpha_amp=0.34)
    # Use mock to control connectivity() and compress() exactly
    class ZeroStructPattern:
        def connectivity(self): return 0.0
        def compress(self): return 0.0
    p = ZeroStructPattern()
    D = pd.compute(p, loss=1e9, capacity=0.0, field_freq=0.0)
    assert D == 0.0  # all three components are zero -> D = 0


def test_density_one_when_all_components_maxed():
    """Max connectivity+compress, zero loss, full capacity, full field_freq -> D near 1."""
    pd = PatternDensity(alpha_conn=0.33, alpha_sat=0.33, alpha_amp=0.34)
    # Mock pattern with connectivity=1.0 and compress=1.0 by subclassing
    class HighDensityPattern:
        def connectivity(self): return 1.0
        def compress(self): return 1.0
    p = HighDensityPattern()
    D = pd.compute(p, loss=0.0, capacity=1.0, field_freq=1.0)
    assert abs(D - 1.0) < 1e-6


def test_structural_component_uses_connectivity_and_compress():
    pd = PatternDensity(alpha_conn=1.0, alpha_sat=0.0, alpha_amp=0.0)
    class FakePattern:
        def connectivity(self): return 0.6
        def compress(self): return 0.4
    p = FakePattern()
    D = pd.compute(p, loss=0.0, capacity=0.0, field_freq=0.0)
    expected_structural = (0.6 + 0.4) / 2  # = 0.5
    assert abs(D - expected_structural) < 1e-9


def test_saturation_high_for_low_loss_high_capacity():
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=1.0, alpha_amp=0.0)
    class ZeroStructPattern:
        def connectivity(self): return 0.0
        def compress(self): return 0.0
    p = ZeroStructPattern()
    # Low loss (0.01) -> loss_norm = 0.01/1.01 ≈ 0.0099 -> (1-norm) ≈ 0.99
    # High capacity (0.9) -> saturation = 0.99 * 0.9 ≈ 0.891
    D = pd.compute(p, loss=0.01, capacity=0.9, field_freq=0.0)
    assert D > 0.8


def test_saturation_low_for_high_loss():
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=1.0, alpha_amp=0.0)
    class ZeroStructPattern:
        def connectivity(self): return 0.0
        def compress(self): return 0.0
    p = ZeroStructPattern()
    # High loss (100) -> loss_norm = 100/101 ≈ 0.99 -> (1-norm) ≈ 0.01
    # Even with full capacity=1.0 -> saturation ≈ 0.01
    D = pd.compute(p, loss=100.0, capacity=1.0, field_freq=0.0)
    assert D < 0.05


def test_field_freq_zero_contribution():
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=0.0, alpha_amp=1.0)
    class ZeroStructPattern:
        def connectivity(self): return 0.0
        def compress(self): return 0.0
    p = ZeroStructPattern()
    D_zero = pd.compute(p, loss=0.0, capacity=0.0, field_freq=0.0)
    D_half = pd.compute(p, loss=0.0, capacity=0.0, field_freq=0.5)
    assert abs(D_zero) < 1e-9
    assert abs(D_half - 0.5) < 1e-9


def test_negative_loss_clamped_to_zero():
    """Defensive: negative loss should be treated as zero (not break the formula)."""
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=1.0, alpha_amp=0.0)
    class ZeroStructPattern:
        def connectivity(self): return 0.0
        def compress(self): return 0.0
    p = ZeroStructPattern()
    D_neg = pd.compute(p, loss=-5.0, capacity=1.0, field_freq=0.0)
    D_zero = pd.compute(p, loss=0.0, capacity=1.0, field_freq=0.0)
    assert abs(D_neg - D_zero) < 1e-9


def test_output_clamped_to_unit_interval():
    """Alphas > 1 should not produce D > 1 due to final clamp."""
    pd = PatternDensity(alpha_conn=5.0, alpha_sat=5.0, alpha_amp=5.0)
    class MaxPattern:
        def connectivity(self): return 1.0
        def compress(self): return 1.0
    p = MaxPattern()
    D = pd.compute(p, loss=0.0, capacity=1.0, field_freq=1.0)
    assert 0.0 <= D <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
pytest tests/dynamics/test_density.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'hpm.dynamics.density'`

- [ ] **Step 3: Implement `hpm/dynamics/density.py`**

```python
# hpm/dynamics/density.py


class PatternDensity:
    """
    Computes pattern density D(h_i) in [0, 1].

    D(h_i) = alpha_conn * structural_i
            + alpha_sat * saturation_i
            + alpha_amp * field_freq_i

    structural_i  = (connectivity() + compress()) / 2
    saturation_i  = (1 - loss_norm) * capacity_i
    loss_norm     = loss / (1 + loss)   maps [0, ∞) -> [0, 1)

    Stateless: call compute() once per pattern per step.
    """

    def __init__(
        self,
        alpha_conn: float = 0.33,
        alpha_sat: float = 0.33,
        alpha_amp: float = 0.34,
    ):
        self.alpha_conn = alpha_conn
        self.alpha_sat = alpha_sat
        self.alpha_amp = alpha_amp

    def compute(
        self,
        pattern,
        loss: float,
        capacity: float,
        field_freq: float,
    ) -> float:
        """
        Args:
            pattern:    GaussianPattern (must have connectivity() and compress())
            loss:       Running loss L_i = -A_i (non-negative). Defensively clamped to >= 0.
            capacity:   1 - novelty from AffectiveEvaluator.last_capacity()
            field_freq: Normalised field frequency from PatternField.freqs_for(), in [0, 1]

        Returns:
            D(h_i) in [0, 1]
        """
        loss = max(loss, 0.0)

        structural = (pattern.connectivity() + pattern.compress()) / 2.0
        loss_norm = loss / (1.0 + loss)
        saturation = (1.0 - loss_norm) * capacity
        field = field_freq

        D = (
            self.alpha_conn * structural
            + self.alpha_sat * saturation
            + self.alpha_amp * field
        )
        return float(max(0.0, min(1.0, D)))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/dynamics/test_density.py -v
```
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add hpm/dynamics/density.py tests/dynamics/test_density.py
git commit -m "feat: add PatternDensity class with structural/saturation/field components"
```

---

### Task 2: AffectiveEvaluator — expose last_capacity()

**Files:**
- Modify: `hpm/evaluators/affective.py`
- Modify: `tests/evaluators/test_affective.py`

**Background for implementer:**
- `AffectiveEvaluator.update()` already computes `capacity = 1.0 - novelty` locally (line 43). We need to store it.
- The store goes **immediately after** `capacity = 1.0 - novelty`, before the `c = pattern.description_length()` line.
- Add `_last_capacity: dict[str, float] = {}` in `__init__`.
- `last_capacity(pattern_id)` returns `self._last_capacity.get(pattern_id, 0.0)`.
- On the first call to `update()` for a pattern, `delta_A = 0` (prev defaults to current), so `novelty = sigmoid(0) = 0.5`, `capacity = 0.5`. So `last_capacity()` will return `0.5` on step 1, not 0.0. The 0.0 fallback only applies to IDs never seen by `update()`.

- [ ] **Step 1: Add last_capacity tests to existing test file**

Open `tests/evaluators/test_affective.py` and append these tests:

```python
def test_last_capacity_returns_zero_for_unknown_pattern_id():
    """ID never passed to update() returns 0.0 (dict default)."""
    evaluator = AffectiveEvaluator()
    assert evaluator.last_capacity("never_seen") == 0.0


def test_last_capacity_reflects_current_step():
    """After one update(), last_capacity() returns the capacity from that step."""
    evaluator = AffectiveEvaluator(k=1.0)
    pattern = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    evaluator.update(pattern, current_accuracy=1.0)
    cap = evaluator.last_capacity(pattern.id)
    # First update: delta_A=0 -> novelty=sigmoid(0)=0.5 -> capacity=0.5
    assert abs(cap - 0.5) < 1e-9


def test_capacity_is_one_minus_novelty():
    """last_capacity() == 1 - novelty for a given step."""
    import numpy as np
    evaluator = AffectiveEvaluator(k=2.0)
    pattern = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    # First step
    evaluator.update(pattern, current_accuracy=5.0)
    # Second step with different accuracy to produce non-zero delta_A
    evaluator.update(pattern, current_accuracy=3.0)
    # delta_A = 3.0 - 5.0 = -2.0, novelty = sigmoid(2.0 * -2.0) = sigmoid(-4)
    expected_novelty = 1.0 / (1.0 + np.exp(4.0))
    expected_capacity = 1.0 - expected_novelty
    assert abs(evaluator.last_capacity(pattern.id) - expected_capacity) < 1e-9
```

Note: ensure imports at top of that file include `GaussianPattern` if not already present — check the existing imports first and add only what's missing.

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/evaluators/test_affective.py::test_last_capacity_returns_zero_for_unknown_pattern_id -v
```
Expected: `AttributeError: 'AffectiveEvaluator' object has no attribute 'last_capacity'`

- [ ] **Step 3: Modify `hpm/evaluators/affective.py`**

In `__init__`, add after `self._prev_accuracy: dict[str, float] = {}`:
```python
self._last_capacity: dict[str, float] = {}
```

In `update()`, add after `capacity = 1.0 - novelty`:
```python
self._last_capacity[pattern.id] = capacity
```

Add new method after `update()`:
```python
def last_capacity(self, pattern_id: str) -> float:
    """Return the most recently computed capacity (1 - novelty) for pattern_id.
    Returns 0.0 if pattern_id has never been passed to update()."""
    return self._last_capacity.get(pattern_id, 0.0)
```

- [ ] **Step 4: Run all affective tests**

```bash
pytest tests/evaluators/test_affective.py -v
```
Expected: all tests PASS (including the 3 new ones)

- [ ] **Step 5: Commit**

```bash
git add hpm/evaluators/affective.py tests/evaluators/test_affective.py
git commit -m "feat: expose last_capacity() on AffectiveEvaluator"
```

---

### Task 3: AgentConfig new fields

**Files:**
- Modify: `hpm/config.py`

**Background for implementer:**
- Add four new fields at the end of the `AgentConfig` dataclass.
- All have backward-compatible defaults so existing code is unaffected.
- No tests needed for dataclass field additions — the agent integration test in Task 5 covers them.

- [ ] **Step 1: Add fields to `hpm/config.py`**

After the `w_cpu` field, append:
```python
    # Pattern density (density bias in MetaPatternRule, §A.8)
    kappa_D: float = 0.0     # density bias weight (0 = off, backward compatible)
    alpha_conn: float = 0.33  # weight of structural connectivity in D(h)
    alpha_sat: float = 0.33   # weight of evaluator saturation in D(h)
    alpha_amp: float = 0.34   # weight of field amplification in D(h)
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
pytest --tb=short -q
```
Expected: all 125 existing tests PASS, 0 failures

- [ ] **Step 3: Commit**

```bash
git add hpm/config.py
git commit -m "feat: add kappa_D, alpha_conn, alpha_sat, alpha_amp to AgentConfig"
```

---

### Task 4: MetaPatternRule — density bias term

**Files:**
- Modify: `hpm/dynamics/meta_pattern_rule.py`
- Create: `tests/dynamics/test_meta_pattern_rule_density.py`

**Background for implementer:**
- Add `kappa_D: float = 0.0` to `MetaPatternRule.__init__`. Store as `self.kappa_D`.
- Existing callers `MetaPatternRule(eta=..., beta_c=..., epsilon=...)` are unaffected.
- Add `densities=None` to `step()`. When `densities` is not None, after computing `replicator` and `conflict`, add `kappa_D * densities[i] * weights[i]` to the update.
- The density bias is applied **inside the per-pattern loop**, before the `np.maximum(..., 0.0)` clip. The floor case (all weights < epsilon) is unchanged — it still returns early without applying densities.
- The existing renormalisation (`new_weights /= total`) already runs after the loop — no change needed.

- [ ] **Step 1: Write failing tests**

```python
# tests/dynamics/test_meta_pattern_rule_density.py
import numpy as np
import pytest
from hpm.dynamics.meta_pattern_rule import MetaPatternRule
from hpm.patterns.gaussian import GaussianPattern


def _two_patterns():
    p1 = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    p2 = GaussianPattern(mu=np.ones(2) * 3.0, sigma=np.eye(2))
    return [p1, p2]


def test_kappa_d_zero_unchanged_from_baseline():
    """kappa_D=0 with densities=[0.9, 0.1] produces identical output to no-densities call."""
    patterns = _two_patterns()
    weights = np.array([0.6, 0.4])
    totals = np.array([1.0, 0.5])

    rule_baseline = MetaPatternRule(eta=0.1, beta_c=0.1, epsilon=1e-4, kappa_D=0.0)
    rule_with = MetaPatternRule(eta=0.1, beta_c=0.1, epsilon=1e-4, kappa_D=0.0)

    # Use same RNG seed for reproducibility
    rule_baseline._rng = np.random.default_rng(42)
    rule_with._rng = np.random.default_rng(42)

    w_baseline = rule_baseline.step(patterns, weights.copy(), totals)
    w_with = rule_with.step(patterns, weights.copy(), totals, densities=[0.9, 0.1])

    np.testing.assert_allclose(w_baseline, w_with, atol=1e-12)


def test_density_bias_increases_high_density_pattern_weight():
    """High-density pattern gains relatively more weight than low-density under kappa_D > 0."""
    patterns = _two_patterns()
    weights = np.array([0.5, 0.5])
    # Equal totals so replicator contributes nothing; only density bias and conflict matter
    totals = np.array([1.0, 1.0])
    densities = [0.9, 0.1]  # pattern 0 is high density

    rule = MetaPatternRule(eta=0.1, beta_c=0.0, epsilon=1e-4, kappa_D=0.5)
    rule._rng = np.random.default_rng(0)
    new_w = rule.step(patterns, weights.copy(), totals, densities=densities)

    # Pattern 0 should gain weight relative to pattern 1
    assert new_w[0] > new_w[1]


def test_renormalisation_holds_after_density_bias():
    """Weights sum to 1.0 after step with density bias."""
    patterns = _two_patterns()
    weights = np.array([0.5, 0.5])
    totals = np.array([1.0, 0.8])
    densities = [0.7, 0.3]

    rule = MetaPatternRule(eta=0.05, beta_c=0.1, epsilon=1e-4, kappa_D=0.2)
    rule._rng = np.random.default_rng(1)
    new_w = rule.step(patterns, weights.copy(), totals, densities=densities)

    assert abs(new_w.sum() - 1.0) < 1e-9


def test_densities_none_behaves_identically():
    """Calling step() with densities=None is identical to omitting the argument."""
    patterns = _two_patterns()
    weights = np.array([0.6, 0.4])
    totals = np.array([1.0, 0.5])

    rule1 = MetaPatternRule(eta=0.1, kappa_D=0.3)
    rule2 = MetaPatternRule(eta=0.1, kappa_D=0.3)
    rule1._rng = np.random.default_rng(5)
    rule2._rng = np.random.default_rng(5)

    w1 = rule1.step(patterns, weights.copy(), totals)          # no densities arg
    w2 = rule2.step(patterns, weights.copy(), totals, densities=None)  # explicit None

    np.testing.assert_allclose(w1, w2, atol=1e-12)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/dynamics/test_meta_pattern_rule_density.py -v 2>&1 | head -20
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'kappa_D'`

- [ ] **Step 3: Modify `hpm/dynamics/meta_pattern_rule.py`**

Change `__init__` signature from:
```python
def __init__(self, eta: float = 0.01, beta_c: float = 0.1, epsilon: float = 1e-4):
    self.eta = eta
    self.beta_c = beta_c
    self.epsilon = epsilon
```
to:
```python
def __init__(self, eta: float = 0.01, beta_c: float = 0.1, epsilon: float = 1e-4, kappa_D: float = 0.0):
    self.eta = eta
    self.beta_c = beta_c
    self.epsilon = epsilon
    self.kappa_D = kappa_D
```

Change `step()` signature from:
```python
def step(self, patterns: list, weights: np.ndarray, totals: np.ndarray) -> np.ndarray:
```
to:
```python
def step(self, patterns: list, weights: np.ndarray, totals: np.ndarray, densities=None) -> np.ndarray:
```

In the per-pattern loop, change:
```python
new_weights[i] = weights[i] + replicator - conflict
```
to:
```python
density_bias = self.kappa_D * densities[i] * weights[i] if densities is not None else 0.0
new_weights[i] = weights[i] + replicator - conflict + density_bias
```

- [ ] **Step 4: Run all dynamics tests**

```bash
pytest tests/dynamics/ -v
```
Expected: all tests PASS (existing meta_pattern_rule tests + 4 new density tests)

- [ ] **Step 5: Commit**

```bash
git add hpm/dynamics/meta_pattern_rule.py tests/dynamics/test_meta_pattern_rule_density.py
git commit -m "feat: add kappa_D density bias to MetaPatternRule"
```

---

### Task 5: Agent wiring + integration tests

**Files:**
- Modify: `hpm/agents/agent.py`
- Modify: `hpm/dynamics/__init__.py`
- Create: `tests/dynamics/test_meta_pattern_rule_density.py` ← add agent integration tests here (append to the file from Task 4)

**Background for implementer:**
- Import and instantiate `PatternDensity` in `Agent.__init__` using the config alpha values.
- In `Agent.step()`, after the affective/epistemic loop and before totals computation, compute densities:
  1. `cap = self.affective.last_capacity(p.id)` per pattern
  2. `loss = -epi` (negate the accuracy to get non-negative loss L_i)
  3. `self.pattern_density.compute(p, loss, cap, ff)` where `ff = field_freqs[i]`
- Pass `densities` to `self.dynamics.step(patterns, weights, totals, densities=densities)`.
- Add `density_mean` to the return dict: `float(np.mean(densities)) if len(densities) > 0 else 0.0`.
- `hpm/dynamics/__init__.py` is currently empty (1 line). Add `from .density import PatternDensity`.

- [ ] **Step 1: Write failing agent integration tests**

Append to `tests/dynamics/test_meta_pattern_rule_density.py`:

```python
import numpy as np
from hpm.config import AgentConfig
from hpm.agents.agent import Agent


def test_agent_step_includes_density_mean():
    """Agent with kappa_D > 0 returns density_mean in step dict."""
    config = AgentConfig(agent_id="test", feature_dim=4, kappa_D=0.1)
    agent = Agent(config)
    result = agent.step(np.zeros(4))
    assert "density_mean" in result
    assert isinstance(result["density_mean"], float)
    assert 0.0 <= result["density_mean"] <= 1.0


def test_agent_step_density_mean_with_kappa_d_zero():
    """density_mean is present even when kappa_D=0 (default)."""
    config = AgentConfig(agent_id="test2", feature_dim=4)  # kappa_D=0.0 default
    agent = Agent(config)
    result = agent.step(np.zeros(4))
    assert "density_mean" in result


def test_kappa_d_zero_density_mean_in_range():
    """Agent with kappa_D=0 (default) still computes and returns a valid density_mean."""
    x = np.zeros(4)
    config = AgentConfig(agent_id="c", feature_dim=4, kappa_D=0.0)
    agent = Agent(config)
    result = agent.step(x)
    assert "density_mean" in result
    assert 0.0 <= result["density_mean"] <= 1.0
    # With a single pattern, density_mean == density of that pattern
    assert result["n_patterns"] >= 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/dynamics/test_meta_pattern_rule_density.py::test_agent_step_includes_density_mean -v
```
Expected: `KeyError: 'density_mean'` or `ImportError`

- [ ] **Step 3: Modify `hpm/dynamics/__init__.py`**

Replace the empty file content with:
```python
from .density import PatternDensity

__all__ = ["PatternDensity"]
```

- [ ] **Step 4: Modify `hpm/agents/agent.py`**

Add import at top:
```python
from ..dynamics.density import PatternDensity
```

In `__init__`, after instantiating `self.resource_cost`, add:
```python
self.pattern_density = PatternDensity(
    alpha_conn=config.alpha_conn,
    alpha_sat=config.alpha_sat,
    alpha_amp=config.alpha_amp,
)
```

In `step()`, after `e_socs = self.social.evaluate_all(freq_totals)` and before the `e_costs` block, add:

```python
        # Compute per-pattern density D(h_i) using current evaluator state
        densities = [
            self.pattern_density.compute(
                p,
                loss=-epi,          # loss L_i = -A_i (non-negative)
                capacity=self.affective.last_capacity(p.id),
                field_freq=ff,
            )
            for p, epi, ff in zip(patterns, epistemic_accs, field_freqs)
        ]
```

Change the dynamics call from:
```python
        new_weights = self.dynamics.step(patterns, weights, totals)
```
to:
```python
        new_weights = self.dynamics.step(patterns, weights, totals, densities=densities)
```

In the return dict, add after `'e_cost_mean'`:
```python
            'density_mean': float(np.mean(densities)) if len(densities) > 0 else 0.0,
```

- [ ] **Step 5: Run the full test suite**

```bash
pytest --tb=short -q
```
Expected: all 125 existing tests PASS + new density and agent tests PASS. No failures.

- [ ] **Step 6: Commit**

```bash
git add hpm/agents/agent.py hpm/dynamics/__init__.py tests/dynamics/test_meta_pattern_rule_density.py
git commit -m "feat: wire PatternDensity into Agent.step() and MetaPatternRule"
```

---

### Task 6: Final check

- [ ] **Step 1: Run full suite one more time**

```bash
pytest -v --tb=short 2>&1 | tail -20
```
Expected: all tests pass, summary shows the new test count (should be ~135+ tests).

- [ ] **Step 2: Quick sanity — kappa_D > 0 affects weights**

```bash
python -c "
import numpy as np
from hpm.config import AgentConfig
from hpm.agents.agent import Agent

x = np.zeros(4)
config = AgentConfig(agent_id='sanity', feature_dim=4, kappa_D=0.5)
agent = Agent(config)
result = agent.step(x)
print('density_mean:', result['density_mean'])
print('max_weight:', result['max_weight'])
print('n_patterns:', result['n_patterns'])
assert 'density_mean' in result
print('OK')
"
```

- [ ] **Step 3: Commit if anything was left unstaged, then push**

```bash
git status
git push
```
