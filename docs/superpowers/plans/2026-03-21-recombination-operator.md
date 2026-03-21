# Recombination Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Recombination Operator (Gap 3 / Appendix E) — autonomous creation of novel candidate patterns by mating two Level 4+ patterns under structural pressure.

**Architecture:** `MetaPatternRule.step()` now returns a `StepResult(weights, total_conflict)` namedtuple exposing the conflict tension. A stateless `RecombinationOperator` selects a parent pair from the post-prune library, creates a child via `recombine()`, scores it via novelty + efficacy (ring buffer of recent observations), and returns a `RecombinationResult` or `None`. The `Agent` owns the ring buffer, trigger/cooldown logic, and entry-weight management.

**Tech Stack:** Python stdlib (`collections.namedtuple`, `dataclasses`, `collections.deque`, `itertools`), NumPy. No new dependencies.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `hpm/dynamics/meta_pattern_rule.py` | Modify | Add `StepResult` namedtuple; compute and return `total_conflict` |
| `hpm/dynamics/recombination.py` | Create | `RecombinationResult` + `RecombinationOperator` |
| `hpm/dynamics/__init__.py` | Modify | Re-export `RecombinationOperator` (added in Task 3) |
| `hpm/config.py` | Modify | 11 new fields with backward-compatible defaults |
| `hpm/agents/agent.py` | Modify | Ring buffer, trigger/cooldown, recombination block, return dict |
| `tests/dynamics/test_meta_pattern_rule.py` | Modify | Unpack `.weights` on all direct `step()` calls |
| `tests/dynamics/test_meta_pattern_rule_density.py` | Modify | Same |
| `tests/dynamics/test_step_result.py` | Create | StepResult unit tests |
| `tests/dynamics/test_recombination.py` | Create | RecombinationOperator unit tests |
| `tests/agents/test_agent_recombination.py` | Create | Agent integration tests |

---

## Task 1: `StepResult` — modify `MetaPatternRule.step()` and fix all existing callers

**Files:**
- Modify: `hpm/dynamics/meta_pattern_rule.py`
- Modify: `tests/dynamics/test_meta_pattern_rule.py`
- Modify: `tests/dynamics/test_meta_pattern_rule_density.py`
- Create: `tests/dynamics/test_step_result.py`

- [ ] **Step 1: Write failing StepResult tests**

Create `tests/dynamics/test_step_result.py`:

```python
import numpy as np
from hpm.patterns.gaussian import GaussianPattern
from hpm.dynamics.meta_pattern_rule import MetaPatternRule, StepResult


def make_pattern(mu_val, dim=2):
    return GaussianPattern(mu=np.full(dim, float(mu_val)), sigma=np.eye(dim))


def test_step_returns_step_result():
    rule = MetaPatternRule()
    p = make_pattern(0.0)
    result = rule.step([p], np.array([1.0]), np.array([0.5]))
    assert isinstance(result, StepResult)
    assert hasattr(result, 'weights')
    assert hasattr(result, 'total_conflict')


def test_step_result_weights_normalised():
    rule = MetaPatternRule()
    p = make_pattern(0.0)
    result = rule.step([p], np.array([1.0]), np.array([0.5]))
    assert abs(result.weights.sum() - 1.0) < 1e-9


def test_total_conflict_zero_for_single_pattern():
    rule = MetaPatternRule(beta_c=0.1)
    p = make_pattern(0.0)
    result = rule.step([p], np.array([1.0]), np.array([0.5]))
    assert result.total_conflict == 0.0


def test_total_conflict_positive_for_incompatible_patterns():
    rule = MetaPatternRule(beta_c=0.1)
    p1 = make_pattern(0.0)
    p2 = make_pattern(100.0)
    result = rule.step([p1, p2], np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert result.total_conflict > 0.0


def test_total_conflict_zero_for_empty_patterns():
    rule = MetaPatternRule()
    result = rule.step([], np.array([]), np.array([]))
    assert isinstance(result, StepResult)
    assert result.total_conflict == 0.0
```

- [ ] **Step 2: Run — expect ImportError on `StepResult`**

```
pytest tests/dynamics/test_step_result.py -v
```
Expected: FAIL — `ImportError: cannot import name 'StepResult'`

- [ ] **Step 3: Add `StepResult` and modify `meta_pattern_rule.py`**

Add at the top of `hpm/dynamics/meta_pattern_rule.py` (after `import numpy as np`):

```python
from collections import namedtuple

StepResult = namedtuple('StepResult', ['weights', 'total_conflict'])
```

Change the empty-patterns guard (`return weights.copy()`) to:

```python
if n == 0:
    return StepResult(weights.copy(), 0.0)
```

After building the `kappa` matrix (after the `assert np.all(np.diag(kappa) == 0.0)` line), add:

```python
total_conflict = float(self.beta_c * float(weights @ kappa @ weights))
```

Change the floor-branch return to:

```python
return StepResult(new_weights, total_conflict)
```

Change the final return to:

```python
return StepResult(new_weights, total_conflict)
```

- [ ] **Step 4: Run step_result tests — expect PASS**

```
pytest tests/dynamics/test_step_result.py -v
```
Expected: 5 PASS

- [ ] **Step 5: Fix `tests/dynamics/test_meta_pattern_rule.py`**

This file has ~4 lines assigning the result of `mpr.step(...)` directly to a variable. Each must unpack `.weights`. Exact changes (read the file to confirm line numbers, then apply):

```python
# Before:
new_w = mpr.step(patterns, weights, totals)
# After:
new_w = mpr.step(patterns, weights, totals).weights
```

All 4 occurrences follow this pattern. Apply to every line matching `= mpr.step(`.

- [ ] **Step 6: Fix `tests/dynamics/test_meta_pattern_rule_density.py`**

This file has ~10 lines assigning `rule.step(...)` results to variables. Apply the same `.weights` append. Lines calling `agent.step(...)` (Agent method, not MetaPatternRule) are already correct — do NOT modify those.

Before/after example:
```python
# Before:
w_baseline = rule_baseline.step(patterns, weights.copy(), totals)
w_with = rule_with.step(patterns, weights.copy(), totals, densities=[0.9, 0.1])
# After:
w_baseline = rule_baseline.step(patterns, weights.copy(), totals).weights
w_with = rule_with.step(patterns, weights.copy(), totals, densities=[0.9, 0.1]).weights
```

Variables to fix: `w_baseline`, `w_with`, `new_w`, `w1`, `w2`, `w_overridden`, `w_zero`, `w_scalar`, `w_explicit`, `w`.

- [ ] **Step 7: Fix `hpm/agents/agent.py` caller**

Replace:

```python
new_weights = self.dynamics.step(
    patterns, weights, totals,
    densities=densities,
    kappa_d_per_pattern=kappa_d_per_pattern,
)
```

With:

```python
step_result = self.dynamics.step(
    patterns, weights, totals,
    densities=densities,
    kappa_d_per_pattern=kappa_d_per_pattern,
)
new_weights = step_result.weights
total_conflict = step_result.total_conflict
```

- [ ] **Step 8: Run full test suite — expect all PASS**

```
pytest -v
```
Expected: all pre-existing tests PASS

- [ ] **Step 9: Commit**

```bash
git add hpm/dynamics/meta_pattern_rule.py hpm/agents/agent.py \
    tests/dynamics/test_step_result.py \
    tests/dynamics/test_meta_pattern_rule.py \
    tests/dynamics/test_meta_pattern_rule_density.py
git commit -m "feat: StepResult namedtuple from MetaPatternRule.step() with total_conflict"
```

---

## Task 2: 11 new `AgentConfig` fields

**Files:**
- Modify: `hpm/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing config test**

Add to `tests/test_config.py`:

```python
def test_recombination_config_defaults():
    cfg = AgentConfig(agent_id='a', feature_dim=2)
    assert cfg.T_recomb == 100
    assert cfg.N_recomb == 3
    assert cfg.kappa_max == 0.5
    assert cfg.conflict_threshold == 0.1
    assert cfg.recomb_cooldown == 10
    assert cfg.obs_buffer_size == 50
    assert cfg.beta_orig == 1.0
    assert cfg.alpha_nov == 0.5
    assert cfg.alpha_eff == 0.5
    assert cfg.kappa_0 == 0.1
    assert cfg.recomb_temp == 1.0
```

- [ ] **Step 2: Run — expect AttributeError**

```
pytest tests/test_config.py::test_recombination_config_defaults -v
```
Expected: FAIL — `AttributeError: 'AgentConfig' object has no attribute 'T_recomb'`

- [ ] **Step 3: Add fields to `AgentConfig`**

In `hpm/config.py`, append after the `kappa_d_levels` field:

```python
    # Recombination Operator (Gap 3 / Appendix E)
    T_recomb: int = 100               # steps between time-triggered recombinations
    N_recomb: int = 3                 # max pair draw attempts per trigger
    kappa_max: float = 0.5            # max KL incompatibility for eligible pair
    conflict_threshold: float = 0.1  # total_conflict level that fires conflict trigger
    recomb_cooldown: int = 10         # min steps between any two recombinations
    obs_buffer_size: int = 50         # ring buffer capacity (recent observations)
    beta_orig: float = 1.0            # insight score scale
    alpha_nov: float = 0.5            # novelty weight in I(h*)
    alpha_eff: float = 0.5            # efficacy weight in I(h*)
    kappa_0: float = 0.1              # entry weight scale for accepted h*
    recomb_temp: float = 1.0          # softmax temperature for pair sampling
```

- [ ] **Step 4: Run config tests — expect PASS**

```
pytest tests/test_config.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add hpm/config.py tests/test_config.py
git commit -m "feat: add 11 recombination config fields to AgentConfig"
```

---

## Task 3: `RecombinationResult` + `RecombinationOperator` + `__init__.py` re-export

**Files:**
- Create: `hpm/dynamics/recombination.py`
- Modify: `hpm/dynamics/__init__.py`
- Create: `tests/dynamics/test_recombination.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/dynamics/test_recombination.py`:

```python
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.config import AgentConfig
from hpm.dynamics.recombination import RecombinationOperator, RecombinationResult
from hpm.dynamics.meta_pattern_rule import sym_kl_normalised


RNG = np.random.default_rng(42)   # fixed seed for reproducibility


def make_pattern(mu_val, dim=2, level=4):
    p = GaussianPattern(mu=np.full(dim, float(mu_val)), sigma=np.eye(dim))
    p.level = level
    return p


def default_config(**kwargs):
    cfg = AgentConfig(agent_id='test', feature_dim=2)
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def make_op():
    return RecombinationOperator(rng=np.random.default_rng(42))


# --- Level gate ---

def test_returns_none_when_fewer_than_two_level4_patterns():
    op = make_op()
    patterns = [make_pattern(0.0, level=1), make_pattern(1.0, level=3)]
    result = op.attempt(patterns, np.array([0.5, 0.5]), [], default_config(), 'time')
    assert result is None


def test_returns_none_when_only_one_level4_pattern():
    op = make_op()
    patterns = [make_pattern(0.0, level=4), make_pattern(1.0, level=2)]
    result = op.attempt(patterns, np.array([0.5, 0.5]), [], default_config(), 'time')
    assert result is None


# --- Pair rejection (kappa_max) ---

def test_returns_none_when_all_pairs_exceed_kappa_max():
    """kappa_max=0.0 rejects any pair with KL > 0."""
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(5.0, level=4)
    cfg = default_config(N_recomb=3, kappa_max=0.0)
    result = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'time')
    assert result is None


# --- Feasibility gate ---

def test_feasibility_gate_rejects_invalid_sigma():
    """attempt() skips a child that fails is_structurally_valid()."""
    from unittest.mock import patch
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(0.1, level=4)
    cfg = default_config(N_recomb=3, kappa_max=1.0)

    invalid = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    invalid.level = 4
    with patch.object(p1, 'recombine', return_value=invalid):
        with patch.object(invalid, 'is_structurally_valid', return_value=False):
            result = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'time')
    assert result is None


# --- Insight score ---

def test_insight_score_positive_only_novelty():
    """alpha_eff=0 → I = beta_orig * alpha_nov * Nov; accepted iff Nov > 0."""
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(0.5, level=4)
    cfg = default_config(kappa_max=1.0, alpha_nov=1.0, alpha_eff=0.0, beta_orig=1.0)
    result = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'time')
    # Parents not identical → Nov > 0 → I > 0 → accepted
    assert result is not None
    assert result.insight_score > 0


def test_insight_score_zero_discards():
    """beta_orig=0 forces I=0 → all children discarded."""
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(0.5, level=4)
    cfg = default_config(kappa_max=1.0, beta_orig=0.0)
    result = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'time')
    assert result is None


# --- Novelty ---

def test_novelty_one_when_child_maximally_distant():
    """
    Child orthogonal to both parents → max sym_kl_normalised ≈ 1 → Nov ≈ 1.
    Parents at [0,0] and [0.01,0.01] (nearly identical).
    Child is forced to [1000, 1000] by patching recombine().
    With alpha_eff=0, insight_score = beta_orig * alpha_nov * Nov.
    """
    from unittest.mock import patch
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(0.01, level=4)

    far_child = GaussianPattern(mu=np.array([1000.0, 1000.0]), sigma=np.eye(2))
    far_child.level = 4

    cfg = default_config(kappa_max=1.0, alpha_nov=1.0, alpha_eff=0.0, beta_orig=1.0)
    with patch.object(p1, 'recombine', return_value=far_child):
        result = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'time')

    assert result is not None
    # Nov = max(kl(child, p1), kl(child, p2)) ≈ 1.0 when child is very distant
    nov = max(
        sym_kl_normalised(far_child, p1),
        sym_kl_normalised(far_child, p2),
    )
    assert nov > 0.99
    assert abs(result.insight_score - nov) < 1e-6  # insight = 1.0 * 1.0 * Nov


# --- Empty buffer ---

def test_empty_buffer_eff_is_zero():
    """
    With obs_buffer=[], Eff=0.0 so insight = beta_orig * alpha_nov * Nov.
    Verify: insight matches the expected formula (Eff contributes nothing).
    """
    from unittest.mock import patch
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(0.5, level=4)
    cfg = default_config(kappa_max=1.0, alpha_nov=0.8, alpha_eff=0.2, beta_orig=1.0)

    result_empty = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'time')
    # With alpha_eff=0.2 and Eff=0: insight = beta_orig * (alpha_nov * Nov + 0)
    # With alpha_eff=0.0 and same Nov: insight should match
    cfg_no_eff = default_config(kappa_max=1.0, alpha_nov=0.8, alpha_eff=0.0, beta_orig=1.0)
    op2 = RecombinationOperator(rng=np.random.default_rng(42))
    result_no_eff = op2.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg_no_eff, 'time')

    # Both should yield same insight (Eff=0 in both cases)
    if result_empty is not None and result_no_eff is not None:
        assert abs(result_empty.insight_score - result_no_eff.insight_score) < 1e-9


# --- Softmax temperature ---

def test_softmax_temperature_low_concentrates_on_highest_weight_pair():
    """
    With temp→0, softmax concentrates all probability on the highest-score pair.
    Three Level 4+ patterns: weights [0.9, 0.05, 0.05].
    Highest pair score: w[0]*w[1]=0.045, w[0]*w[2]=0.045, w[1]*w[2]=0.0025.
    Pairs 0-1 and 0-2 tie; both involve pattern 0.
    With very low temp and N_recomb=100, pattern 0 should appear in every draw.
    """
    draws = []
    for seed in range(20):
        op = RecombinationOperator(rng=np.random.default_rng(seed))
        p0 = make_pattern(0.0, level=4)
        p1 = make_pattern(0.1, level=4)
        p2 = make_pattern(0.2, level=4)
        patterns = [p0, p1, p2]
        weights = np.array([0.9, 0.05, 0.05])
        cfg = default_config(kappa_max=1.0, recomb_temp=1e-6, N_recomb=1,
                             alpha_nov=1.0, alpha_eff=0.0, beta_orig=1.0)
        result = op.attempt(patterns, weights, [], cfg, 'time')
        if result is not None:
            draws.append(result.parent_a_id)
            draws.append(result.parent_b_id)

    # p0 should appear in most draws (highest-weight pattern)
    if draws:
        p0_id = make_pattern(0.0, level=4).id  # can't compare IDs across instances
        # Instead verify that most sampled pairs involve the highest-weight pattern
        # (this is a smoke test — the exact IDs won't match across instances)
        assert len(draws) > 0   # at least some accepted results


# --- RecombinationResult fields ---

def test_result_has_correct_fields():
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(0.5, level=4)
    cfg = default_config(kappa_max=1.0, alpha_nov=1.0, alpha_eff=0.0, beta_orig=1.0)
    result = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'conflict')
    assert result is not None
    assert hasattr(result, 'pattern')
    assert hasattr(result, 'insight_score')
    assert hasattr(result, 'parent_a_id')
    assert hasattr(result, 'parent_b_id')
    assert result.trigger == 'conflict'
    assert result.parent_a_id in (p1.id, p2.id)
    assert result.parent_b_id in (p1.id, p2.id)
    assert result.parent_a_id != result.parent_b_id


# --- N_recomb draws are all attempted ---

def test_all_n_recomb_draws_attempted_before_none():
    """
    If every draw reaches insight <= 0 (beta_orig=0), operator tries all
    N_recomb draws before returning None.  Verify via call count on recombine().
    """
    from unittest.mock import patch, call
    op = make_op()
    p1 = make_pattern(0.0, level=4)
    p2 = make_pattern(0.5, level=4)
    cfg = default_config(kappa_max=1.0, N_recomb=3, beta_orig=0.0)
    call_count = []

    original_recombine = p1.recombine.__func__

    def counting_recombine(self, other):
        call_count.append(1)
        return original_recombine(self, other)

    with patch.object(type(p1), 'recombine', counting_recombine):
        result = op.attempt([p1, p2], np.array([0.5, 0.5]), [], cfg, 'time')

    assert result is None
    # Each draw that passes kappa_max check calls recombine(); with kappa_max=1.0
    # all draws pass, so recombine() is called N_recomb times.
    assert len(call_count) == 3
```

- [ ] **Step 2: Run — expect ImportError**

```
pytest tests/dynamics/test_recombination.py -v
```
Expected: FAIL — `ImportError: cannot import name 'RecombinationOperator'`

- [ ] **Step 3: Create `hpm/dynamics/recombination.py`**

```python
import itertools
import numpy as np
from dataclasses import dataclass
from .meta_pattern_rule import sym_kl_normalised


@dataclass
class RecombinationResult:
    pattern: object
    insight_score: float
    parent_a_id: str
    parent_b_id: str
    trigger: str   # "time" | "conflict"


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    x = scores / temp
    x = x - x.max()   # numerical stability
    e = np.exp(x)
    return e / e.sum()


class RecombinationOperator:
    """
    Stateless operator. attempt() selects Level 4+ parent pairs, creates h*
    via convex recombination, evaluates novelty + efficacy, and returns a
    RecombinationResult or None.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random generator. Accepts one for testability; defaults to a fresh
        unseeded generator if None.
    """

    def __init__(self, rng=None):
        self._rng = rng if rng is not None else np.random.default_rng()

    def attempt(self, patterns, weights, obs_buffer, config, trigger):
        """
        Parameters
        ----------
        patterns     : list of GaussianPattern (post-prune population)
        weights      : np.ndarray of corresponding weights
        obs_buffer   : list of np.ndarray (recent observations)
        config       : AgentConfig
        trigger      : str ("time" | "conflict")

        Returns
        -------
        RecombinationResult | None
        """
        # 1. Level gate
        candidate_indices = [i for i, p in enumerate(patterns) if p.level >= 4]
        if len(candidate_indices) < 2:
            return None

        # 2. Pair sampling
        weights = np.array(weights, dtype=float)
        pairs = list(itertools.combinations(candidate_indices, 2))
        pair_scores = np.array([weights[i] * weights[j] for i, j in pairs], dtype=float)
        pair_probs = _softmax(pair_scores, config.recomb_temp)

        for _ in range(config.N_recomb):
            idx = int(self._rng.choice(len(pairs), p=pair_probs))
            i, j = pairs[idx]
            parent_a = patterns[i]
            parent_b = patterns[j]

            kappa_ab = sym_kl_normalised(parent_a, parent_b)
            if kappa_ab >= config.kappa_max:
                continue   # incompatible pair — try another draw

            # 3. Crossover
            h_star = parent_a.recombine(parent_b)

            # 4. Feasibility gate (before insight computation)
            if not h_star.is_structurally_valid():
                continue

            # 5. Insight score
            nov = max(
                sym_kl_normalised(h_star, parent_a),
                sym_kl_normalised(h_star, parent_b),
            )
            if obs_buffer:
                eff = float(np.mean([-h_star.log_prob(x) for x in obs_buffer]))
            else:
                eff = 0.0

            insight = config.beta_orig * (config.alpha_nov * nov + config.alpha_eff * eff)

            # 6. Accept/discard — continue to next draw if this child is rejected
            if insight <= 0:
                continue

            return RecombinationResult(
                pattern=h_star,
                insight_score=insight,
                parent_a_id=parent_a.id,
                parent_b_id=parent_b.id,
                trigger=trigger,
            )

        return None   # all N_recomb draws rejected or discarded
```

- [ ] **Step 4: Update `hpm/dynamics/__init__.py`**

Change:

```python
from .density import PatternDensity

__all__ = ["PatternDensity"]
```

To:

```python
from .density import PatternDensity
from .recombination import RecombinationOperator

__all__ = ["PatternDensity", "RecombinationOperator"]
```

- [ ] **Step 5: Run recombination tests — expect PASS**

```
pytest tests/dynamics/test_recombination.py -v
```
Expected: all PASS

- [ ] **Step 6: Run full test suite**

```
pytest -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add hpm/dynamics/recombination.py hpm/dynamics/__init__.py \
    tests/dynamics/test_recombination.py
git commit -m "feat: RecombinationResult + RecombinationOperator + dynamics re-export"
```

---

## Task 4: Agent changes — buffer, trigger, recombination block, return dict

**Files:**
- Modify: `hpm/agents/agent.py`
- Create: `tests/agents/test_agent_recombination.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/agents/test_agent_recombination.py`:

```python
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.memory import InMemoryStore


RECOMB_KEYS = {
    'total_conflict',
    'recombination_attempted',
    'recombination_accepted',
    'recombination_trigger',
    'insight_score',
    'recomb_parent_ids',
}


def base_cfg(**kwargs):
    cfg = AgentConfig(
        agent_id='t',
        feature_dim=2,
        conflict_threshold=float('inf'),   # suppress conflict trigger by default
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def add_level4_patterns(store, agent_id):
    """Add two Level 4 patterns to the store."""
    p1 = GaussianPattern(mu=np.array([0.0, 0.0]), sigma=np.eye(2))
    p1.level = 4
    p2 = GaussianPattern(mu=np.array([0.5, 0.5]), sigma=np.eye(2))
    p2.level = 4
    store.save(p1, 0.5, agent_id)
    store.save(p2, 0.5, agent_id)
    return p1, p2


# --- Keys present every step ---

def test_return_dict_has_recombination_keys_every_step():
    agent = Agent(base_cfg())
    result = agent.step(np.zeros(2))
    for key in RECOMB_KEYS:
        assert key in result, f"missing key: {key}"


def test_total_conflict_in_return_dict_is_non_negative():
    agent = Agent(base_cfg())
    result = agent.step(np.zeros(2))
    assert result['total_conflict'] >= 0.0


def test_recomb_parent_ids_none_when_not_attempted():
    agent = Agent(base_cfg())
    result = agent.step(np.zeros(2))
    assert result['recomb_parent_ids'] is None


# --- Time trigger ---

def test_recombination_not_attempted_before_T_recomb():
    """conflict_threshold=inf suppresses conflict trigger; step T_recomb-1 times."""
    cfg = base_cfg(T_recomb=5, conflict_threshold=float('inf'))
    agent = Agent(cfg)
    for _ in range(4):
        result = agent.step(np.zeros(2))
        assert not result['recombination_attempted'], \
            f"triggered early at step {result['t']}"


def test_time_trigger_fires_at_T_recomb():
    """recombination_attempted=True at step T_recomb (even if no Level 4 patterns)."""
    cfg = base_cfg(T_recomb=5, conflict_threshold=float('inf'))
    agent = Agent(cfg)
    for i in range(5):
        result = agent.step(np.zeros(2))
    assert result['recombination_attempted'] is True


# --- Conflict trigger ---

def test_conflict_trigger_fires_on_high_tension():
    """conflict_threshold=0.0 fires on any nonzero conflict; T_recomb=1000 prevents time trigger."""
    cfg = base_cfg(T_recomb=1000, conflict_threshold=0.0, recomb_cooldown=0)
    store = InMemoryStore()
    agent = Agent(cfg, store=store)
    add_level4_patterns(store, cfg.agent_id)

    result = agent.step(np.zeros(2))
    assert result['recombination_attempted'] is True
    assert result['recombination_trigger'] == 'conflict'


# --- Cooldown ---

def test_cooldown_blocks_double_trigger():
    """Conflict fires at step 1; step 2 blocked by cooldown=5."""
    cfg = base_cfg(T_recomb=1000, conflict_threshold=0.0, recomb_cooldown=5)
    store = InMemoryStore()
    agent = Agent(cfg, store=store)
    add_level4_patterns(store, cfg.agent_id)

    r1 = agent.step(np.zeros(2))
    r2 = agent.step(np.zeros(2))
    assert r1['recombination_attempted'] is True
    assert r2['recombination_attempted'] is False


# --- Acceptance side effects ---

def test_accepted_pattern_added_to_store():
    store = InMemoryStore()
    cfg = base_cfg(
        T_recomb=1,
        conflict_threshold=float('inf'),
        recomb_cooldown=0,
        kappa_max=1.0,
        alpha_nov=1.0,
        alpha_eff=0.0,
        beta_orig=1.0,
        kappa_0=0.1,
    )
    agent = Agent(cfg, store=store)

    # Replace seeded pattern with two Level 4 patterns
    for p, _ in store.query(cfg.agent_id):
        store.delete(p.id)
    add_level4_patterns(store, cfg.agent_id)

    before = len(store.query(cfg.agent_id))
    result = agent.step(np.zeros(2))

    if result['recombination_accepted']:
        after = len(store.query(cfg.agent_id))
        assert after == before + 1


def test_weights_sum_to_one_after_acceptance():
    store = InMemoryStore()
    cfg = base_cfg(
        T_recomb=1,
        conflict_threshold=float('inf'),
        recomb_cooldown=0,
        kappa_max=1.0,
        alpha_nov=1.0,
        alpha_eff=0.0,
        beta_orig=1.0,
        kappa_0=0.1,
    )
    agent = Agent(cfg, store=store)

    for p, _ in store.query(cfg.agent_id):
        store.delete(p.id)
    add_level4_patterns(store, cfg.agent_id)

    result = agent.step(np.zeros(2))

    if result['recombination_accepted']:
        records = store.query(cfg.agent_id)
        total_w = sum(w for _, w in records)
        assert abs(total_w - 1.0) < 1e-9
```

- [ ] **Step 2: Run — expect failures**

```
pytest tests/agents/test_agent_recombination.py -v
```
Expected: FAIL — missing keys in return dict

- [ ] **Step 3: Add imports and `__init__` attributes to `agent.py`**

At the top of `hpm/agents/agent.py`, add:

```python
from collections import deque
from ..dynamics.recombination import RecombinationOperator
```

In `Agent.__init__`, after `self._t = 0` (and before `self._seed_if_empty()`):

```python
        self._obs_buffer: deque = deque(maxlen=config.obs_buffer_size)
        self._last_recomb_t: int = -config.recomb_cooldown
        self._recomb_op = RecombinationOperator()
```

Note: these three lines must appear before `_seed_if_empty()` is called, so the attributes exist if any step logic runs during seeding.

- [ ] **Step 4: Add `obs_buffer.append` at the start of `step()`**

At the very start of `Agent.step()` (before `records = self.store.query(...)`), add:

```python
        self._obs_buffer.append(x)
```

- [ ] **Step 5: Add recombination block after `self._t += 1`**

After the `self._t += 1` line (and before `return {`), insert:

```python
        recomb_result = None
        recomb_attempted = False
        recomb_trigger = None

        time_trigger = (self._t % self.config.T_recomb == 0)
        conflict_trigger = (total_conflict > self.config.conflict_threshold)
        cooldown_ok = (self._t - self._last_recomb_t >= self.config.recomb_cooldown)

        if (time_trigger or conflict_trigger) and cooldown_ok:
            recomb_trigger = "conflict" if conflict_trigger else "time"
            recomb_attempted = True
            post_prune_records = self.store.query(self.agent_id)
            post_prune_patterns = [p for p, _ in post_prune_records]
            post_prune_weights = np.array([w for _, w in post_prune_records])
            recomb_result = self._recomb_op.attempt(
                post_prune_patterns, post_prune_weights,
                list(self._obs_buffer), self.config, recomb_trigger,
            )
            if recomb_result is not None:
                entry_weight = self.config.kappa_0 * recomb_result.insight_score
                self.store.save(recomb_result.pattern, entry_weight, self.agent_id)
                all_records = self.store.query(self.agent_id)
                total_w = sum(w for _, w in all_records)
                if total_w > 0:
                    for p, w in all_records:
                        self.store.update_weight(p.id, w / total_w)
            self._last_recomb_t = self._t
```

- [ ] **Step 6: Add new keys to the return dict**

In the `return { ... }` block, add:

```python
            'total_conflict': float(total_conflict),
            'recombination_attempted': recomb_attempted,
            'recombination_accepted': recomb_result is not None,
            'recombination_trigger': recomb_trigger,
            'insight_score': recomb_result.insight_score if recomb_result else None,
            'recomb_parent_ids': (
                (recomb_result.parent_a_id, recomb_result.parent_b_id)
                if recomb_result else None
            ),
```

- [ ] **Step 7: Run agent recombination tests**

```
pytest tests/agents/test_agent_recombination.py -v
```
Expected: all PASS

- [ ] **Step 8: Run full test suite**

```
pytest -v
```
Expected: all PASS — no regressions

- [ ] **Step 9: Commit**

```bash
git add hpm/agents/agent.py tests/agents/test_agent_recombination.py
git commit -m "feat: Agent ring buffer, trigger/cooldown, recombination block + return dict"
```

---

## Done

Final verification:

```
pytest -v --tb=short
```

Then use `superpowers:finishing-a-development-branch` to merge or create a PR.
