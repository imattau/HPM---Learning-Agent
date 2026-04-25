# Plan: HierarchicalPattern Simplification (3-Level Joint → Simple HMM)

**Date:** 2026-04-25
**Spec:** `docs/superpowers/specs/2026-04-25-hierarchical-pattern-simplification-design.md`
**Branch:** `hpm-ai-v4-dev`
**TDD approach:** Write tests first, implement to pass, then fix downstream.

---

## Task 1: Write new `HierarchicalPattern` and green tests

### 1a. Create test file first

**File:** `hpm_ai_v4/tests/test_simple_hmm_pattern.py` (CREATE)

```python
import numpy as np
import pytest
from hpm_ai_v4.pattern import HierarchicalPattern


def test_init_shapes():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert p.A.shape == (2, 2)
    assert p.B.shape == (2, 6)
    assert p.pi.shape == (2,)
    assert p.A.dtype == np.float32


def test_rows_sum_to_one():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert np.allclose(p.A.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.B.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.pi.sum(), 1.0, atol=1e-5)


def test_log_likelihood_finite():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    ll = p.log_likelihood([0, 1, 2, 0, 1])
    assert np.isfinite(ll)
    assert ll < 0


def test_update_parameters_online():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    obs = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10
    p.update_parameters_online(obs, window_size=30)
    assert np.allclose(p.A.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.B.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.pi.sum(), 1.0, atol=1e-5)


def test_get_top_state():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    state = p.get_top_state([0, 1, 0, 1])
    assert state in {0, 1}


def test_get_top_state_empty():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    state = p.get_top_state([])
    assert isinstance(state, int)
    assert 0 <= state < 2


def test_compression_nonneg():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert p.compression() >= 0


def test_compression_no_obs_arg():
    # compression() takes no obs_seq argument in new interface
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    c = p.compression()
    assert isinstance(c, float)


def test_predict_next():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    pred = p.predict_next([0, 1, 2])
    assert 0 <= pred < 6


def test_predict_next_distribution_shape():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    dist = p.predict_next_distribution([0, 1, 2])
    assert dist.shape == (6,)
    assert np.allclose(dist.sum(), 1.0, atol=1e-5)


def test_log_cache_refreshed():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert np.allclose(np.exp(p.logA), p.A, atol=1e-5)
    assert np.allclose(np.exp(p.logB), p.B, atol=1e-5)


def test_update_running_loss():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    p.update_running_loss([0, 1, 2, 0, 1])
    assert np.isfinite(p.running_loss)
    assert p.running_loss > 0


def test_warns_if_k_gt_4():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(0, latent_dim=5, obs_dim=6)
        assert len(w) == 1


def test_no_warn_k_eq_4():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(0, latent_dim=4, obs_dim=6)
        assert len(w) == 0


def test_log_likelihood_empty():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert p.log_likelihood([]) == 0.0


def test_predictive_entropy_finite():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    h = p.predictive_entropy([0, 1, 2])
    assert np.isfinite(h)
    assert h >= 0
```

**Run:** `PYTHONPATH=. pytest hpm_ai_v4/tests/test_simple_hmm_pattern.py -v`
**Expected:** all 16 tests fail (HierarchicalPattern still has old interface)

---

### 1b. Rewrite `HierarchicalPattern` in `hpm_ai_v4/pattern.py`

**Replace the entire `HierarchicalPattern` class. Keep `FlatPattern` unchanged except update its `super().__init__` call.**

```python
import numpy as np
import warnings


class HierarchicalPattern:
    """Single-level HMM. Hierarchy emerges from stacking patterns via get_top_state()."""

    def __init__(self, pattern_id, latent_dim=2, obs_dim=6):
        if latent_dim > 4:
            warnings.warn(
                f"HierarchicalPattern created with latent_dim={latent_dim} > 4. "
                "HPM architecture requires small-K (max 4). Use depth, not width.",
                stacklevel=2,
            )
        self.id = pattern_id
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.complexity = 1  # single-level; kept for DevelopmentalStage compatibility

        self.A = np.random.dirichlet(np.ones(latent_dim), size=latent_dim).astype(np.float32)
        self.B = np.random.dirichlet(np.ones(obs_dim), size=latent_dim).astype(np.float32)
        self.pi = np.random.dirichlet(np.ones(latent_dim)).astype(np.float32)

        self.running_loss = 0.0
        self.weight = 1.0
        self.creation_step = 0

        self._refresh_log_cache()

    def _refresh_log_cache(self):
        self.logA = np.log(self.A + 1e-12).astype(np.float32)
        self.logB = np.log(self.B + 1e-12).astype(np.float32)

    def _forward(self, obs_seq):
        """Scaled forward pass. Returns alpha (T, K) and scales (T,)."""
        T, K = len(obs_seq), self.latent_dim
        alpha = np.zeros((T, K), dtype=np.float32)
        alpha[0] = self.pi * self.B[:, int(obs_seq[0]) % self.obs_dim]
        s = alpha[0].sum()
        alpha[0] /= s + 1e-12
        scales = [s]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, int(obs_seq[t]) % self.obs_dim]
            s = alpha[t].sum()
            alpha[t] /= s + 1e-12
            scales.append(s)
        return alpha, np.array(scales, dtype=np.float32)

    def _forward_backward(self, obs_seq):
        """Standard Baum-Welch. Returns gamma (T, K) and xi (T-1, K, K)."""
        T, K = len(obs_seq), self.latent_dim
        alpha, scales = self._forward(obs_seq)

        beta = np.ones((T, K), dtype=np.float32)
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * (beta[t + 1] * self.B[:, int(obs_seq[t + 1]) % self.obs_dim])).sum(axis=1)
            s = beta[t].sum()
            beta[t] /= s + 1e-12

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12

        xi = np.zeros((T - 1, K, K), dtype=np.float32)
        for t in range(T - 1):
            obs_next = int(obs_seq[t + 1]) % self.obs_dim
            xi[t] = alpha[t:t + 1].T * self.A * (beta[t + 1] * self.B[:, obs_next])
            xi[t] /= xi[t].sum() + 1e-12

        return gamma, xi

    def log_likelihood(self, obs_seq):
        """Compute log p(x_{1:T}) via scaled forward algorithm."""
        if len(obs_seq) == 0:
            return 0.0
        _, scales = self._forward(obs_seq)
        return float(np.sum(np.log(scales + 1e-12)))

    def update_parameters_online(self, obs_seq, window_size=100):
        """Windowed Baum-Welch EM update with exponential smoothing."""
        if len(obs_seq) < 5:
            return
        window = list(obs_seq[-window_size:])
        gamma, xi = self._forward_backward(window)
        K = self.latent_dim

        new_A = xi.sum(axis=0)
        new_B = np.zeros((K, self.obs_dim), dtype=np.float32)
        for t, o in enumerate(window):
            new_B[:, int(o) % self.obs_dim] += gamma[t]
        new_pi = gamma[0].copy()

        # Normalise
        new_A /= new_A.sum(axis=1, keepdims=True) + 1e-12
        new_B /= new_B.sum(axis=1, keepdims=True) + 1e-12
        new_pi /= new_pi.sum() + 1e-12

        # Exponential smoothing blend
        alpha = 0.7
        self.A = (alpha * new_A + (1 - alpha) * self.A).astype(np.float32)
        self.B = (alpha * new_B + (1 - alpha) * self.B).astype(np.float32)
        self.pi = (alpha * new_pi + (1 - alpha) * self.pi).astype(np.float32)

        # Re-normalise
        self.A /= self.A.sum(axis=1, keepdims=True) + 1e-12
        self.B /= self.B.sum(axis=1, keepdims=True) + 1e-12
        self.pi /= self.pi.sum() + 1e-12

        self._refresh_log_cache()

    def update_running_loss(self, obs_seq, lambda_l=0.1):
        if len(obs_seq) == 0:
            return
        ll = self.log_likelihood(obs_seq[-30:])
        loss = -ll / max(1, len(obs_seq[-30:]))
        self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * loss

    def get_top_state(self, obs_seq):
        """Most probable current latent state given obs_seq."""
        if not obs_seq:
            return int(np.argmax(self.pi))
        alpha, _ = self._forward(obs_seq[-20:])
        return int(np.argmax(alpha[-1]))

    def predict_next(self, obs_seq):
        """Argmax of predictive distribution over next observation."""
        return int(np.argmax(self.predict_next_distribution(obs_seq)))

    def predict_next_distribution(self, obs_seq):
        """Predictive distribution P(x_{t+1} | x_{1:t}) as (obs_dim,) array."""
        if not obs_seq:
            next_state = self.pi @ self.A
        else:
            alpha, _ = self._forward(obs_seq[-20:])
            next_state = alpha[-1] @ self.A
        pred = next_state @ self.B
        pred /= pred.sum() + 1e-12
        return pred.astype(np.float32)

    def compression(self):
        """
        Mutual information I(z_t; z_{t+1}) under stationary distribution of A.
        Measures how much the transition structure compresses state uncertainty.
        No obs_seq argument — computed analytically from A.
        """
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        idx = np.where(np.isclose(np.real(eigvals), 1.0))[0]
        if len(idx) == 0:
            stat = np.ones(self.latent_dim, dtype=np.float32) / self.latent_dim
        else:
            stat = np.abs(np.real(eigvecs[:, idx[0]])).astype(np.float32)
            stat /= stat.sum() + 1e-12

        H_state = float(-np.sum(stat * np.log(stat + 1e-12)))
        H_cond = float(sum(
            stat[i] * (-np.sum(self.A[i] * np.log(self.A[i] + 1e-12)))
            for i in range(self.latent_dim)
        ))
        return max(0.0, H_state - H_cond)

    def predictive_entropy(self, obs_seq):
        """Entropy of the predictive distribution over the next observation."""
        pred = self.predict_next_distribution(obs_seq)
        return float(-np.sum(pred * np.log(pred + 1e-12)))
```

**Note on `FlatPattern`:** `FlatPattern` inherits from `HierarchicalPattern`. Its `__init__` calls `super().__init__(pattern_id, latent_dim=1, obs_dim=obs_dim)`. After the rewrite this still works correctly — `A=(1,1)`, `B=(1,obs_dim)`, `pi=(1,)`. `FlatPattern.complexity` stays at 1. No changes needed to `FlatPattern` body except removing any direct references to `A3`/`pi3` if present (there are none in current code).

**Run:** `PYTHONPATH=. pytest hpm_ai_v4/tests/test_simple_hmm_pattern.py -v`
**Expected:** all 16 tests pass

**Commit:** `refactor: replace 3-level joint HMM with simple single-level HMM in HierarchicalPattern`

---

## Task 2: Fix downstream breakage in existing tests

**Run:** `PYTHONPATH=. pytest hpm_ai_v4/tests/ -v 2>&1 | head -80`

### Expected failures and fixes

#### `test_pattern_growth.py`
Tests reference `A3`, `A32`, `A21`, `pi3`, `SS_A3`, `grow_latent`. All must be rewritten.

**Replace entire file with:**

```python
import pytest
import numpy as np
import warnings
from hpm_ai_v4.pattern import HierarchicalPattern


@pytest.fixture
def pattern():
    np.random.seed(42)
    return HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)


def test_A_shape(pattern):
    assert pattern.A.shape == (2, 2)


def test_B_shape(pattern):
    assert pattern.B.shape == (2, 5)


def test_pi_shape(pattern):
    assert pattern.pi.shape == (2,)


def test_A_rows_sum_to_one(pattern):
    np.testing.assert_allclose(pattern.A.sum(axis=1), np.ones(2), atol=1e-6)


def test_B_rows_sum_to_one(pattern):
    np.testing.assert_allclose(pattern.B.sum(axis=1), np.ones(2), atol=1e-6)


def test_pi_sums_to_one(pattern):
    assert abs(pattern.pi.sum() - 1.0) < 1e-6


def test_A_dtype(pattern):
    assert pattern.A.dtype == np.float32


def test_log_likelihood_finite_after_init(pattern):
    ll = pattern.log_likelihood([0, 1, 2, 3, 4, 0, 1])
    assert np.isfinite(ll)


def test_get_top_state_returns_valid_int():
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    result = p.get_top_state([0, 1, 0, 2, 1])
    assert isinstance(result, int)
    assert result in {0, 1}


def test_get_top_state_empty_returns_int():
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    result = p.get_top_state([])
    assert isinstance(result, int)


def test_get_top_state_k3():
    p = HierarchicalPattern(pattern_id=0, latent_dim=3, obs_dim=5)
    result = p.get_top_state([0, 1, 2, 0, 1])
    assert result in {0, 1, 2}


def test_constructor_warns_if_k_gt_4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(pattern_id=0, latent_dim=5, obs_dim=5)
        assert len(w) == 1
        assert "latent_dim" in str(w[0].message).lower() or "K" in str(w[0].message)


def test_constructor_no_warn_k_eq_4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(pattern_id=0, latent_dim=4, obs_dim=5)
        assert len(w) == 0
```

#### `test_fast_learning.py`
Tests for `A3`, `A32`, `A21`, `pi3`, `update_parameters_online_fast`, `maybe_update`, `obs_chunk` are all obsolete. Replace with tests for the new interface:

```python
import numpy as np
import pytest
from hpm_ai_v4.pattern import HierarchicalPattern


class TestFloat32Storage:
    def test_A_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.A.dtype == np.float32

    def test_B_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.B.dtype == np.float32

    def test_pi_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.pi.dtype == np.float32

    def test_logA_shape(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.logA.shape == (2, 2)

    def test_logA_dtype(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.logA.dtype == np.float32

    def test_logA_consistent_with_A(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert np.allclose(np.exp(p.logA), p.A, atol=1e-5)

    def test_logB_consistent_with_B(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert np.allclose(np.exp(p.logB), p.B, atol=1e-5)

    def test_log_cache_refreshed_after_update(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        obs_seq = [0, 1, 2, 1, 0, 1, 2, 3, 4, 0] * 5
        p.update_parameters_online(obs_seq)
        assert np.allclose(np.exp(p.logA), p.A, atol=1e-4)


class TestOnlineUpdate:
    def setup_method(self):
        np.random.seed(42)
        self.p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        self.obs_seq = [0, 1, 2, 1, 0, 3, 4, 2, 1, 0] * 10  # length 100

    def test_runs_without_error(self):
        self.p.update_parameters_online(self.obs_seq)

    def test_A_rows_sum_to_one(self):
        self.p.update_parameters_online(self.obs_seq)
        assert np.allclose(self.p.A.sum(axis=1), 1.0, atol=1e-5)

    def test_B_rows_sum_to_one(self):
        self.p.update_parameters_online(self.obs_seq)
        assert np.allclose(self.p.B.sum(axis=1), 1.0, atol=1e-5)

    def test_pi_sums_to_one(self):
        self.p.update_parameters_online(self.obs_seq)
        assert np.allclose(self.p.pi.sum(), 1.0, atol=1e-5)

    def test_log_likelihood_finite_after_updates(self):
        for _ in range(10):
            self.p.update_parameters_online(self.obs_seq)
        ll = self.p.log_likelihood(self.obs_seq)
        assert np.isfinite(ll)

    def test_running_loss_updates(self):
        initial = self.p.running_loss
        self.p.update_running_loss(self.obs_seq[:10])
        assert self.p.running_loss != initial


class TestAgentFastPath:
    def setup_method(self):
        np.random.seed(7)
        from hpm_ai_v4.agents.agent import HPMAgent
        self.agent = HPMAgent(num_initial_patterns=3, obs_dim=5)

    def test_perceive_and_learn_runs(self):
        self.agent.perceive_and_learn(0)

    def test_200_steps_without_error(self):
        for i in range(200):
            self.agent.perceive_and_learn(i % 5)

    def test_pattern_weights_positive_after_200_steps(self):
        for i in range(200):
            self.agent.perceive_and_learn(i % 5)
        weights = np.array([p.weight for p in self.agent.patterns])
        assert weights.sum() > 0.1
```

#### `test_reasoning.py`
Check for references to `get_belief`, `A3`, `A32`, `A21`. The `simulate()` method in `Reasoner` uses all three — it must be rewritten (see Task 3).

#### `test_full_stack.py`, `test_learning_dynamics.py`, `test_meta_layer.py`
Scan for `A3`, `pi3`, `SS_A3`, `compression(obs_seq)`, `maybe_update`, `obs_chunk`. Fix each:
- `compression(obs_seq)` → `compression()`
- `p.A3` → `p.A`
- `p.pi3` → `p.pi`
- Remove `SS_A3`, `SS_A32`, `SS_A21` references

#### `metrics.py` — fix `affective_score` and `pattern_density`

```python
# affective_score: change compression(obs_seq) -> compression()
comp = pattern.compression()   # was: pattern.compression(obs_seq)

# pattern_density: update n_params formula
n_params = pattern.latent_dim**2 + pattern.latent_dim * pattern.obs_dim + pattern.latent_dim
```

#### `agent.py` — fix `_maybe_grow_patterns` and `perceive_and_learn`

In `_maybe_grow_patterns`:
```python
# change:
compression = p.compression(self.obs_buffer)
# to:
compression = p.compression()
```

In `perceive_and_learn`, the parallel worker writes back named parameters. Update `result_by_id` write-back:
```python
# Remove:
p.A3 = r['A3']; p.A32 = r['A32']; p.A21 = r['A21']; p.pi3 = r['pi3']
p.SS_A3 = r['SS_A3']; p.SS_A32 = r['SS_A32']; p.SS_A21 = r['SS_A21']; p.SS_B = r['SS_B']
# Add:
p.A = r['A']; p.B = r['B']; p.pi = r['pi']
p.running_loss = r['running_loss']
```

Also remove the `maybe_update` chunk-buffering call (lines 157-159 in current `agent.py`).

#### `operators/parallel.py` — fix worker serialisation

The parallel worker returns a dict of named parameter arrays. Update to return `A`, `B`, `pi` instead of `A3`, `A32`, `A21`, `pi3`, `SS_*`.

#### `reasoning.py` — fix `simulate()` and `explain()`

```python
def simulate(self, pattern, initial_obs_seq, steps=10):
    """Generate future sequence using single-level HMM."""
    if initial_obs_seq:
        alpha, _ = pattern._forward(initial_obs_seq[-20:])
        state_dist = alpha[-1]
    else:
        state_dist = pattern.pi.copy()

    simulated = []
    K = pattern.latent_dim
    for _ in range(steps):
        # Sample current state
        z = np.random.choice(K, p=state_dist / (state_dist.sum() + 1e-12))
        # Sample observation
        obs_probs = pattern.B[z]
        next_obs = np.random.choice(pattern.obs_dim, p=obs_probs / (obs_probs.sum() + 1e-12))
        simulated.append(int(next_obs))
        # Transition
        state_dist = pattern.A[z]
    return simulated

def explain(self, pattern):
    likely_obs = np.argmax(pattern.B[np.argmax(pattern.pi)])
    return (f"Hierarchical pattern (ID:{pattern.id}) predicts observation {likely_obs} "
            f"via {pattern.latent_dim} latent states.")
```

**Run:** `PYTHONPATH=. pytest hpm_ai_v4/tests/ -v`
**Expected:** all tests pass

**Commit:** `fix: update tests and downstream code after HierarchicalPattern simplification`

---

## Task 3: Verify simulation still runs end-to-end

**Run:** `PYTHONPATH=. python3 hpm_ai_v4/simulations/hpm_ai_simulation_1.py`

**Expected:** runs to completion without error, hierarchical pattern weight rises above flat pattern.

If the simulation file references removed attributes (`A3`, `pi3`, etc.), patch them inline.

**Commit (only if changes made):** `fix: update simulation for simplified HierarchicalPattern`

---

## Task 4: Final check

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/ -v --tb=short 2>&1 | tail -20
```

All tests green. No references to `A3`, `A32`, `A21`, `pi3`, `SS_A3` remain outside `FlatPattern` (which inherits cleanly).

**Commit both spec and plan documents:**
```bash
git add docs/superpowers/specs/2026-04-25-hierarchical-pattern-simplification-design.md \
        docs/superpowers/plans/2026-04-25-hierarchical-pattern-simplification.md
git commit -m "feat: add HierarchicalPattern simplification spec and plan (3-level joint → simple HMM)"
```

---

## Summary of file changes

| File | Action |
|---|---|
| `hpm_ai_v4/pattern.py` | REWRITE `HierarchicalPattern`; `FlatPattern` unchanged |
| `hpm_ai_v4/tests/test_simple_hmm_pattern.py` | CREATE (16 new tests) |
| `hpm_ai_v4/tests/test_pattern_growth.py` | REWRITE (remove grow_latent, A3/A32/A21 tests) |
| `hpm_ai_v4/tests/test_fast_learning.py` | REWRITE (remove chunked EM tests, add online update tests) |
| `hpm_ai_v4/evaluators/metrics.py` | PATCH (compression() call, n_params formula) |
| `hpm_ai_v4/agents/agent.py` | PATCH (write-back dict keys, remove maybe_update call, fix compression() call) |
| `hpm_ai_v4/operators/parallel.py` | PATCH (worker return dict: A/B/pi not A3/A32/A21/pi3/SS_*) |
| `hpm_ai_v4/agents/reasoning.py` | PATCH (simulate(), explain()) |
| Other test files | PATCH as needed (remove A3/pi3/SS_ references, fix compression() call signature) |
