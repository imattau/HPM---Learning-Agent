# Fast Online Learning — Implementation Plan

**Date:** 2026-04-25
**Spec:** `docs/superpowers/specs/2026-04-25-fast-online-learning-design.md`
**Branch:** hpm-ai-v4-dev
**Approach:** TDD — write failing tests first, then implement

---

## Task 1: float32 Storage + Log Cache

### File
`hpm_ai_v4/pattern.py` — MODIFY

### Failing tests (write first)

File: `hpm_ai_v4/tests/test_fast_learning.py` — CREATE

```python
import numpy as np
import pytest
from hpm_ai_v4.pattern import HierarchicalPattern


class TestFloat32Storage:
    def test_A3_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.A3.dtype == np.float32, f"Expected float32, got {p.A3.dtype}"

    def test_A32_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.A32.dtype == np.float32

    def test_A21_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.A21.dtype == np.float32

    def test_B_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.B.dtype == np.float32

    def test_pi3_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.pi3.dtype == np.float32

    def test_logA3_shape(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.logA3.shape == (2, 2)

    def test_logA3_dtype(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.logA3.dtype == np.float32

    def test_logA3_consistent_with_A3(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert np.allclose(np.exp(p.logA3), p.A3, atol=1e-5)

    def test_logB_consistent_with_B(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert np.allclose(np.exp(p.logB), p.B, atol=1e-5)

    def test_log_cache_refreshed_after_adapt(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        obs_seq = [0, 1, 2, 1, 0, 1, 2, 3, 4, 0] * 5
        p.adapt(obs_seq)
        # After adapt, logA3 should match A3 (not stale)
        assert np.allclose(np.exp(p.logA3), p.A3, atol=1e-4)

    def test_log_cache_refreshed_after_reestimate(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        p.SS_A3 *= 2.0
        p._reestimate_parameters()
        assert np.allclose(np.exp(p.logA3), p.A3, atol=1e-4)
```

### Run command
```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_fast_learning.py::TestFloat32Storage -v
```

### Expected failure
```
AttributeError: 'HierarchicalPattern' object has no attribute 'logA3'
# and dtype assertions fail: Expected float32, got float64
```

### Implementation

In `hpm_ai_v4/pattern.py`, `HierarchicalPattern.__init__()`:

1. Add `dtype=np.float32` to all `np.random.dirichlet` calls and cast results:
```python
self.A3  = np.random.dirichlet(np.ones(latent_dim) * 0.1, size=latent_dim).astype(np.float32)
self.pi3 = np.random.dirichlet(np.ones(latent_dim) * 0.1).astype(np.float32)
self.A32 = np.random.dirichlet(np.ones(latent_dim) * 0.1, size=latent_dim).astype(np.float32)
self.A21 = np.random.dirichlet(np.ones(latent_dim) * 0.1, size=latent_dim).astype(np.float32)
self.B   = np.random.dirichlet(np.ones(obs_dim) * 0.1, size=latent_dim).astype(np.float32)
```

2. Cast sufficient statistics to float32:
```python
self.SS_A3  = (np.ones((latent_dim, latent_dim)) * 0.1).astype(np.float32)
self.SS_A32 = (np.ones((latent_dim, latent_dim)) * 0.1).astype(np.float32)
self.SS_A21 = (np.ones((latent_dim, latent_dim)) * 0.1).astype(np.float32)
self.SS_B   = (np.ones((latent_dim, obs_dim)) * 0.1).astype(np.float32)
```

3. Add `_refresh_log_cache()` method:
```python
def _refresh_log_cache(self) -> None:
    """Recompute cached log-parameter arrays from current A3, A32, A21, B, pi3."""
    self.logA3  = np.log(self.A3  + 1e-7).astype(np.float32)
    self.logA32 = np.log(self.A32 + 1e-7).astype(np.float32)
    self.logA21 = np.log(self.A21 + 1e-7).astype(np.float32)
    self.logB   = np.log(self.B   + 1e-7).astype(np.float32)
    self.logPi3 = np.log(self.pi3 + 1e-7).astype(np.float32)
```

4. Call `self._refresh_log_cache()` at the end of `__init__()`.

5. Call `self._refresh_log_cache()` at the end of `_reestimate_parameters()`.

6. Call `self._refresh_log_cache()` at the end of `grow_latent()` (after all array expansions).

### Commit
```
perf: add float32 storage and log cache to HierarchicalPattern
```

---

## Task 2: Forward-Filter-Only EM

### File
`hpm_ai_v4/pattern.py` — MODIFY (add new methods)

### Failing tests (append to `test_fast_learning.py`)

```python
class TestForwardFilterEM:
    def setup_method(self):
        np.random.seed(42)
        self.p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        self.obs_seq = ([0, 1, 2, 1, 0, 3, 4, 2, 1, 0] * 10)  # length 100

    def test_runs_without_error(self):
        self.p.update_parameters_online_fast(self.obs_seq)

    def test_A3_rows_sum_to_one(self):
        self.p.update_parameters_online_fast(self.obs_seq)
        row_sums = self.p.A3.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"A3 row sums: {row_sums}"

    def test_A32_rows_sum_to_one(self):
        self.p.update_parameters_online_fast(self.obs_seq)
        row_sums = self.p.A32.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_A21_rows_sum_to_one(self):
        self.p.update_parameters_online_fast(self.obs_seq)
        row_sums = self.p.A21.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_B_rows_sum_to_one(self):
        self.p.update_parameters_online_fast(self.obs_seq)
        row_sums = self.p.B.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_pi3_sums_to_one(self):
        self.p.update_parameters_online_fast(self.obs_seq)
        assert np.allclose(self.p.pi3.sum(), 1.0, atol=1e-5)

    def test_log_likelihood_non_trivial(self):
        # After 10 updates on the same seq, log_likelihood should be finite
        for _ in range(10):
            self.p.update_parameters_online_fast(self.obs_seq)
        ll = self.p.log_likelihood(self.obs_seq)
        assert np.isfinite(ll), f"log_likelihood not finite: {ll}"
        assert ll > -np.inf

    def test_log_cache_consistent_after_fast_update(self):
        self.p.update_parameters_online_fast(self.obs_seq)
        assert np.allclose(np.exp(self.p.logA3), self.p.A3, atol=1e-4)
```

### Run command
```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_fast_learning.py::TestForwardFilterEM -v
```

### Expected failure
```
AttributeError: 'HierarchicalPattern' object has no attribute 'update_parameters_online_fast'
```

### Implementation

Add two methods to `HierarchicalPattern` in `hpm_ai_v4/pattern.py`:

```python
def _forward_filter(self, obs_seq):
    """
    Forward-filter pass on the 3-level HMM.
    Returns alpha_norm: (T, K, K, K) normalised filtered posteriors,
    and scales: (T,) normalisation constants.
    Uses cached log arrays if available, else recomputes.
    """
    K = self.latent_dim
    T = len(obs_seq)

    # Use cached log arrays
    l_pi3 = self.logPi3
    l_A3  = self.logA3
    l_A32 = self.logA32
    l_A21 = self.logA21
    l_B   = self.logB

    alpha = np.full((T, K, K, K), -np.inf, dtype=np.float32)

    alpha[0] = (l_pi3[:, None, None] +
                l_A32[:, :, None] +
                l_A21[None, :, :] +
                l_B[:, int(obs_seq[0]) % self.obs_dim][None, None, :])

    for t in range(1, T):
        l_pz3_prev = logsumexp(alpha[t-1], axis=(1, 2))
        l_pz3_curr = logsumexp(l_pz3_prev[:, None] + l_A3, axis=0)
        alpha[t] = (l_pz3_curr[:, None, None] +
                    l_A32[:, :, None] +
                    l_A21[None, :, :] +
                    l_B[:, int(obs_seq[t]) % self.obs_dim][None, None, :])

    # Normalise each alpha[t] to get filtered posterior
    scales = np.zeros(T, dtype=np.float32)
    alpha_norm = np.zeros_like(alpha)
    for t in range(T):
        scale_t = float(logsumexp(alpha[t]))
        scales[t] = scale_t
        alpha_norm[t] = np.exp(alpha[t] - scale_t).astype(np.float32)

    return alpha_norm, scales


def update_parameters_online_fast(self, obs_seq) -> None:
    """
    Forward-filter-only EM update (Technique A).
    Uses filtered posteriors instead of smoothed posteriors.
    Approximation is tight for long sequences.
    Keeps adapt() (full FB) for offline/validation use.
    """
    if self.complexity < 2:
        return
    T = len(obs_seq)
    if T < 2:
        return

    K = self.latent_dim
    gamma_f, scales = self._forward_filter(obs_seq)  # (T, K, K, K)

    # Decay existing sufficient statistics
    self.SS_A3  *= self.statistics_decay
    self.SS_A32 *= self.statistics_decay
    self.SS_A21 *= self.statistics_decay
    self.SS_B   *= self.statistics_decay

    # Emission and cross-level statistics from gamma_f
    # SS_A32[i,j] += sum_t P(z3=i, z2=j | x_{0:t})  (marginalise z1)
    self.SS_A32 += np.sum(gamma_f, axis=3).sum(axis=0) * 2.0  # sum over T, z1

    # SS_A21[j,k] += sum_t P(z2=j, z1=k | x_{0:t})  (marginalise z3)
    self.SS_A21 += np.sum(gamma_f, axis=1).sum(axis=0) * 2.0  # sum over T, z3

    # SS_B[k, x_t] += sum_t P(z1=k | x_{0:t})
    gamma_z1 = np.sum(gamma_f, axis=(1, 2))  # (T, K)
    for t in range(T):
        self.SS_B[:, int(obs_seq[t]) % self.obs_dim] += gamma_z1[t] * 5.0

    # One-step xi for A3: xi[t,i,j] ~ P(z3_t=i)*A3[i,j]*P(z3_{t+1}=j)
    # Using filtered alpha marginalised over z2, z1
    gamma_z3 = np.sum(gamma_f, axis=(2, 3))  # (T, K)
    for t in range(T - 1):
        # outer product approximation: xi[i,j] ~ alpha_z3[t,i] * A3[i,j] * alpha_z3[t+1,j]
        xi_t = (gamma_z3[t, :, None] *
                self.A3 *
                gamma_z3[t + 1, None, :])
        xi_sum = xi_t.sum() + 1e-7
        self.SS_A3 += (xi_t / xi_sum) * 5.0

    self._reestimate_parameters()
    # _reestimate_parameters calls _refresh_log_cache
```

### Commit
```
perf: add forward-filter-only EM as update_parameters_online_fast()
```

---

## Task 3: Fixed-Chunk Buffering

### File
`hpm_ai_v4/pattern.py` — MODIFY

### Failing tests (append to `test_fast_learning.py`)

```python
class TestFixedChunkBuffering:
    def setup_method(self):
        np.random.seed(0)
        self.p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)

    def test_obs_chunk_starts_empty(self):
        assert self.p.obs_chunk == []

    def test_chunk_size_default(self):
        assert self.p.chunk_size == 100

    def test_no_em_before_chunk_full(self):
        A3_before = self.p.A3.copy()
        full_buf = list(range(5)) * 20
        for i in range(99):
            self.p.maybe_update(i % 5, full_buf)
        # A3 should be unchanged (no EM run yet)
        assert np.allclose(self.p.A3, A3_before, atol=1e-6), \
            "A3 changed before chunk was full"

    def test_em_runs_at_chunk_boundary(self):
        A3_before = self.p.A3.copy()
        full_buf = list(range(5)) * 20
        for i in range(100):
            self.p.maybe_update(i % 5, full_buf)
        # A3 should have changed after 100th call
        assert not np.allclose(self.p.A3, A3_before, atol=1e-6), \
            "A3 unchanged after chunk boundary — EM did not run"

    def test_obs_chunk_clears_after_em(self):
        full_buf = list(range(5)) * 20
        for i in range(100):
            self.p.maybe_update(i % 5, full_buf)
        assert self.p.obs_chunk == [], \
            f"obs_chunk not cleared after EM: {self.p.obs_chunk}"

    def test_running_loss_updated_every_step(self):
        losses = []
        full_buf = [0, 1, 2, 3, 4] * 20
        initial_loss = self.p.running_loss
        for i in range(10):
            self.p.maybe_update(i % 5, full_buf)
            losses.append(self.p.running_loss)
        # running_loss should change each step (not just at chunk boundaries)
        assert losses[0] != initial_loss, "running_loss not updated at step 1"
        assert len(set(losses)) > 1, "running_loss never changed across 10 steps"
```

### Run command
```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_fast_learning.py::TestFixedChunkBuffering -v
```

### Expected failure
```
AttributeError: 'HierarchicalPattern' object has no attribute 'obs_chunk'
```

### Implementation

In `HierarchicalPattern.__init__()`, add after existing attributes:
```python
# Fixed-chunk EM buffering (Technique B)
self.obs_chunk: list = []
self.chunk_size: int = 100
```

Add two methods:

```python
def _update_loss_one_step(self, obs: int) -> None:
    """
    Cheap one-step loss update using a single forward step from the last
    alpha state (or from prior if no history). O(K^3) per call.
    Updates self.running_loss without running full forward-backward.
    """
    # Approximate: use predict_next_distribution and cross-entropy
    if not self.obs_chunk:
        # Use prior prediction
        pred = self.predict_next_distribution([obs])
    else:
        pred = self.predict_next_distribution(self.obs_chunk[-5:] + [obs])
    idx = int(obs) % self.obs_dim
    log_p = float(np.log(pred[idx] + 1e-7))
    loss = -log_p
    lambda_l = 0.1
    self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * loss


def maybe_update(self, obs: int, full_buffer: list) -> None:
    """
    Per-step update entry point (Technique B: fixed-chunk EM).

    Every call:
      - Updates running_loss via cheap one-step prediction (O(K^3))

    Every chunk_size calls:
      - Runs update_parameters_online_fast() on buffered observations
      - Clears the chunk buffer
    """
    self._update_loss_one_step(obs)
    self.obs_chunk.append(obs)
    if len(self.obs_chunk) >= self.chunk_size:
        self.update_parameters_online_fast(self.obs_chunk)
        self.obs_chunk.clear()
```

### Commit
```
perf: add fixed-chunk EM buffering — run full EM every 100 steps
```

---

## Task 4: Wire into HPMAgent.perceive_and_learn()

### File
`hpm_ai_v4/agents/agent.py` — MODIFY

### Failing tests (append to `test_fast_learning.py`)

```python
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

    def test_pattern_weights_approximately_sum_to_one_after_200_steps(self):
        for i in range(200):
            self.agent.perceive_and_learn(i % 5)
        weights = np.array([p.weight for p in self.agent.patterns])
        total = weights.sum()
        assert total > 0.1, f"Total weight collapsed to {total}"

    def test_obs_chunk_used_between_boundaries(self):
        # After fewer than chunk_size steps, at least one pattern has a non-empty chunk
        for i in range(50):
            self.agent.perceive_and_learn(i % 5)
        hierarchical = [p for p in self.agent.patterns if p.complexity >= 2]
        if hierarchical:
            chunk_lens = [len(p.obs_chunk) for p in hierarchical]
            # At least one pattern should have buffered some observations
            assert max(chunk_lens) > 0, \
                f"No pattern has buffered observations after 50 steps: {chunk_lens}"
```

### Run command
```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_fast_learning.py::TestAgentFastPath -v
```

### Expected failure (before implementation)
```
# Tests likely pass already for basic run, but obs_chunk test will fail
# because worker pool writes back A3/A32 etc but not obs_chunk state
AssertionError: No pattern has buffered observations after 50 steps
```

### Implementation

The parallel worker pool (`operators/parallel.py`) serialises pattern state into dicts and writes back `A3`, `A32`, `A21`, `B`, `pi3`, `SS_*`, `running_loss` but not `obs_chunk`. The fast path should run `maybe_update` on the main-process pattern objects **after** the worker pool writes back state.

In `perceive_and_learn()`, after the write-back loop (step 3), add:

```python
# Fast-path: chunk-buffered EM update for each hierarchical pattern
for p in self.patterns:
    if p.complexity >= 2:
        p.maybe_update(obs, self.obs_buffer)
```

This replaces the worker pool's `adapt()` call for EM — the worker pool can be simplified to only compute scores (not run EM) in a future refactor. For now the two paths coexist; `maybe_update` runs in addition. The chunk-based path will dominate learning signal as `obs_chunk` fills.

Note: a follow-on task should refactor the worker pool to skip `adapt()` when the fast path is enabled, to fully realise the speedup. For this task, correctness (tests green) is the goal.

### Commit
```
perf: wire fast online learning into HPMAgent.perceive_and_learn()
```

---

## Task 5: Benchmark

### File
`hpm_ai_v4/simulations/benchmark_fast_learning.py` — CREATE

### No test — prints timing comparison

```python
"""
Benchmark: fast online learning vs baseline.

Usage:
    PYTHONPATH=. python hpm_ai_v4/simulations/benchmark_fast_learning.py
"""
import time
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern

N_STEPS = 1000
N_PATTERNS = 10
OBS_DIM = 10
K = 2
CHUNK = 100

rng = np.random.default_rng(42)
obs_stream = rng.integers(0, OBS_DIM, size=N_STEPS).tolist()

# --- Baseline: adapt() every step ---
patterns_baseline = [HierarchicalPattern(i, latent_dim=K, obs_dim=OBS_DIM)
                     for i in range(N_PATTERNS)]
buffer = []
t0 = time.perf_counter()
for step, obs in enumerate(obs_stream):
    buffer.append(obs)
    if len(buffer) > 100:
        buffer = buffer[-100:]
    for p in patterns_baseline:
        if len(buffer) >= 2:
            p.adapt(buffer)
        p.update_running_loss(buffer)
baseline_time = time.perf_counter() - t0

# --- Fast path: maybe_update() every step ---
patterns_fast = [HierarchicalPattern(i, latent_dim=K, obs_dim=OBS_DIM)
                 for i in range(N_PATTERNS)]
buffer2 = []
t0 = time.perf_counter()
for step, obs in enumerate(obs_stream):
    buffer2.append(obs)
    if len(buffer2) > 100:
        buffer2 = buffer2[-100:]
    for p in patterns_fast:
        p.maybe_update(obs, buffer2)
fast_time = time.perf_counter() - t0

print(f"Baseline ({N_STEPS} steps, {N_PATTERNS} patterns): {baseline_time:.3f}s")
print(f"Fast path ({N_STEPS} steps, {N_PATTERNS} patterns): {fast_time:.3f}s")
if fast_time > 0:
    print(f"Speedup: {baseline_time / fast_time:.1f}x")
else:
    print("Fast path too fast to measure precisely")
```

### Run command
```bash
PYTHONPATH=. python hpm_ai_v4/simulations/benchmark_fast_learning.py
```

### Commit
```
perf: add benchmark for fast online learning
```

---

## Final commit (both documents)

```bash
git add docs/superpowers/specs/2026-04-25-fast-online-learning-design.md \
        docs/superpowers/plans/2026-04-25-fast-online-learning.md
git commit -m "feat: add fast online learning spec and plan (filter-only EM + chunked updates + float32)"
```

---

## Summary of Changes

| Task | File | Type | Commit message |
|------|------|------|----------------|
| 1 | `hpm_ai_v4/pattern.py` | MODIFY | `perf: add float32 storage and log cache to HierarchicalPattern` |
| 1 | `hpm_ai_v4/tests/test_fast_learning.py` | CREATE | (included in task 1 commit) |
| 2 | `hpm_ai_v4/pattern.py` | MODIFY | `perf: add forward-filter-only EM as update_parameters_online_fast()` |
| 3 | `hpm_ai_v4/pattern.py` | MODIFY | `perf: add fixed-chunk EM buffering — run full EM every 100 steps` |
| 4 | `hpm_ai_v4/agents/agent.py` | MODIFY | `perf: wire fast online learning into HPMAgent.perceive_and_learn()` |
| 5 | `hpm_ai_v4/simulations/benchmark_fast_learning.py` | CREATE | `perf: add benchmark for fast online learning` |
