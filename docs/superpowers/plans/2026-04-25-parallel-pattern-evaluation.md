# Plan: Parallel Pattern Evaluation for HPM v4

**Date:** 2026-04-25
**Spec:** `docs/superpowers/specs/2026-04-25-parallel-pattern-evaluation-design.md`
**Branch:** `hpm-ai-v4-dev`

---

## Task 1: `pattern_worker` pure function

### Files
- CREATE `hpm_ai_v4/operators/parallel.py`
- CREATE `hpm_ai_v4/tests/test_parallel.py`

### Failing test (write first)

```python
# hpm_ai_v4/tests/test_parallel.py
import numpy as np
import pytest
from hpm_ai_v4.operators.parallel import pattern_worker

def make_state_dict(complexity=3, latent_dim=2, obs_dim=2):
    K, D = latent_dim, obs_dim
    def rand_trans(r, c):
        m = np.random.dirichlet(np.ones(c), size=r)
        return m
    return {
        'pattern_id': 0,
        'complexity': complexity,
        'latent_dim': K,
        'obs_dim': D,
        'A3':  rand_trans(K, K),
        'A32': rand_trans(K, K),
        'A21': rand_trans(K, K),
        'B':   rand_trans(K, D),
        'pi3': np.random.dirichlet(np.ones(K)),
        'SS_A3':  np.ones((K, K)) * 0.1,
        'SS_A32': np.ones((K, K)) * 0.1,
        'SS_A21': np.ones((K, K)) * 0.1,
        'SS_B':   np.ones((K, D)) * 0.1,
        'running_loss': 0.5,
        'weight': 0.2,
        'statistics_decay': 0.9,
    }

def make_params(pattern_id=0):
    return {
        'learning_rate': 0.02,
        'lambda_l': 0.1,
        'adapt_window': 10,
        'beta_aff': 0.4,
        'gamma_soc': 0.3,
        'external_soc': 0.5,
    }

def test_worker_returns_required_keys():
    np.random.seed(42)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {0: 0.3}
    state = make_state_dict()
    params = make_params()
    result = pattern_worker(state, obs_buffer, field_freq, params)
    required = {'pattern_id', 'A3', 'A32', 'A21', 'B', 'pi3',
                'SS_A3', 'SS_A32', 'SS_A21', 'SS_B',
                'running_loss', 'ep_score', 'aff_score',
                'soc_score', 'total_score'}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

def test_worker_scores_are_finite():
    np.random.seed(7)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {0: 0.2}
    state = make_state_dict()
    params = make_params()
    result = pattern_worker(state, obs_buffer, field_freq, params)
    for key in ('ep_score', 'aff_score', 'soc_score', 'total_score'):
        assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

def test_worker_flat_pattern():
    np.random.seed(3)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {0: 0.1}
    state = make_state_dict(complexity=1, latent_dim=1, obs_dim=2)
    params = make_params()
    result = pattern_worker(state, obs_buffer, field_freq, params)
    required = {'pattern_id', 'running_loss', 'total_score'}
    assert required.issubset(result.keys())
```

### Run command
```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v4/tests/test_parallel.py::test_worker_returns_required_keys -x
# Expected: FAILED (ImportError or similar — module does not exist yet)
```

### Implementation

```python
# hpm_ai_v4/operators/parallel.py
"""
Pure-function worker and pool wrapper for parallel pattern evaluation.
Uses stdlib multiprocessing only — no Ray, no extra dependencies.
"""
import numpy as np
from multiprocessing import Pool
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Helpers: pattern state serialisation
# ---------------------------------------------------------------------------

def pattern_to_dict(pattern) -> dict:
    """Serialise a HierarchicalPattern or FlatPattern to a plain dict."""
    return {
        'pattern_id': pattern.id,
        'complexity': pattern.complexity,
        'latent_dim': pattern.latent_dim,
        'obs_dim': pattern.obs_dim,
        'A3':   pattern.A3.copy(),
        'A32':  pattern.A32.copy(),
        'A21':  pattern.A21.copy(),
        'B':    pattern.B.copy(),
        'pi3':  pattern.pi3.copy(),
        'SS_A3':  pattern.SS_A3.copy(),
        'SS_A32': pattern.SS_A32.copy(),
        'SS_A21': pattern.SS_A21.copy(),
        'SS_B':   pattern.SS_B.copy(),
        'running_loss': float(pattern.running_loss),
        'weight': float(pattern.weight),
        'statistics_decay': float(pattern.statistics_decay),
    }


def dict_to_pattern(d: dict):
    """Reconstruct a minimal pattern object from a state dict."""
    from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
    if d['complexity'] == 1:
        p = FlatPattern(d['pattern_id'], obs_dim=d['obs_dim'])
    else:
        p = HierarchicalPattern(d['pattern_id'],
                                latent_dim=d['latent_dim'],
                                obs_dim=d['obs_dim'])
    p.A3   = d['A3']
    p.A32  = d['A32']
    p.A21  = d['A21']
    p.B    = d['B']
    p.pi3  = d['pi3']
    p.SS_A3  = d['SS_A3']
    p.SS_A32 = d['SS_A32']
    p.SS_A21 = d['SS_A21']
    p.SS_B   = d['SS_B']
    p.running_loss = d['running_loss']
    p.weight = d['weight']
    p.statistics_decay = d['statistics_decay']
    return p


# ---------------------------------------------------------------------------
# Pure worker function
# ---------------------------------------------------------------------------

def pattern_worker(state_dict: dict, obs_buffer: list,
                   field_freq: dict, params: dict) -> dict:
    """
    Pure function: reconstruct pattern, run observe/adapt/update_running_loss,
    compute scores, return updated state dict + scores.

    No shared state.  Safe to call from multiprocessing.Pool workers.
    """
    from hpm_ai_v4.evaluators.metrics import (
        epistemic_score, affective_score, social_score, total_score,
    )

    p = dict_to_pattern(state_dict)

    # --- per-step update (mirrors HPMAgent.perceive_and_learn inner loop) ---
    if obs_buffer:
        obs = obs_buffer[-1]
        p.observe(obs, learning_rate=params['learning_rate'])

        adapt_window = params.get('adapt_window', 20)
        if p.complexity >= 2 and len(obs_buffer) >= 10:
            p.adapt(obs_buffer[-adapt_window:])

        p.update_running_loss(obs_buffer, lambda_l=params['lambda_l'])

    # --- evaluator scores ---
    ep  = epistemic_score(p)
    aff = affective_score(p, obs_buffer)
    soc = social_score(p, field_freq)
    tot = total_score(
        p, obs_buffer, field_freq,
        beta_aff=params['beta_aff'],
        gamma_soc=params['gamma_soc'],
        external_soc=params.get('external_soc', 0.5),
    )

    # --- return updated state + scores ---
    result = pattern_to_dict(p)
    result['ep_score']    = float(ep)
    result['aff_score']   = float(aff)
    result['soc_score']   = float(soc)
    result['total_score'] = float(tot)
    return result


# ---------------------------------------------------------------------------
# Pool wrapper
# ---------------------------------------------------------------------------

class ParallelPatternPool:
    """
    Wraps multiprocessing.Pool for parallel per-pattern updates.

    num_workers=1  -> sequential fallback (no pool spawned, no IPC overhead).
    num_workers>1  -> pool created once at construction, reused each step.
    """

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self._pool = Pool(processes=num_workers) if num_workers > 1 else None

    def map_patterns(self, patterns: list, obs_buffer: list,
                     field_freq: dict, params: dict) -> list:
        """
        Run pattern_worker for each pattern.  Returns list of result dicts
        in the same order as `patterns`.
        """
        task_args = [
            (pattern_to_dict(p), list(obs_buffer), dict(field_freq),
             {**params, 'external_soc': params.get('external_soc_map', {}).get(p.id, 0.5)})
            for p in patterns
        ]
        if self._pool is None:
            # Sequential fallback
            return [pattern_worker(*args) for args in task_args]
        else:
            return self._pool.starmap(pattern_worker, task_args)

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.close()
```

### Run command (green)
```bash
python -m pytest hpm_ai_v4/tests/test_parallel.py -x -q
# Expected: 3 passed
```

### Commit
```bash
git add hpm_ai_v4/operators/parallel.py hpm_ai_v4/tests/test_parallel.py
git commit -m "feat: add pattern_worker pure function for parallel evaluation"
```

---

## Task 2: `ParallelPatternPool` class tests

### Files
- MODIFY `hpm_ai_v4/tests/test_parallel.py`

### Failing tests (append to test file)

```python
# append to hpm_ai_v4/tests/test_parallel.py

from hpm_ai_v4.operators.parallel import ParallelPatternPool
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern

def make_pattern_population(n=4, obs_dim=2):
    patterns = []
    for i in range(n - 1):
        p = HierarchicalPattern(pattern_id=i, obs_dim=obs_dim)
        p.weight = 1.0 / n
        patterns.append(p)
    flat = FlatPattern(pattern_id=n - 1, obs_dim=obs_dim)
    flat.weight = 1.0 / n
    patterns.append(flat)
    return patterns

def test_pool_map_returns_n_results():
    np.random.seed(1)
    patterns = make_pattern_population(n=4)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {p.id: 0.25 for p in patterns}
    params = {'learning_rate': 0.02, 'lambda_l': 0.1, 'adapt_window': 10,
              'beta_aff': 0.4, 'gamma_soc': 0.3}
    pool = ParallelPatternPool(num_workers=1)
    results = pool.map_patterns(patterns, obs_buffer, field_freq, params)
    assert len(results) == 4
    pool.close()

def test_pool_sequential_no_subprocess():
    """num_workers=1 must not spawn a pool (checked via _pool attribute)."""
    pool = ParallelPatternPool(num_workers=1)
    assert pool._pool is None
    pool.close()

def test_pool_result_ids_match_input_order():
    np.random.seed(2)
    patterns = make_pattern_population(n=3)
    obs_buffer = list(np.random.randint(0, 2, size=15))
    field_freq = {}
    params = {'learning_rate': 0.02, 'lambda_l': 0.1, 'adapt_window': 10,
              'beta_aff': 0.4, 'gamma_soc': 0.3}
    pool = ParallelPatternPool(num_workers=1)
    results = pool.map_patterns(patterns, obs_buffer, field_freq, params)
    for p, r in zip(patterns, results):
        assert p.id == r['pattern_id']
    pool.close()
```

### Run command (red, then green after Task 1 implementation is in place)
```bash
python -m pytest hpm_ai_v4/tests/test_parallel.py -x -q
# Expected after implementation: 6 passed
```

### Commit
```bash
git add hpm_ai_v4/tests/test_parallel.py
git commit -m "test: add ParallelPatternPool tests"
```

---

## Task 3: Integrate into `HPMAgent`

### Files
- MODIFY `hpm_ai_v4/agents/agent.py`
- MODIFY `hpm_ai_v4/tests/test_parallel.py` (add integration tests)

### Failing tests (append to test file)

```python
# append to hpm_ai_v4/tests/test_parallel.py

from hpm_ai_v4.agents.agent import HPMAgent

def test_agent_parallel_no_error():
    """HPMAgent with num_workers=2 should run without error."""
    np.random.seed(99)
    agent = HPMAgent(num_initial_patterns=3, obs_dim=2, num_workers=2)
    for _ in range(5):
        agent.perceive_and_learn(int(np.random.randint(0, 2)))
    agent._pool.close()

def test_agent_weights_sum_to_one():
    """Pattern weights must sum to ~1.0 after parallel update."""
    np.random.seed(11)
    agent = HPMAgent(num_initial_patterns=4, obs_dim=2, num_workers=1)
    for _ in range(10):
        agent.perceive_and_learn(int(np.random.randint(0, 2)))
    weights = sum(p.weight for p in agent.patterns)
    assert abs(weights - 1.0) < 1e-6, f"Weights sum to {weights}, not 1.0"
```

### Implementation changes to `hpm_ai_v4/agents/agent.py`

1. Add import at top:
```python
from hpm_ai_v4.operators.parallel import ParallelPatternPool, pattern_to_dict, dict_to_pattern
```

2. Add `num_workers: int = 1` parameter to `HPMAgent.__init__`:
```python
def __init__(self, num_initial_patterns: int = 5,
             external_substrate=None, obs_dim: int = 2,
             num_workers: int = 1):
    ...
    self._pool = ParallelPatternPool(num_workers=num_workers)
```

3. Replace the sequential pattern-update loop in `perceive_and_learn`:

**Before (lines 103–110 of agent.py):**
```python
for p in self.patterns:
    p.observe(obs, learning_rate=0.02)
    if p.complexity >= 2 and len(self.obs_buffer) >= 10:
        p.adapt(self.obs_buffer[-20:])
    p.update_running_loss(self.obs_buffer, lambda_l=0.1)
```

**After:**
```python
# Build params dict for workers
worker_params = {
    'learning_rate': 0.02,
    'lambda_l': 0.1,
    'adapt_window': 20,
    'beta_aff': self.beta_aff,
    'gamma_soc': self.gamma_soc,
    'external_soc_map': self.external_social_scores,
}

# Parallel per-pattern update + score computation
results = self._pool.map_patterns(
    self.patterns, self.obs_buffer, field_freq, worker_params
)

# Write updated state back into pattern objects
result_by_id = {r['pattern_id']: r for r in results}
for p in self.patterns:
    r = result_by_id[p.id]
    p.A3   = r['A3'];  p.A32  = r['A32']
    p.A21  = r['A21']; p.B    = r['B']
    p.pi3  = r['pi3']
    p.SS_A3  = r['SS_A3'];  p.SS_A32 = r['SS_A32']
    p.SS_A21 = r['SS_A21']; p.SS_B   = r['SS_B']
    p.running_loss = r['running_loss']

# Build totals from parallel results (replaces old loop in step 3)
totals = {r['pattern_id']: r['total_score'] for r in results}
```

4. Remove the now-redundant step-3 `totals` loop (old lines 117–126).

5. Move `field_freq = self.field.update(self.patterns)` to before the parallel
   dispatch (it was already at step 2 — ensure it is computed before
   `map_patterns` is called).

### Run command
```bash
python -m pytest hpm_ai_v4/tests/test_parallel.py -x -q
# Expected: 8 passed
```

### Commit
```bash
git add hpm_ai_v4/agents/agent.py hpm_ai_v4/tests/test_parallel.py
git commit -m "feat: integrate ParallelPatternPool into HPMAgent.perceive_and_learn"
```

---

## Task 4: Benchmark script

### Files
- CREATE `hpm_ai_v4/simulations/benchmark_parallel.py`

### No test needed

### Implementation

```python
# hpm_ai_v4/simulations/benchmark_parallel.py
"""
Benchmark steps/second for HPMAgent with varying num_workers.

Usage:
    python -m hpm_ai_v4.simulations.benchmark_parallel
"""
import time
import numpy as np
from hpm_ai_v4.agents.agent import HPMAgent

STEPS = 500
NUM_PATTERNS = 16
OBS_DIM = 4
CONFIGS = [1, 2, 4, 8]


def run_benchmark(num_workers: int) -> float:
    np.random.seed(42)
    agent = HPMAgent(num_initial_patterns=NUM_PATTERNS,
                     obs_dim=OBS_DIM,
                     num_workers=num_workers)
    obs_seq = list(np.random.randint(0, OBS_DIM, size=STEPS))

    t0 = time.perf_counter()
    for obs in obs_seq:
        agent.perceive_and_learn(int(obs))
    elapsed = time.perf_counter() - t0

    agent._pool.close()
    return STEPS / elapsed


def main():
    print(f"\nBenchmark: {STEPS} steps, {NUM_PATTERNS} patterns, obs_dim={OBS_DIM}")
    print(f"{'num_workers':>12} {'steps/s':>10} {'speedup':>10}")
    print("-" * 36)

    baseline = None
    for nw in CONFIGS:
        sps = run_benchmark(nw)
        if baseline is None:
            baseline = sps
        speedup = sps / baseline
        print(f"{nw:>12} {sps:>10.1f} {speedup:>10.2f}x")


if __name__ == '__main__':
    main()
```

### Run command
```bash
python -m hpm_ai_v4.simulations.benchmark_parallel
# Expected output (values will vary by machine):
#
# Benchmark: 500 steps, 16 patterns, obs_dim=4
#  num_workers    steps/s    speedup
# ------------------------------------
#            1       24.3      1.00x
#            2       43.1      1.77x
#            4       76.8      3.16x
#            8      121.4      4.99x
```

### Commit
```bash
git add hpm_ai_v4/simulations/benchmark_parallel.py
git commit -m "feat: add benchmark script for parallel pattern evaluation"
```

---

## Final commit (spec + plan)

```bash
git add docs/superpowers/specs/2026-04-25-parallel-pattern-evaluation-design.md \
        docs/superpowers/plans/2026-04-25-parallel-pattern-evaluation.md
git commit -m "feat: add parallelisation spec and plan for HPM v4 pattern evaluation"
```
