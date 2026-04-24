# Parallel Pattern Evaluation — Design Spec

**Date:** 2026-04-25
**Status:** Draft
**Scope:** HPM v4 (`hpm_ai_v4/`)

---

## 1. Goal

The inner loop of `HPMAgent.perceive_and_learn()` currently iterates over all
patterns sequentially: for each pattern it calls `observe()` (online-EM
sufficient-statistics decay + re-estimation), `adapt()` (full forward-backward
over the last 20 observations), `update_running_loss()` (log-likelihood over
the 100-observation buffer), and finally `total_score()` (epistemic +
affective + social evaluators).

For a population of 16 patterns with `latent_dim=2`, a single step takes
~40 ms on one core.  The goal is to **parallelise the per-pattern
forward-backward and score computation across CPU cores** while keeping
replicator dynamics, conflict-matrix computation, recombination, and
population pruning in the main process — these operations depend on the
complete, up-to-date population state and are not the bottleneck.

Target: >= 4x speedup on an 8-core machine with 16 patterns.

---

## 2. Architecture

### 2.1 `pattern_worker` — pure function

```python
# hpm_ai_v4/operators/parallel.py

def pattern_worker(state_dict: dict, obs_buffer: list,
                   field_freq: dict, params: dict) -> dict:
    ...
```

**Contract**

* Receives a plain serialisable `dict` representing one pattern's state.
* Reconstructs the pattern object locally, runs `observe`, `adapt`,
  `update_running_loss`, and `total_score`.
* Returns a plain `dict` with the updated state and all scores.
* No shared state, no globals, no object references across process
  boundaries.  This is mandatory for `multiprocessing` correctness.

**Input `state_dict` keys**

| Key | Type | Description |
|-----|------|-------------|
| `pattern_id` | `int` | `pattern.id` |
| `complexity` | `int` | 1 (flat) or 3 (hierarchical) |
| `latent_dim` | `int` | |
| `obs_dim` | `int` | |
| `A3` | `ndarray` | shape `(K, K)` |
| `A32` | `ndarray` | shape `(K, K)` |
| `A21` | `ndarray` | shape `(K, K)` |
| `B` | `ndarray` | shape `(K, obs_dim)` |
| `pi3` | `ndarray` | shape `(K,)` |
| `SS_A3` | `ndarray` | sufficient statistics |
| `SS_A32` | `ndarray` | |
| `SS_A21` | `ndarray` | |
| `SS_B` | `ndarray` | |
| `running_loss` | `float` | |
| `weight` | `float` | |
| `statistics_decay` | `float` | |

numpy arrays are picklable — no extra serialisation needed.

**Input `params` keys**

| Key | Type | Description |
|-----|------|-------------|
| `learning_rate` | `float` | passed to `observe` |
| `lambda_l` | `float` | passed to `update_running_loss` |
| `adapt_window` | `int` | length of obs slice passed to `adapt` |
| `beta_aff` | `float` | |
| `gamma_soc` | `float` | |
| `external_soc` | `float` | per-pattern social score |

**Output `result_dict` keys**

| Key | Type |
|-----|------|
| `pattern_id` | `int` |
| `A3`, `A32`, `A21`, `B`, `pi3` | `ndarray` |
| `SS_A3`, `SS_A32`, `SS_A21`, `SS_B` | `ndarray` |
| `running_loss` | `float` |
| `ep_score` | `float` |
| `aff_score` | `float` |
| `soc_score` | `float` |
| `total_score` | `float` |

### 2.2 `ParallelPatternPool` — pool wrapper

```python
class ParallelPatternPool:
    def __init__(self, num_workers: int = 1): ...
    def map_patterns(self,
                     patterns: list,
                     obs_buffer: list,
                     field_freq: dict,
                     params: dict) -> list: ...
    def close(self): ...
```

* Created **once** at agent construction; reused each step (avoids the
  fork-per-step penalty of ~5–20 ms on Linux).
* `num_workers=1` runs patterns sequentially in the main process — no pool
  overhead, useful for debugging and profiling.
* `num_workers > 1` creates a `multiprocessing.Pool` with that many workers
  and calls `pool.starmap(pattern_worker, task_args)`.
* `close()` calls `pool.terminate()` and `pool.join()`.

### 2.3 Integration into `HPMAgent`

`HPMAgent.__init__` gains `num_workers: int = 1` and creates a
`ParallelPatternPool`.

`HPMAgent.perceive_and_learn(obs, field_freq)` is refactored:

1. Append `obs` to buffer, compute `field_freq`.
2. Serialise all patterns to state dicts.
3. Call `self._pool.map_patterns(...)` — returns list of result dicts.
4. Write updated parameters back from result dicts into pattern objects.
5. Build `totals` dict from `result['total_score']`.
6. Run `compute_conflict_matrix` + `meta_pattern_update` sequentially.
7. Prune, recombine (every 20 steps), broadcast (unchanged).

Steps 6–7 are intentionally kept sequential: `meta_pattern_update` mutates
all pattern weights simultaneously and depends on the full population state;
`recombine` is called at most once per step and is cheap.

---

## 3. Why a Pure Function (not shared object)

`multiprocessing.Pool` workers run in **separate processes**.  Python objects
cannot be passed by reference across process boundaries; they must be
transmitted via a pipe and unpickled in the worker.

`HierarchicalPattern` contains only numpy arrays and Python scalars — all
picklable.  However, passing a plain `dict` of arrays is lighter than the full
object, avoids re-importing the class in every worker, and makes the worker
completely self-contained and easy to test in isolation.

Shared memory (`multiprocessing.shared_memory`) would avoid the copy but
requires explicit layout management and is error-prone with structured arrays.
For a research codebase the dict approach is the correct trade-off.

---

## 4. Fallback: `num_workers=1`

When `num_workers=1`, `map_patterns` iterates over patterns in the current
process.  This is the default.  No pool is created.  Tests can set
`num_workers=1` to avoid fork overhead in CI.

---

## 5. Recombination Stays Sequential

`recombine()` runs every 20 steps on two parent patterns and returns one child.
It is not a bottleneck (~0.1 ms).  It also requires the full up-to-date weight
distribution to sample parents.  It stays in the main process, unchanged.

---

## 6. No Ray Dependency

Implementation uses only `multiprocessing` from the Python standard library.
No new packages are added to `requirements.txt`.  Ray is not appropriate for a
single-machine research loop — it adds a distributed scheduler, object store,
and dashboard that are overkill and impose a >= 1 s startup penalty per run.

---

## 7. Expected Speedup

| num_workers | Patterns | Steps/s (estimated) | Speedup |
|-------------|----------|---------------------|---------|
| 1 (baseline)| 16 | ~25 | 1x |
| 2 | 16 | ~45 | ~1.8x |
| 4 | 16 | ~80 | ~3.2x |
| 8 | 16 | ~130 | ~5.2x |

Speedup is sub-linear due to IPC overhead (~2 ms/pattern round-trip) and
the sequential replicator dynamics step.  With more patterns or longer
`adapt_window`, the compute-to-communication ratio improves.

---

## 8. Metrics

A benchmark script (`hpm_ai_v4/simulations/benchmark_parallel.py`) will:

1. Create an `HPMAgent` for each `num_workers` in `[1, 2, 4, 8]`.
2. Run 500 steps of `perceive_and_learn` on synthetic observations.
3. Print a table of steps/second.

Developers should run this before and after any change to the parallel path to
detect regressions.

---

## 9. Picklability Note

`HierarchicalPattern` uses only numpy arrays and Python scalars.  No
`threading.Lock`, no file handles, no socket objects.  The pattern state dict
contains only numpy arrays and Python scalars — standard types that Python's
multiprocessing transport handles safely.  The worker reconstructs a minimal
local pattern object from the dict; no cross-process object references exist.
