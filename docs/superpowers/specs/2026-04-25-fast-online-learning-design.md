# Fast Online Learning — Design Spec

**Date:** 2026-04-25
**Branch:** hpm-ai-v4-dev
**Status:** Draft

---

## Goal

Reduce per-step inference cost by ~200× through three composable optimisations applied to `HierarchicalPattern` in `hpm_ai_v4/pattern.py`, without changing HPM dynamics or any external interface.

---

## Current Architecture (baseline)

`HierarchicalPattern` implements a **3-level HMM** with joint state space (z3, z2, z1) of size K³:

- `A3`  — transition matrix at level 3: p(z3_t | z3_{t-1}), shape (K, K)
- `A32` — cross-level emission: p(z2_t | z3_t), shape (K, K)
- `A21` — cross-level emission: p(z1_t | z2_t), shape (K, K)
- `B`   — observation emission: p(x_t | z1_t), shape (K, obs_dim)
- `pi3` — initial distribution over z3, shape (K,)

The forward-backward pass in `_forward_backward()` operates on tensors of shape (T, K, K, K) — O(T · K³) space and O(T · K⁴) time (the K⁴ arises from the z3→z3 transition summed over predecessors at each step).

`adapt(obs_seq)` is the full batch EM method. `update_running_loss()` calls `log_likelihood()` which calls `_forward_backward()` — O(T · K⁴) per step.

The parallel worker pool in `operators/parallel.py` drives per-pattern updates each step from `perceive_and_learn()`.

---

## Technique A: Forward-Filter-Only EM

### Current cost
`_forward_backward()` allocates two (T, K, K, K) arrays and runs 2T passes.

### Proposed change
Replace the full E-step with a **forward-filter-only** pass that produces filtered (not smoothed) posteriors:

```
alpha[t] normalised  ->  gamma_filtered[t] ≈ P(z3,z2,z1 | x_{0:t})
```

The forward recursion already exists in `get_belief()` and `get_top_state()`. The fast EM reuses this structure.

**Sufficient statistics from filtered posteriors:**

```
gamma_f[t] = exp(alpha_t - logsumexp(alpha_t))        # (K,K,K) normalised

SS_A3  += sum_t  one-step xi from alpha[t-1] and alpha[t]
SS_A32 += sum_t  gamma_f[t].sum(axis=2)               # marginalise z1
SS_A21 += sum_t  gamma_f[t].sum(axis=0)               # marginalise z3
SS_B   += sum_t  gamma_f_z1[t, k] * 1[x_t = v]        # emission counts
```

This is "online EM with filtering" — a well-established approximation (Shumway & Stoffer 1982; Mongillo & Deneve 2008). The smoothed-vs-filtered difference is largest at sequence boundaries; for long sequences the approximation is tight.

**New method:** `update_parameters_online_fast(obs_seq)` — keeps `adapt()` (full FB) for offline/validation use.

**Speedup:** ~2× (no backward pass, no beta array allocation).

---

## Technique B: Fixed-Chunk EM

### Current cost
`perceive_and_learn()` calls the parallel worker pool every step, which runs full forward-backward over the obs buffer (up to T=100) for every pattern at every step — O(T · K⁴) per pattern per step.

### Proposed change
Buffer observations in `obs_chunk` of length C=100. Trigger full EM only when the chunk is full:

- **Between chunk boundaries:** run only a one-step forward update to maintain `running_loss` — O(K³) per step.
- **At chunk boundary (every C steps):** run `update_parameters_online_fast(obs_chunk)` — O(C · K⁴) amortised over C steps = O(K⁴) per step.

This is ~C× fewer full EM calls (C=100 gives 100× reduction).

**New state on `HierarchicalPattern`:**
```python
self.obs_chunk: List[int] = []
self.chunk_size: int = 100
```

**New method:**
```python
def maybe_update(self, obs: int, full_buffer: List[int]) -> None:
    # always: cheap one-step loss update
    self._update_loss_one_step(obs)
    # buffer and conditionally run EM
    self.obs_chunk.append(obs)
    if len(self.obs_chunk) >= self.chunk_size:
        self.update_parameters_online_fast(self.obs_chunk)
        self.obs_chunk.clear()
```

---

## Technique C: float32 Storage + Log Cache

### Current cost
All parameter arrays (`A3`, `A32`, `A21`, `B`, `pi3`) are float64 (numpy default). Every call to `_forward_backward()`, `get_belief()`, `get_top_state()` recomputes `np.log(A + 1e-12)` from scratch — 5 log calls per forward pass invocation.

### Proposed change

1. Store `A3`, `A32`, `A21`, `B`, `pi3`, all sufficient statistics as `np.float32` from `__init__`.
2. Maintain cached log arrays: `logA3`, `logA32`, `logA21`, `logB`, `logPi3` (float32).
3. Invalidate and refresh caches via `_refresh_log_cache()` called at the end of `_reestimate_parameters()`.
4. Use cached log arrays in forward passes instead of recomputing `np.log(...)` inline.

**Speedup:** ~1.5–2× for K=2–4 (fewer flops, better cache locality, no repeated log calls).

**Precision:** float32 has ~7 decimal digits of precision. For K ≤ 4 and obs_dim ≤ 256, this is sufficient; the floor in log computation should be raised to 1e-7 in float32 to avoid subnormals.

---

## Combined Effect

| Technique | Mechanism | Theoretical speedup |
|-----------|-----------|-------------------|
| A: Filter-only EM | Drop backward pass | ~2× |
| B: Fixed-chunk EM | 100× fewer EM calls | ~100× |
| C: float32 + log cache | Dtype + cached log | ~1.5–2× |
| **Combined** | | **~300× theoretical / ~100–200× practical** |

Practical overhead: numpy dispatch, chunk management, Python loops over patterns. Realistic gain: 100–150×.

---

## Interface Contract (unchanged)

The following external interfaces must remain identical:

| Interface | Guarantee |
|-----------|-----------|
| `HierarchicalPattern.log_likelihood(obs_seq)` | Same semantics, same return type (float) |
| `HierarchicalPattern.compression(obs_seq)` | Unchanged — still calls full `_forward_backward()` |
| `HierarchicalPattern.adapt(obs_seq)` | Unchanged — full FB EM kept for offline/validation |
| `HierarchicalPattern.get_belief(obs_seq)` | Unchanged |
| `HierarchicalPattern.grow_latent()` | Unchanged; must also call `_refresh_log_cache()` after resize |
| `HPMAgent.perceive_and_learn(obs)` | External signature unchanged |
| All existing tests | Must pass without modification |

---

## Files Modified

| File | Change type |
|------|-------------|
| `hpm_ai_v4/pattern.py` | MODIFY — add float32 init, log cache, `_refresh_log_cache()`, `_forward_filter()`, `update_parameters_online_fast()`, `obs_chunk`/`chunk_size`, `maybe_update()`, `_update_loss_one_step()` |
| `hpm_ai_v4/agents/agent.py` | MODIFY — `perceive_and_learn()` internal call pattern (no signature change) |
| `hpm_ai_v4/tests/test_fast_learning.py` | CREATE — TDD tests for all three techniques |
| `hpm_ai_v4/simulations/benchmark_fast_learning.py` | CREATE — timing benchmark |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| float32 underflow in log-space accumulation | Keep 1e-7 floor in log cache; use logsumexp throughout |
| Filter-only EM diverges for small K | Keep `adapt()` (full FB) available; benchmark learning curves |
| Chunk size C=100 too coarse for fast-changing sequences | Make `chunk_size` a constructor parameter with default 100 |
| `grow_latent()` resizes parameter arrays but not log cache | Call `_refresh_log_cache()` at end of `grow_latent()` |
