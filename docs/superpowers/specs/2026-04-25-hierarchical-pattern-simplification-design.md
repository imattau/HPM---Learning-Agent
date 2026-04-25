# Spec: HierarchicalPattern Simplification — 3-Level Joint HMM → Simple Single-Level HMM

**Date:** 2026-04-25
**Branch:** hpm-ai-v4-dev
**File:** `hpm_ai_v4/pattern.py`

---

## 1. Problem

The current `HierarchicalPattern` implements a 3-level joint HMM with latent variables z³, z², z¹ combined into a joint tensor of shape `(K, K, K)`. This causes:

- **Forward algorithm complexity:** O(T·K⁴) — for K=4 that is 256× slower than necessary
- **Memory:** joint state tensor `alpha[T, K, K, K]` grows as K⁴
- **Conceptual mismatch:** HPM hierarchy emerges from **stacking patterns between levels** (via `get_top_state()`), not from multi-level structure within a single pattern
- **Barrier to fast-online-learning:** standard HMM math (Baum-Welch) cannot be applied cleanly while the joint tensor structure is in place

---

## 2. Goal

Replace the 3-level joint HMM with a simple single-level HMM:

- K latent states, O(T·K²) forward-backward
- Single `A` (K×K) transition matrix, single `B` (K×obs_dim) emission matrix, single `pi` (K,) initial distribution
- Hierarchical structure in the population emerges from agents stacking patterns across levels via `get_top_state()`, exactly as HPM specifies

---

## 3. New `HierarchicalPattern` Interface

All public method signatures are preserved or backward-compatible where possible.

```python
class HierarchicalPattern:
    def __init__(self, pattern_id, latent_dim=2, obs_dim=6):
        self.A: np.ndarray          # (K, K) float32 transition matrix
        self.B: np.ndarray          # (K, obs_dim) float32 emission matrix
        self.pi: np.ndarray         # (K,) float32 initial distribution
        self.logA: np.ndarray       # cached log A
        self.logB: np.ndarray       # cached log B
        self.running_loss: float
        self.weight: float
        self.latent_dim: int        # = K
        self.obs_dim: int

    def log_likelihood(self, obs_seq: List[int]) -> float
    def update_parameters_online(self, obs_seq: List[int], window_size=100) -> None
    def update_running_loss(self, obs_seq: List[int]) -> None
    def compression(self) -> float          # MI between consecutive latent states (no obs_seq arg)
    def predictive_entropy(self, obs_seq: List[int]) -> float
    def predict_next(self, obs_seq: List[int]) -> int   # argmax of next obs distribution
    def get_top_state(self, obs_seq: List[int]) -> int  # most likely current latent state
    def predict_next_distribution(self, obs_seq: List[int]) -> np.ndarray  # (obs_dim,)
```

---

## 4. What Changes

### Removed
- `A3`, `A32`, `A21` — three separate transition matrices
- `pi3` — replaced by `pi`
- `logA3`, `logA32`, `logA21`, `logPi3` — replaced by `logA`
- `SS_A3`, `SS_A32`, `SS_A21`, `SS_B` — sufficient statistics arrays (all 4 replaced by 2: `SS_A`, `SS_B`)
- `statistics_decay`, `obs_chunk`, `chunk_size` — chunked EM buffering removed (replaced by windowed Baum-Welch)
- `_forward_backward` returning joint tensor `(T, K, K, K)` — replaced by standard HMM forward-backward `(T, K)`
- `_forward_filter` with joint tensor — replaced by standard scaled forward pass
- `adapt()` — replaced by `update_parameters_online()`
- `update_parameters_online_fast()` — merged into `update_parameters_online()`
- `maybe_update()` — removed (chunked buffering no longer needed)
- `observe()` — removed
- `_reestimate_parameters()` — inlined into `update_parameters_online()`
- `grow_latent()` — removed (K-growth was a workaround for the joint tensor's poor scaling; simple HMM can start at the right K)
- `complexity` attribute — removed (was 3 for HierarchicalPattern, 1 for FlatPattern; FlatPattern remains unchanged)
- `get_belief()` — removed (returned joint (K,K,K) posterior; callers use `get_top_state()` instead)

### Added
- `A`, `pi`, `logA` — simple single-level HMM parameters
- `_forward(obs_seq)` — standard scaled forward: returns `alpha (T, K)`, `scales (T,)`
- `_forward_backward(obs_seq)` — standard: returns `gamma (T, K)`, `xi (T-1, K, K)`
- `compression()` — now takes no `obs_seq` arg; computed analytically from stationary distribution of `A`
- `predictive_entropy(obs_seq)` — entropy of next-obs distribution under stationary distribution

### Signature changes (breaking, tests must be updated)
| Method | Old signature | New signature |
|---|---|---|
| `compression` | `compression(obs_seq)` | `compression()` |
| `predict_next_distribution` | uses joint belief (K,K,K) | uses simple alpha (K,) |
| `update_parameters_online` | did not exist (was `adapt`) | new, replaces `adapt` |

### Unchanged
- `HierarchicalPattern` class name and module location (`hpm_ai_v4/pattern.py`)
- `FlatPattern` class — completely unchanged
- `log_likelihood(obs_seq)` — same semantics, now O(T·K²)
- `update_running_loss(obs_seq)` — same semantics
- `get_top_state(obs_seq)` — same semantics, now returns argmax of marginalised alpha[-1] over K states
- `weight` attribute
- `running_loss` attribute
- `latent_dim`, `obs_dim` attributes
- Replicator dynamics — unchanged (`meta_pattern_update`, `recombine`)
- Recombination — `recombine()` performs row-wise crossover of `A`, `B`, `pi` — already correct for single matrices
- Reasoner interface — `get_relevant_patterns`, `compose_predictions`, `plan`, `simulate_future` all unchanged
- Evaluator inputs — `running_loss`, `compression()`, `predictive_entropy()` all remain

---

## 5. HPM Conceptual Alignment

The 3-level joint HMM was a misapplication of HPM's hierarchy concept. HPM specifies:

> Hierarchy emerges from **stacking patterns between levels**: a higher-level pattern takes as input the `get_top_state()` output of a lower-level pattern.

A single `HierarchicalPattern` is one node in that stack — it should be a simple, efficient HMM. Depth comes from population architecture, not from internal multi-level tensor structure.

The new design is consistent with HPM Appendix E: "each level maintains a latent state space of size K; the hierarchy is encoded in how states at one level gate or condition the dynamics at the next."

---

## 6. Speedup

| Quantity | Old (3-level joint) | New (single-level) |
|---|---|---|
| Forward-backward | O(T·K⁴) | O(T·K²) |
| Memory (alpha) | T × K × K × K | T × K |
| For K=4, T=100 | 25,600 cells | 400 cells |
| Speedup at K=4 | — | ~256× |

---

## 7. Migration Impact on Existing Tests

Tests that access pattern internals (`A3`, `A32`, `A21`, `pi3`, `SS_A3`, etc.) will fail and must be updated to use `A`, `pi`.

Tests that use external interface (`log_likelihood`, `weight`, `compression`, `get_top_state`) need only minor updates (e.g. `compression()` loses its `obs_seq` argument).

Tests in `test_fast_learning.py` (chunked EM, `update_parameters_online_fast`, `maybe_update`, `obs_chunk`) are entirely replaced by the new `update_parameters_online` tests.

Tests in `test_pattern_growth.py` (`grow_latent`, `A32`, `A21`, `pi3`, `SS_A3`) must be rewritten for the new single-matrix structure.

`FlatPattern` tests are unaffected.

---

## 8. `FlatPattern` — No Change

`FlatPattern` remains a subclass of `HierarchicalPattern` with `latent_dim=1`. Its `log_likelihood`, `observe`, `adapt`, `compression` overrides are unchanged. The only adjustment: `FlatPattern.__init__` will call `super().__init__` which now initialises `A`, `B`, `pi` rather than `A3`, `A32`, `A21` — this is automatically correct since `latent_dim=1` produces `A=(1,1)`, `B=(1,obs_dim)`, `pi=(1,)`.

The `complexity` attribute on `FlatPattern` (value=1) is preserved for backward compatibility with `DevelopmentalStage` and `HPMAgent._maybe_grow_patterns` (which gates on `p.complexity < 2`). `HierarchicalPattern` sets `complexity=1` (single-level is still "one level of latent structure").

---

## 9. `Reasoner` Impact

`reasoning.py` uses:
- `predict_next_distribution(obs_seq)` — preserved, new implementation uses simple alpha
- `get_belief(obs_seq)` — **removed**; `simulate()` calls this. `simulate()` must be rewritten to use `get_top_state` + `A` directly
- `pattern.A3`, `pattern.A32`, `pattern.A21` — used in `simulate()`, must be updated to `pattern.A`
- `pattern.B[z1_idx]` — unchanged (B still indexed by latent state)
- `pattern.complexity` — preserved

`simulate()` rewrite is straightforward: sample from `alpha[-1]` (single state index), then transition via `A`, emit via `B`.

---

## 10. `metrics.py` Impact

`affective_score` calls `pattern.compression(obs_seq)`. This must change to `pattern.compression()` (no argument). The evaluator is otherwise unchanged.

`pattern_density` references `pattern.latent_dim**2 * 3` (3 transition matrices). This should be updated to `pattern.latent_dim**2` (1 transition matrix) + `pattern.latent_dim * pattern.obs_dim` (1 emission matrix).

`HPMAgent._maybe_grow_patterns` calls `p.compression(self.obs_buffer)` — must change to `p.compression()`.
