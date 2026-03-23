# Sub-project 6: L4 (Generative Rules) and L5 (Meta-Patterns) — Math Benchmark

## Goal

Extend the StructuredOrchestrator with two new cognitive layers: L4 (Forward Prediction / Generative Rules) and L5 (Meta-Monitor / Strategy Selector). Validate on the structured math benchmark. L4 learns to predict L3 relational summaries from L2 structural features alone (System 1 intuition). L5 monitors L4's prediction error across training pairs and adaptively selects between exploitation (trust intuition) and exploration (full L3 analysis).

## Architecture

### L4: Generative Head (`hpm/agents/l4_generative.py`)

L4 is a domain-agnostic online ridge regressor that maps L2 feature vectors to predicted L3 relational summary vectors.

**Class: `L4GenerativeHead`**
- `feature_dim_in: int` — L2 feature dimension (10 for math)
- `feature_dim_out: int` — L3 feature dimension (12 for math)
- `alpha: float = 0.01` — ridge regularisation coefficient
- `_X: list[np.ndarray]` — accumulated L2 vectors (training pairs)
- `_Y: list[np.ndarray]` — accumulated L3 vectors (training pairs)
- `_W: np.ndarray | None` — fitted weight matrix (10×12), None until fit() called

**Methods:**
- `accumulate(l2_vec: np.ndarray, actual_l3_vec: np.ndarray) -> None` — store training pair
- `fit() -> None` — solve `W = (X^T X + αI)^{-1} X^T Y` via `np.linalg.solve`; no-op if fewer than 2 pairs
- `predict(l2_vec: np.ndarray) -> np.ndarray | None` — return `l2_vec @ W`; None if not fitted
- `reset() -> None` — clear accumulator and weight matrix (call at task boundary)

**Design notes:**
- Batch ridge solve at test time (not incremental) — optimal for N=3–4 training pairs
- Stateless between tasks (reset at task boundary)
- No dependency on agent protocol, pattern store, or ring buffer
- Weight matrix is 10×12 = 120 parameters; computationally trivial

### L5: Meta-Monitor (`hpm/agents/l5_monitor.py`)

L5 monitors L4's prediction surprise across training pairs within a task and outputs a `strategic_confidence` scalar that gates the scoring interpolation.

**Class: `L5MetaMonitor`**
- `theta_low: float = 0.2` — surprise below this → Exploit mode
- `theta_high: float = 0.5` — surprise above this → Explore mode
- `direction_weight: float = 0.7` — weight on cosine distance component
- `_surprises: list[float]` — accumulated surprise values

**Surprise metric** (computed after each training pair where L4 has a prediction):
```
surprise_t = 0.7 * cos_dist(Ŷ, actual_L3)
           + 0.3 * |‖Ŷ‖ - ‖actual_L3‖| / max(‖actual_L3‖, 1e-8)
```
- Cosine distance (directional error): detects wrong transformation family
- Magnitude delta (scale error): detects correct family but wrong intensity
- Direction dominates (0.7) — wrong family is more catastrophic than wrong magnitude

**Strategy selection** from running mean `S̄ = mean(_surprises)`:

| Running surprise | Mode | `strategic_confidence` |
|---|---|---|
| S̄ < 0.2 | Exploit — trust L4 | 0.9 |
| 0.2 ≤ S̄ ≤ 0.5 | Neutral | 1 − S̄ |
| S̄ > 0.5 | Explore — override L4 | 0.3 |

**Methods:**
- `update(l4_pred: np.ndarray | None, actual_l3: np.ndarray) -> None` — compute surprise and accumulate; no-op if l4_pred is None
- `strategic_confidence() -> float` — return confidence from table above; 1.0 if no surprises yet (no data = full trust until evidence arrives)
- `reset() -> None` — clear surprises (call at task boundary)

### LevelBundle Extension (`hpm/agents/hierarchical.py`)

Add optional field to `LevelBundle` (or equivalent extraction): `strategic_confidence: float = 1.0`. Default preserves backward compatibility — existing code ignores it.

### StructuredOrchestrator Integration (`hpm/agents/structured.py`)

Activate `generative_head` and `meta_monitor` slots (currently raise `NotImplementedError` for non-None values).

**Training pair step (each call):**
1. L1 → L2 → L3 encoders run as before, producing `l2_vec`, `actual_l3_vec`
2. `L4.accumulate(l2_vec, actual_l3_vec)`
3. `l4_pred = L4.predict(l2_vec)` — None until ≥2 pairs accumulated
4. `L5.update(l4_pred, actual_l3_vec)` — skip update if l4_pred is None

**Test-time scoring (per candidate):**
1. `L4.fit()` — solve ridge regression once per task (idempotent)
2. `γ = L5.strategic_confidence()`
3. For each candidate: run L2 encoder → `l2_c`; run L3 encoder → `l3_c`
4. `l3_pred = L4.predict(l2_c)` — L4 intuition for this candidate
5. `l4_score = cosine_distance(l3_c, l3_pred)` if l3_pred is not None, else 0.0
6. `l3_nll = standard_L3_NLL(l3_c, l3_prototype)` — existing prototype-based score
7. **Combined score:** `γ × l4_score + (1 − γ) × l3_nll`

The combined score implements the System 1 / System 2 interpolation:
- γ → 1 (exploit): score by how well candidate's L3 matches L4's intuitive prediction
- γ → 0 (explore): fall back to standard L3 prototype NLL

**Reset:** `L4.reset()` and `L5.reset()` called at task boundary (beginning of each new task).

### Benchmark (`benchmarks/structured_math_l4l5.py`)

Extends `structured_math.py` with three conditions:

| Condition | Description |
|---|---|
| `l2l3` | Existing best: L2+L3 scoring, γ=1.0 fixed (baseline at 96.7%) |
| `l4_only` | L4 prediction scoring only, γ=1.0 always (no L5 monitoring) |
| `l4l5_full` | Full L4+L5: adaptive γ from L5 meta-monitor |

**Benchmark parameters:** `n_per_family=60`, 3 transformation families → 180 tasks total.

**Expected results:**
- `l4_only` validates whether structural features predict transformation classes (ablation: intuition without supervision)
- `l4l5_full` should match or exceed `l2l3` on standard tasks and outperform `l4_only` on tasks where L4 misfires (L5 detects surprise and falls back)

## File Layout

**New files:**
- `hpm/agents/l4_generative.py` — `L4GenerativeHead`
- `hpm/agents/l5_monitor.py` — `L5MetaMonitor`
- `benchmarks/structured_math_l4l5.py` — benchmark with 3 conditions
- `tests/agents/test_l4_generative.py` — unit tests for L4
- `tests/agents/test_l5_monitor.py` — unit tests for L5

**Modified files:**
- `hpm/agents/structured.py` — activate generative_head/meta_monitor extension points
- `hpm/agents/hierarchical.py` — optional strategic_confidence in LevelBundle

## Testing

**L4 unit tests:**
- Accumulate 3 pairs, fit succeeds, predict returns shape (12,)
- Ridge with α=0.01 is stable when X is underdetermined (N < features)
- reset() clears state; predict() returns None after reset

**L5 unit tests:**
- Low surprise (S̄=0.1) → strategic_confidence=0.9
- High surprise (S̄=0.6) → strategic_confidence=0.3
- Neutral (S̄=0.35) → strategic_confidence≈0.65
- No data yet → strategic_confidence=1.0
- reset() clears surprises

**Integration smoke test:**
- Run `structured_math_l4l5.py` with n_per_family=3, all conditions return float in [0,1]

## Scope

- Math benchmark only (SP6). ARC integration is SP7 once L4/L5 are validated in the deterministic symbolic domain.
- No forgetting factor needed (per-task reset; stationary mapping within task).
- No SymPy dependency in L4/L5 — they operate on numpy vectors only.
