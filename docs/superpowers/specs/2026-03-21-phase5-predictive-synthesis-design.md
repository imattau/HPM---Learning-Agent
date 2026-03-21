# Phase 5: Predictive Synthesis Agent Design Specification

**Date:** 2026-03-21
**Status:** Draft v1

---

## Overview

The Predictive Synthesis Agent ("The Forecaster") is a Phase 5 component that moves the HPM system from passive observation to active prediction. It uses the `μ` of the highest-weight Level 4+ GaussianPattern as the population's primary structural law prediction, then validates that prediction's robustness via an additive-noise Far Transfer probe.

The Forecaster provides two signals:
1. **Prediction** (`x̂ = μ_top`) — the system's current best guess at the next observation, derived from the most dominant deep structural law
2. **Fragility score** — a ΔNll robustness index measuring how well the top pattern tolerates noise-perturbed observations; high ΔNll = pattern is fragile (likely a surface-level pattern masquerading as a deep law)

These signals feed into the broader Phase 5 ecosystem: the Recombination Strategist can trigger a burst when fragility is high; the Structural Law Monitor can cross-reference `top_pattern_id` to track which laws produce robust forecasts.

The Forecaster is implemented as a standalone `PredictiveSynthesisAgent` class in `hpm/monitor/predictive_synthesis.py`, composed into `MultiAgentOrchestrator` as an optional `forecaster=None` kwarg. It runs **every step** (prediction and scoring use only `log_prob()` — no network calls).

---

## 1. File Structure

```
hpm/monitor/predictive_synthesis.py              # new: PredictiveSynthesisAgent class
tests/monitor/test_predictive_synthesis.py       # new: unit + integration tests
hpm/monitor/__init__.py                          # existing: add PredictiveSynthesisAgent export
hpm/agents/multi_agent.py                        # existing: add forecaster=None kwarg + call
```

---

## 2. PredictiveSynthesisAgent

### 2.1 Constructor

```python
from collections import deque  # standard library — no extra dependency

def __init__(
    self,
    store,
    probe_k: int = 10,
    probe_n: int = 5,
    probe_sigma_scale: float = 0.5,
    fragility_threshold: float = 1.0,
    min_bridge_level: int = 4,
):
    self._store = store
    self.probe_k = probe_k
    self.probe_n = probe_n
    self.probe_sigma_scale = probe_sigma_scale
    self.fragility_threshold = fragility_threshold
    self.min_bridge_level = min_bridge_level
    self._obs_history: deque = deque(maxlen=probe_k)
```

| Parameter | Default | Role |
|-----------|---------|------|
| `store` | required | Shared SQLiteStore — held as `self._store` |
| `probe_k` | 10 | Number of recent observations used in Far Transfer probe |
| `probe_n` | 5 | Number of noisy copies generated per observation |
| `probe_sigma_scale` | 0.5 | Noise scale: `σ_probe = probe_sigma_scale × √mean(diag(top_pattern.sigma))` |
| `fragility_threshold` | 1.0 | ΔNll above which pattern is flagged as fragile |
| `min_bridge_level` | 4 | Minimum level for top-pattern selection (fallback to 3) |

### 2.2 Internal State

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_obs_history` | `deque[np.ndarray]` | Rolling buffer of recent observations (maxlen=probe_k) |

### 2.3 Main Method

```python
def step(self, step_t: int, current_obs: dict, field_quality: dict) -> dict:
    """
    Called by MultiAgentOrchestrator every step.

    Args:
        step_t:       Current orchestrator step counter.
        current_obs:  Dict {agent_id: np.ndarray} from orchestrator.step() call.
        field_quality: Dict from StructuralLawMonitor.step() (or {}).

    Returns:
        forecast_report dict (always, every step).
    """
```

**Returns** `forecast_report` dict with these keys (always present):

| Key | Type | Meaning |
|-----|------|---------|
| `prediction` | `np.ndarray \| None` | x̂ = μ of top-weight Level 4+ pattern; None if no eligible pattern |
| `prediction_error` | `float \| None` | NLL of top pattern on current observation; None if no obs or no pattern |
| `fragility_score` | `float \| None` | Mean NLL across noise-perturbed observations; None if < probe_k history |
| `delta_nll` | `float \| None` | fragility_score − baseline_nll; None if probe not run |
| `fragility_flag` | bool | True if delta_nll > fragility_threshold |
| `top_pattern_level` | `int \| None` | Level of selected top pattern (4 or 3 in fallback) |
| `top_pattern_id` | `str \| None` | ID of selected top pattern (for Librarian cross-referencing) |

---

## 3. Step Logic

### 3.1 Fast Gate

```python
if field_quality.get("level4plus_count", 0) == 0:
    # No Level 4+ patterns — return uncertainty signal immediately
    return _empty_report()
```

`_empty_report()` returns the full dict with all values set to None/False.

**Rationale:** Using `field_quality["level4plus_count"]` as the gate ensures the Forecaster and Librarian always agree on what constitutes an eligible law, without redundant level classification.

### 3.2 Query and Select Top Pattern

Call `self._store.query_all()` → list of `(pattern, weight, agent_id)` triples.

Filter to `pattern.level >= self.min_bridge_level` (default 4). Select the pattern with maximum weight.

**Fallback:** If no Level 4+ patterns are found in the store (timing discrepancy with the Librarian), fall back to the highest-weight Level 3 pattern. If no Level 3+ patterns exist, return empty report.

### 3.3 Extract Observation

Take one observation from `current_obs`: `x_obs = next(iter(current_obs.values()))`.

Trim or pad `x_obs` to match `top_pattern.mu.shape[0]` if dimensions differ (defensive handling for multi-agent configs with heterogeneous feature_dim). Trimming removes trailing elements; padding appends zeros. Use `np.resize` or explicit slice/concatenate — whichever is clearest in the implementation.

### 3.4 Predict and Score

```python
prediction = top_pattern.mu.copy()
prediction_error = float(top_pattern.log_prob(x_obs))
```

### 3.5 Update Obs History

```python
self._obs_history.append(x_obs.copy())
```

### 3.6 Far Transfer Probe

Only runs if `len(self._obs_history) >= self.probe_k`.

**Noise scale:**
```python
sigma_diag_mean = float(np.mean(np.diag(top_pattern.sigma)))
sigma_probe = self.probe_sigma_scale * np.sqrt(max(sigma_diag_mean, 1e-9))
```

**Perturbation and scoring:**
```python
rng = np.random.default_rng()  # unseeded for variety in production
nlls = []
for x_hist in self._obs_history:
    for _ in range(self.probe_n):
        noise = rng.normal(0.0, sigma_probe, size=x_hist.shape)
        x_perturbed = x_hist + noise
        nlls.append(float(top_pattern.log_prob(x_perturbed)))

fragility_score = float(np.mean(nlls))
delta_nll = fragility_score - prediction_error
fragility_flag = delta_nll > self.fragility_threshold
```

**Note on RNG:** The probe uses `np.random.default_rng()` (new-style Generator), which is independent of the legacy `np.random` module — seeding `np.random` will NOT affect it. For deterministic tests, use `probe_sigma_scale=0` (zero noise means all perturbed observations equal the originals, so fragility_score equals prediction_error and delta_nll is 0.0). Do not rely on seeding the global RNG state for test determinism.

---

## 4. MultiAgentOrchestrator Integration

### 4.1 Constructor Change

```python
def __init__(self, agents, field: PatternField, seed_pattern=None, groups=None,
             monitor=None, strategist=None, bridge=None, forecaster=None):
```

`forecaster: PredictiveSynthesisAgent | None = None` added. Fully backward compatible.

### 4.2 step() Change

After `bridge.step()`, add:

```python
forecast_report = (
    self.forecaster.step(self._t, observations, field_quality)
    if self.forecaster is not None
    else {}
)
```

`observations` is the dict already available as the `step()` parameter.

Updated return:
```python
return {
    **metrics,
    "field_quality": field_quality,
    "interventions": interventions,
    "bridge_report": bridge_report,
    "forecast_report": forecast_report,
}
```

### 4.3 Typical Usage

```python
store = SQLiteStore("runs/experiment.db")
agents = [Agent(config, store=store) for _ in range(4)]
monitor = StructuralLawMonitor(store, T_monitor=10)
forecaster = PredictiveSynthesisAgent(store, probe_k=10, fragility_threshold=1.0)
orchestrator = MultiAgentOrchestrator(
    agents, field=field,
    monitor=monitor, forecaster=forecaster
)
```

---

## 5. Testing Strategy

Tests in `tests/monitor/test_predictive_synthesis.py`. Use a real `SQLiteStore` via `tmp_path`. Seed `np.random` before calls to `step()` for deterministic probe results.

**Helper:**
```python
def _make_pattern(level, mu, sigma_scale=1.0, dim=4):
    p = GaussianPattern(mu=np.array(mu), sigma=np.eye(dim) * sigma_scale)
    p.level = level
    return p
```

**Test cases:**

- `test_uncertainty_when_no_l4_patterns` — `field_quality={"level4plus_count": 0}` → all keys None/False, store not queried
- `test_prediction_is_top_pattern_mu` — two Level 4 patterns with different weights → `prediction` equals μ of higher-weight pattern
- `test_fallback_to_level3` — no Level 4 patterns in store, Level 3 exists → `top_pattern_level=3`, prediction=Level3 μ
- `test_prediction_error_computed` — prediction_error equals `top_pattern.log_prob(obs)`
- `test_probe_none_before_k_obs` — fewer than `probe_k` observations → `fragility_score=None`, `delta_nll=None`
- `test_robust_pattern_not_flagged` — wide-sigma pattern (large Σ diag → noise absorbed) + fragility_threshold=1000 → `fragility_flag=False`
- `test_fragile_pattern_flagged` — narrow-sigma pattern (tiny Σ diag → noise destroys fit) + low fragility_threshold → `fragility_flag=True`
- `test_noise_scaled_to_pattern_sigma` — probe sigma proportional to pattern diag(Σ): use probe_sigma_scale=0 → nlls equal prediction_error (no noise)
- `test_forecast_report_keys_always_present` — all 7 keys always in returned dict
- `test_orchestrator_no_forecaster` — `forecaster=None` → `forecast_report == {}`
- `test_orchestrator_forecaster_integrated` — forecaster returns forecast_report with all expected keys after one step

---

## 6. What Is NOT in Scope

- Multi-pattern ensemble prediction (weighted average of all μ_i) — top-pattern μ only in v1
- Explicit `sample()` method on GaussianPattern (not needed; probe uses log_prob)
- Writing forecast_report to SQLiteStore or NDJSON log (observability deferred to future)
- Using fragility_flag to directly trigger recombination (Strategist reads field_quality; forecaster signal is in forecast_report — the coupling is indirect, via future Strategist enhancement)
- Heterogeneous feature_dim agents beyond defensive trim/pad
