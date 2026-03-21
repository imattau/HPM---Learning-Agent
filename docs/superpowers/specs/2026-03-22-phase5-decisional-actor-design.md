# Phase 5: Decisional RL Actor Design Specification

**Date:** 2026-03-22
**Status:** Draft v1

---

## Overview

The Decisional RL Actor ("The Actor") is the final Phase 5 component. It sits at the very end of `MultiAgentOrchestrator.step()` and has the final word on both external action selection and internal meta-action dispatch.

The Actor reads two inputs every step:
- `forecast_report` — from `PredictiveSynthesisAgent`: prediction (top-pattern μ), fragility_flag, etc.
- `field_quality` — from `StructuralLawMonitor`: redundancy, level4plus_count, etc.

It makes two kinds of decisions, each with its own Q-value learning head:

1. **ExternalHead** — selects a discrete external action using top-pattern log-prob as a learned prior, updated by environment reward (TD(0) bandit)
2. **InternalHead** — selects a meta-action that mutates orchestrator component state (forecaster's `min_bridge_level`, bridge's `_t`), updated by Δfield_quality reward at the next trigger

---

## 1. File Structure

```
hpm/agents/actor.py              # new: DecisionalActor class (+ ExternalHead, InternalHead)
tests/agents/test_actor.py       # new: 13 unit + integration tests
hpm/agents/multi_agent.py        # existing: add actor=None kwarg + call
hpm/agents/__init__.py           # existing: add DecisionalActor export
```

---

## 2. DecisionalActor

### 2.1 Constructor

```python
def __init__(
    self,
    action_vectors: np.ndarray,             # shape (n_actions, feature_dim)
    forecaster=None,                         # ref to PredictiveSynthesisAgent
    bridge=None,                             # ref to SubstrateBridgeAgent
    alpha_ext: float = 0.1,
    alpha_int: float = 0.1,
    beta: float = 1.0,
    temperature: float = 1.0,
    redundancy_threshold: float = 0.3,
    min_bridge_level_bounds: tuple = (2, 6),
):
```

| Parameter | Default | Role |
|-----------|---------|------|
| `action_vectors` | required | Feature vectors for each external action, shape `(n_actions, feature_dim)` |
| `forecaster` | None | Ref to `PredictiveSynthesisAgent` — mutated by EXPLOIT/EXPLORE internal actions |
| `bridge` | None | Ref to `SubstrateBridgeAgent` — mutated by REGROUND internal action |
| `alpha_ext` | 0.1 | Learning rate for ExternalHead Q-update |
| `alpha_int` | 0.1 | Learning rate for InternalHead Q-update |
| `beta` | 1.0 | Scale of top-pattern log-prob prior in ExternalHead logits |
| `temperature` | 1.0 | Softmax temperature for both heads (→0 = argmax) |
| `redundancy_threshold` | 0.3 | Minimum redundancy to trigger InternalHead |
| `min_bridge_level_bounds` | (2, 6) | Clamp range for EXPLOIT/EXPLORE mutations |

### 2.2 Internal State

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_external_head` | `ExternalHead` | External action selection + Q-learning |
| `_internal_head` | `InternalHead` | Meta-action selection + delayed Q-learning |

### 2.3 Main Method

```python
def step(
    self,
    step_t: int,
    field_quality: dict,
    forecast_report: dict,
    external_reward: float = 0.0,
) -> dict:
```

Returns `actor_report` dict — all 4 keys always present:

| Key | Type | Meaning |
|-----|------|---------|
| `external_action` | `int \| None` | Chosen external action index; None if prediction=None |
| `internal_action` | `str \| None` | Name of meta-action taken; None if not triggered |
| `external_q_values` | `list[float]` | Current Q-values for all external actions |
| `internal_q_values` | `list[float]` | Current Q-values for all internal actions |

---

## 3. ExternalHead

### 3.1 State

```python
self.q_values = np.zeros(n_actions)
self._last_action: int | None = None
```

### 3.2 Action Selection

Called every step. Requires `forecast_report["prediction"] is not None`.

```python
log_prob_scores = np.array([
    top_pattern.log_prob(action_vectors[i]) for i in range(n_actions)
])
logits = beta * log_prob_scores + self.q_values
probs = softmax(logits / temperature)
action = np.random.choice(n_actions, p=probs)
self._last_action = action
```

If `prediction is None` (no top pattern): return `external_action=None`, skip update.

**Fallback when no GaussianPattern is available:** logits = q_values only (no log_prob contribution).

### 3.3 Q-Update

Called every step after action selection, when `external_reward` is available:

```python
if self._last_action is not None:
    q = self.q_values[self._last_action]
    self.q_values[self._last_action] = q + alpha_ext * (external_reward - q)
```

---

## 4. InternalHead

### 4.1 Meta-Actions (fixed set, v1)

| ID | Name | Effect |
|----|------|--------|
| 0 | `EXPLOIT` | `forecaster.min_bridge_level = min(current + 1, bounds[1])` |
| 1 | `EXPLORE` | `forecaster.min_bridge_level = max(current - 1, bounds[0])` |
| 2 | `REGROUND` | `bridge._t = bridge.T_substrate - 1` (forces pass next step) |

`EXPLOIT` and `EXPLORE` are no-ops if `forecaster is None`. `REGROUND` is a no-op if `bridge is None`.

### 4.2 Trigger Condition

InternalHead fires when ANY of the following is true:
- `field_quality.get("redundancy", 0.0) > redundancy_threshold`
- `forecast_report.get("fragility_flag", False) is True`

Otherwise: `internal_action=None`, no selection, no update.

### 4.3 State

```python
self.q_values = np.zeros(3)          # one per meta-action
self._last_action: int | None = None
self._baseline_score: float | None = None
```

### 4.4 Action Selection

```python
probs = softmax(self.q_values / temperature)
action = np.random.choice(3, p=probs)
self._last_action = action
self._baseline_score = _field_score(field_quality)
```

Where:
```python
def _field_score(field_quality: dict) -> float:
    count = field_quality.get("level4plus_count", 0) or 0
    redundancy = field_quality.get("redundancy") or 0.0
    return float(count) * (1.0 - float(redundancy))
```

### 4.5 Delayed Q-Update

On the **next trigger** (next step where the trigger condition fires):

```python
current_score = _field_score(field_quality)
reward = current_score - self._baseline_score
q = self.q_values[self._last_action]
self.q_values[self._last_action] = q + alpha_int * (reward - q)
# then select new action for this trigger
```

If `_baseline_score is None` (first trigger ever): skip reward, just select action.

---

## 5. MultiAgentOrchestrator Integration

### 5.1 Constructor Change

```python
def __init__(self, agents, field, seed_pattern=None, groups=None,
             monitor=None, strategist=None, bridge=None, forecaster=None, actor=None):
```

`actor: DecisionalActor | None = None` added. Fully backward compatible.

### 5.2 step() Change

After `forecast_report`, add:

```python
actor_report = (
    self.actor.step(
        self._t,
        field_quality,
        forecast_report,
        external_reward=rewards.get("__actor__", 0.0),
    )
    if self.actor is not None
    else {}
)
```

External reward is passed via the special key `"__actor__"` in the `rewards` dict supplied to `orchestrator.step()`.

Updated return:
```python
return {
    **metrics,
    "field_quality": field_quality,
    "interventions": interventions,
    "bridge_report": bridge_report,
    "forecast_report": forecast_report,
    "actor_report": actor_report,
}
```

### 5.3 Typical Usage

```python
action_vectors = np.eye(4)   # 4 one-hot actions in 4-dim space
actor = DecisionalActor(
    action_vectors,
    forecaster=forecaster,
    bridge=bridge,
    alpha_ext=0.1,
    alpha_int=0.1,
    redundancy_threshold=0.3,
)
orchestrator = MultiAgentOrchestrator(
    agents, field=field,
    monitor=monitor, forecaster=forecaster, bridge=bridge, actor=actor
)
```

---

## 6. Testing Strategy

Tests in `tests/agents/test_actor.py`. Use stub `forecast_report` and `field_quality` dicts — no real store or orchestrator needed for unit tests. Use `temperature=1e-9` (≈ argmax) for deterministic action selection in tests.

**Helper:**
```python
def _make_forecast(prediction=None, fragility_flag=False):
    return {
        "prediction": prediction,
        "fragility_flag": fragility_flag,
        "top_pattern_id": None,
        "top_pattern_level": None,
        "prediction_error": None,
        "fragility_score": None,
        "delta_nll": None,
    }

def _make_field_quality(redundancy=0.0, level4plus_count=0):
    return {"redundancy": redundancy, "level4plus_count": level4plus_count}
```

**Test cases:**

- `test_external_action_selected_by_logprob` — GaussianPattern centred on action_vectors[2]; temperature→0 → selected action == 2
- `test_q_values_update_on_reward` — reward=1.0, alpha_ext=1.0 → Q[selected] == 1.0 after one step
- `test_no_action_when_no_prediction` — `prediction=None` → `external_action=None`
- `test_internal_action_triggered_on_redundancy` — redundancy > threshold → `internal_action` not None
- `test_internal_action_triggered_on_fragility_flag` — fragility_flag=True, low redundancy → `internal_action` not None
- `test_internal_action_not_triggered_below_threshold` — redundancy=0, fragility_flag=False → `internal_action=None`
- `test_internal_q_update_on_delayed_reward` — trigger at t0 with score=1.0, trigger at t1 with score=2.0 → Q[last_action] moves toward reward=1.0
- `test_exploit_raises_min_bridge_level` — EXPLOIT fires → `forecaster.min_bridge_level` increases by 1
- `test_explore_lowers_min_bridge_level` — EXPLORE fires → `forecaster.min_bridge_level` decreases by 1
- `test_explore_respects_lower_bound` — EXPLORE at lower bound → level unchanged
- `test_reground_resets_bridge_t` — REGROUND fires → `bridge._t == bridge.T_substrate - 1`
- `test_actor_report_keys_always_present` — all 4 keys always in returned dict
- `test_orchestrator_no_actor` — `actor=None` → `actor_report == {}`
- `test_orchestrator_actor_integrated` — actor wired in, step returns `actor_report` with all 4 keys

---

## 7. What Is NOT in Scope

- Neural network policy head (linear Q-table only in v1)
- Continuous external actions (discrete action_vectors only)
- Multi-step TD (only single-step TD(0) bandit update)
- Actor writing actor_report to SQLiteStore or NDJSON log
- Unified reward normalisation across internal/external signals
- More than 3 internal meta-actions in v1
