# Phase 5: Recombination Strategist Design Specification

**Date:** 2026-03-21
**Status:** Draft v1

---

## Overview

The Recombination Strategist ("The Innovator") is a Phase 5 population-level governor that monitors the shared pattern population's innovative climate and adjusts per-agent configuration parameters to prevent premature convergence and stimulate structural novelty when the population is stuck.

It does **not** perform recombination itself — that remains the responsibility of each agent's `RecombinationOperator`. Instead, it acts as a macro-level throttle on the parameters that control how readily recombination fires and how aggressively new patterns are adopted.

The Strategist reads `field_quality` metrics produced by `StructuralLawMonitor` ("The Librarian") and applies three types of intervention:

1. **Recombination Burst** — temporarily lowers `conflict_threshold` on all agents when the population is stagnating
2. **Adoption Scaling** — dynamically adjusts `kappa_0` based on diversity trend
3. **Conflict Scale Damping** — reduces `beta_c` when conflict persists without diversity recovery

It is implemented as a standalone `RecombinationStrategist` class composed into `MultiAgentOrchestrator` as an optional `strategist=None` keyword argument — the same pattern as `StructuralLawMonitor`.

---

## 1. File Structure

```
hpm/monitor/recombination_strategist.py   # new: RecombinationStrategist class
hpm/monitor/__init__.py                   # existing: add RecombinationStrategist export
tests/monitor/test_recombination_strategist.py  # new: unit + integration tests
```

**Existing files modified:**
- `hpm/agents/multi_agent.py` — add `strategist=None` kwarg, call after monitor.step()
- `hpm/monitor/__init__.py` — export `RecombinationStrategist`

---

## 2. RecombinationStrategist

### 2.1 Constructor

```python
def __init__(
    self,
    diversity_low: float = 0.5,
    conflict_high: float = 0.3,
    stagnation_window: int = 3,
    burst_conflict_threshold: float = 0.01,
    burst_duration: int = 50,
    burst_cooldown: int = 100,
    kappa_0_min: float = 0.05,
    kappa_0_max: float = 0.3,
    kappa_0_ema_alpha: float = 0.2,
    beta_c_min: float = 0.1,
    beta_c_decay: float = 0.9,
):
```

| Parameter | Default | Role |
|-----------|---------|------|
| `diversity_low` | 0.5 | Diversity threshold below which stagnation counter advances |
| `conflict_high` | 0.3 | Conflict threshold above which stagnation counter advances |
| `stagnation_window` | 3 | Monitor cycles at stagnation before burst fires |
| `burst_conflict_threshold` | 0.01 | Temporary `conflict_threshold` applied to all agents during burst |
| `burst_duration` | 50 | Steps the burst lasts before originals are restored |
| `burst_cooldown` | 100 | Steps post-burst before stagnation counter can re-advance |
| `kappa_0_min` | 0.05 | Floor for adoption scaling |
| `kappa_0_max` | 0.3 | Ceiling for adoption scaling |
| `kappa_0_ema_alpha` | 0.2 | EMA smoothing factor for diversity trend |
| `beta_c_min` | 0.1 | Floor for conflict scale damping |
| `beta_c_decay` | 0.9 | Per-cycle multiplier when damping `beta_c` |

### 2.2 Internal State

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_stagnation_count` | int | Consecutive monitor cycles with low diversity + high conflict |
| `_burst_steps_remaining` | int | Steps left in current burst (0 = no burst active) |
| `_cooldown_steps_remaining` | int | Steps left in post-burst cooldown |
| `_original_conflict_thresholds` | dict[str, float] | Per-agent backup of `conflict_threshold` before burst (restored on burst end) |
| `_diversity_ema` | float \| None | Exponential moving average of diversity; `None` until first heavy-metric step |
| `_conflict_persistent_cycles` | int | Consecutive heavy-metric cycles with conflict > conflict_high |

**Note:** `_original_beta_cs` is NOT stored — `beta_c` damping is a one-way ratchet with no restoration, so no backup is needed.

### 2.3 Main Method

```python
def step(self, step_t: int, field_quality: dict, agents: list) -> dict:
    """
    Called by MultiAgentOrchestrator after monitor.step().

    Args:
        step_t:        Current orchestrator step counter.
        field_quality: Dict returned by StructuralLawMonitor.step()
                       (or {} if no monitor).
    agents:            List of Agent instances (for config mutation).

    Returns:
        interventions dict — always present, empty if no action taken.
    """
```

Returns an `interventions` dict with the following keys (all always present):

| Key | Type | Meaning |
|-----|------|---------|
| `burst_active` | bool | Whether a burst is currently active |
| `kappa_0` | float \| None | Current kappa_0 value applied to agents; None if no change |
| `beta_c_scaled` | bool | Whether beta_c has been damped this cycle |
| `stagnation_count` | int | Current stagnation counter value |
| `cooldown_remaining` | int | Steps remaining in post-burst cooldown |

---

## 3. Intervention Logic

### 3.1 Recombination Burst

**Trigger condition** (checked only when heavy metrics are available, i.e. `field_quality["diversity"] is not None`):

```
diversity < diversity_low
AND conflict > conflict_high
AND _cooldown_steps_remaining == 0
```

If both conditions hold: `_stagnation_count += 1`. Otherwise reset to 0.

When `_stagnation_count >= stagnation_window`:
1. Back up each agent's `config.conflict_threshold` into `_original_conflict_thresholds`
2. Set each agent's `config.conflict_threshold = burst_conflict_threshold`
3. Set `_burst_steps_remaining = burst_duration`
4. Reset `_stagnation_count = 0`

**Burst lifecycle** (decremented every `step()` call, in this order):

1. If `_burst_steps_remaining > 0`: decrement by 1.
   - If it reaches 0 after decrement: restore each agent's `config.conflict_threshold` from `_original_conflict_thresholds`, then set `_cooldown_steps_remaining = burst_cooldown`.
2. Else if `_cooldown_steps_remaining > 0`: decrement by 1. Stagnation counter is frozen at 0 and not advanced this cycle.
3. Stagnation check and burst-fire check run **after** the above decrements (so a burst fires on the step immediately following `stagnation_window` consecutive stagnant cycles, not during them).

**Ordering invariant:** `_burst_steps_remaining > 0` and `_cooldown_steps_remaining > 0` are mutually exclusive; only one can be non-zero at any time.

**Effect:** `conflict_threshold` lowered to near-zero means any non-trivial conflict level fires the conflict recombination trigger in each agent, producing a burst of recombination attempts across the population.

### 3.2 Adoption Scaling (kappa_0)

Applied every `step()` call when heavy metrics available (`diversity` not None):

1. If `_diversity_ema is None` (first heavy-metric step): set `_diversity_ema = diversity` and skip nudge this cycle (no trend yet to compare against).
2. Otherwise, update EMA first, then compare:
   - `_diversity_ema = kappa_0_ema_alpha * diversity + (1 - kappa_0_ema_alpha) * _diversity_ema`
   - If `diversity > _diversity_ema` (trend improving): nudge each agent's `kappa_0` up by `kappa_0_ema_alpha * (kappa_0_max - current_kappa_0)`, clamped to `[kappa_0_min, kappa_0_max]`
   - If `diversity < _diversity_ema` (trend falling): nudge each agent's `kappa_0` down by `kappa_0_ema_alpha * (current_kappa_0 - kappa_0_min)`, clamped to `[kappa_0_min, kappa_0_max]`
   - If `diversity == _diversity_ema`: no nudge.

**Note:** The EMA is updated **before** the comparison so the comparison reflects the new smoothed value, not the old one. This means on step 2, the EMA has already incorporated the first two diversity readings before any nudge fires.

**Effect:** When diversity is increasing (novelty working), the population becomes more receptive to new patterns. When falling (converging), it becomes more conservative.

### 3.3 Conflict Scale Damping (beta_c)

Applied every `step()` call when heavy metrics available:

- If `conflict > conflict_high`: `_conflict_persistent_cycles += 1`
- Else: reset `_conflict_persistent_cycles = 0`

When `_conflict_persistent_cycles >= stagnation_window` (same threshold, reused for simplicity):
- Apply: `agent.config.beta_c = max(beta_c_min, agent.config.beta_c * beta_c_decay)`
- If `agent.config` does not have a `beta_c` attribute, skip that agent silently (graceful degradation).

No backup is taken — damping is a one-way ratchet (see Section 6).

**Note:** `beta_c` damping is NOT reversed automatically (unlike burst). It persists until the agent's natural dynamics re-learn an appropriate conflict scale. This reflects the design intent: damping prevents "sticky inaccurate" attractors by permanently softening the conflict penalty until the population self-corrects.

---

## 4. MultiAgentOrchestrator Integration

### 4.1 Constructor Change

```python
def __init__(self, agents, field: PatternField, seed_pattern=None, groups=None, monitor=None, strategist=None):
```

`strategist: RecombinationStrategist | None = None` added as new keyword argument. Fully backward compatible.

### 4.2 step() Change

After `field_quality` is computed from `monitor.step()` (or `{}` if no monitor):

```python
interventions = (
    self.strategist.step(self._t, field_quality, self.agents)
    if self.strategist is not None
    else {}
)
```

Return dict gains a top-level `"interventions"` key:

```python
return {**metrics, "field_quality": field_quality, "interventions": interventions}
```

### 4.3 Dependency

The Strategist is most useful when a `StructuralLawMonitor` is also present (it reads `diversity` and `conflict` from `field_quality`). However, it degrades gracefully: if `field_quality` is empty or `diversity` is `None`, stagnation counting is skipped and no burst fires. Adoption scaling and beta_c damping also require non-None diversity. This means the Strategist can be composed without a monitor (all interventions simply never fire).

### 4.4 Typical Usage

```python
store = SQLiteStore("runs/experiment.db")
agents = [Agent(config, store=store) for _ in range(4)]
monitor = StructuralLawMonitor(store, T_monitor=10)
strategist = RecombinationStrategist(diversity_low=0.5, stagnation_window=3)
orchestrator = MultiAgentOrchestrator(agents, field=field, monitor=monitor, strategist=strategist)
```

---

## 5. Testing Strategy

### RecombinationStrategist (`tests/monitor/test_recombination_strategist.py`)

All tests use a stub `field_quality` dict and a list of mock agents with mutable `.config` attributes.

- `test_no_intervention_when_healthy` — high diversity + low conflict → no config mutations, `burst_active=False`
- `test_burst_fires_after_stagnation_window` — N consecutive low-diversity/high-conflict steps → `conflict_threshold` lowered on all agents, `burst_active=True`
- `test_burst_restores_config_after_duration` — after `burst_duration` steps → original `conflict_threshold` restored
- `test_burst_cooldown_prevents_retrigger` — burst fires, stagnation immediately resumes → burst does NOT re-fire during cooldown period
- `test_stagnation_skipped_when_diversity_none` — `field_quality["diversity"] = None` → stagnation counter does not advance
- `test_kappa_0_rises_when_diversity_improving` — diversity above EMA → `kappa_0` nudged toward `kappa_0_max`
- `test_kappa_0_falls_when_diversity_falling` — diversity below EMA → `kappa_0` nudged toward `kappa_0_min`
- `test_kappa_0_clamped_to_bounds` — `kappa_0` never exceeds `kappa_0_max` or falls below `kappa_0_min`
- `test_beta_c_damped_when_conflict_persists` — conflict > `conflict_high` for `stagnation_window` cycles → `beta_c` reduced
- `test_beta_c_floored_at_minimum` — repeated damping → `beta_c` never falls below `beta_c_min`
- `test_interventions_dict_always_present` — returned dict always has `burst_active`, `kappa_0`, `beta_c_scaled`, `stagnation_count`, `cooldown_remaining`
- `test_orchestrator_with_no_strategist` — orchestrator with `strategist=None` returns `interventions == {}`
- `test_orchestrator_strategist_integrated` — orchestrator with monitor + strategist returns both `field_quality` and `interventions`
- `test_empty_agents_list` — `step()` with `agents=[]` returns valid interventions dict, no errors
- `test_agent_missing_beta_c_skipped` — agent config without `beta_c` attribute is skipped silently during damping
- `test_kappa_0_no_nudge_on_first_heavy_step` — first step with non-None diversity initialises EMA but does not nudge `kappa_0`
- `test_burst_fires_on_step_after_stagnation_window_not_during` — stagnation count reaches window on step N; burst parameters applied on step N+1

---

## 6. What Is NOT in Scope

- Cross-agent recombination (parent patterns drawn from different agents' stores) — achieved naturally via PatternField communication and copy semantics
- Automatic `beta_c` restoration — damping is a one-way ratchet by design
- Per-agent burst override (some agents burst, others don't) — all agents receive the same intervention
- Visualisation or dashboard integration
- Strategist-triggered pattern deletion or pruning
