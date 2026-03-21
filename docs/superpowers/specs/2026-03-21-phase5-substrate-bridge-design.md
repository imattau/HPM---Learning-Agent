# Phase 5: Substrate Bridge Agent Design Specification

**Date:** 2026-03-21
**Status:** Draft v1

---

## Overview

The Substrate Bridge Agent ("The Translator") is a Phase 5 component that anchors internal GaussianPattern weights to external symbolic systems. It prevents "echo chamber" effects in multi-agent populations — where agents over-reinforce internally consistent but externally ungrounded patterns — by periodically querying ExternalSubstrates and adjusting weights based on real-world field frequency.

The Bridge Agent operates as a cadence-gated post-step processor in `MultiAgentOrchestrator`. Every `T_substrate` steps it:
1. Queries Level 3+ patterns from the shared `SQLiteStore`
2. Computes `field_frequency()` for each pattern against a connected `ExternalSubstrate`
3. Applies a standard alignment boost to all queried patterns
4. Applies an echo-chamber grounding penalty to ungrounded patterns when `StructuralLawMonitor` reports high redundancy

It writes updated weights directly to `SQLiteStore` via `update_weight()` and renormalises per-agent weights after each pass.

---

## 1. File Structure

```
hpm/substrate/bridge.py                  # new: SubstrateBridgeAgent class
tests/substrate/test_bridge.py           # new: unit + integration tests
hpm/substrate/__init__.py                # existing: add SubstrateBridgeAgent export
hpm/agents/multi_agent.py               # existing: add bridge=None kwarg + call
```

---

## 2. SubstrateBridgeAgent

### 2.1 Constructor

```python
def __init__(
    self,
    substrate: ExternalSubstrate,
    store,                              # shared SQLiteStore (or any PatternStore)
    T_substrate: int = 20,
    min_bridge_level: int = 3,
    alpha: float = 0.1,
    gamma: float = 0.2,
    redundancy_threshold: float = 0.3,
    frequency_low_threshold: float = 0.2,
    cache_distance_threshold: float = 0.05,
):
    self._substrate = substrate
    self._store = store
    self.T_substrate = T_substrate
    self.min_bridge_level = min_bridge_level
    self.alpha = alpha
    self.gamma = gamma
    self.redundancy_threshold = redundancy_threshold
    self.frequency_low_threshold = frequency_low_threshold
    self.cache_distance_threshold = cache_distance_threshold
    self._t = 0
    self._freq_cache: dict[str, tuple] = {}
```

| Parameter | Default | Role |
|-----------|---------|------|
| `substrate` | required | Any `ExternalSubstrate` instance (Wikipedia, Linguistic, Math, etc.) |
| `store` | required | Shared `SQLiteStore` — held as `self._store`, not passed to `step()` |
| `T_substrate` | 20 | Steps between substrate query passes |
| `min_bridge_level` | 3 | Minimum pattern level included in frequency checks |
| `alpha` | 0.1 | Alignment boost scale: `w × (1 + alpha × f_freq)` |
| `gamma` | 0.2 | Echo-chamber grounding penalty: `w × (1 - gamma)` |
| `redundancy_threshold` | 0.3 | Redundancy level (from Librarian) above which penalty pass activates |
| `frequency_low_threshold` | 0.2 | `f_freq` below which a pattern is considered "ungrounded" |
| `cache_distance_threshold` | 0.05 | L2 norm below which cached `f_freq` is reused without re-querying substrate |

### 2.2 Internal State

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_t` | int | Internal step counter (incremented every `step()` call) |
| `_freq_cache` | dict[str, tuple[np.ndarray, float]] | Maps `pattern.id → (cached_mu, f_freq)`. In-memory only; resets on agent recreation. |

### 2.3 Main Method

```python
def step(self, step_t: int, field_quality: dict) -> dict:
    """
    Called by MultiAgentOrchestrator after strategist.step().

    Args:
        step_t:        Current orchestrator step counter.
        field_quality: Dict from StructuralLawMonitor.step() (or {}).

    Returns:
        bridge_report dict (empty {} on non-cadence steps).

    Note: store is held as self._store (passed to constructor), not as a parameter here.
    This is consistent with how StructuralLawMonitor holds its store reference.
    """
```

Returns `bridge_report` dict with the following keys (only on cadence steps):

| Key | Type | Meaning |
|-----|------|---------|
| `patterns_checked` | int | Number of Level 3+ patterns queried this pass |
| `cache_hits` | int | Number of patterns whose `f_freq` was served from cache |
| `echo_chamber_penalty_applied` | bool | Whether the penalty pass ran this cycle |
| `mean_field_frequency` | float | Mean `f_freq` across all checked patterns; 0.0 if none |

On non-cadence steps, returns `{}`.

---

## 3. Step Logic

### 3.1 Cadence Gate

```python
self._t += 1
if self._t % self.T_substrate != 0:
    return {}
```

### 3.2 Snapshot

Call `self._store.query_all()` → list of `(pattern, weight, agent_id)` triples. Filter to `pattern.level >= min_bridge_level`. If no candidates, return early with a zeroed `bridge_report`.

### 3.3 Frequency Cache

For each candidate pattern:

1. Compute cache key: `pattern.id`
2. If `pattern.id` in `_freq_cache`:
   - Compute `np.linalg.norm(pattern.mu - cached_mu)`
   - If `< cache_distance_threshold`: cache hit — reuse `f_freq`, increment `cache_hits`
   - Else: cache miss — re-query substrate
3. If not cached: query `self.substrate.field_frequency(pattern)`, store `(pattern.mu.copy(), f_freq)` in `_freq_cache`

### 3.4 Standard Alignment Pass

For every candidate pattern with its computed `f_freq`:

```python
new_weight = weight * (1.0 + self.alpha * f_freq)
self._store.update_weight(pattern.id, new_weight)
```

This boosts all Level 3+ patterns proportionally to their external grounding. A pattern with `f_freq=0.0` receives no boost (multiplier = 1.0). A pattern with `f_freq=1.0` receives maximum boost (multiplier = `1 + alpha`).

### 3.5 Echo-Chamber Audit

Only runs if ALL of the following are true:
- `field_quality.get("redundancy") is not None`
- `field_quality["redundancy"] > redundancy_threshold`

If active, for each candidate pattern where `f_freq < frequency_low_threshold`:

```python
# Read the already-updated weight from the standard pass
current_weight = (updated weight from standard pass)
penalised_weight = current_weight * (1.0 - self.gamma)
self._store.update_weight(pattern.id, penalised_weight)
```

Set `echo_chamber_penalty_applied = True` in the report.

**Note:** The penalty is applied to the weight already modified by the standard pass, not the original weight. Order matters: boost first, penalise second.

### 3.6 Per-Agent Normalisation

After all weight updates, renormalise per agent so each agent's pattern weights sum to 1.0:

1. For each unique `agent_id` in the candidate set: call `self._store.query(agent_id)` to get current weights
2. Compute `total = sum(weights)`
3. If `total > 0`: for each pattern, call `self._store.update_weight(pattern.id, weight / total)`

Normalisation is independent per agent — agent A's weight mass does not affect agent B's.

---

## 4. MultiAgentOrchestrator Integration

### 4.1 Constructor Change

```python
def __init__(self, agents, field: PatternField, seed_pattern=None, groups=None, monitor=None, strategist=None, bridge=None):
```

`bridge: SubstrateBridgeAgent | None = None` added as new keyword argument. Fully backward compatible.

### 4.2 step() Change

After `strategist.step()`, add:

```python
bridge_report = (
    self.bridge.step(self._t, field_quality)
    if self.bridge is not None
    else {}
)
```

The Bridge Agent holds the shared `SQLiteStore` as `self._store` (passed at construction time). The orchestrator only passes `step_t` (its internal `_t` counter) and `field_quality` (from monitor). This is consistent with how `StructuralLawMonitor` holds its store reference.

The orchestrator's return dict gains a `"bridge_report"` key:

```python
return {**metrics, "field_quality": field_quality, "interventions": interventions, "bridge_report": bridge_report}
```

### 4.3 Typical Usage

```python
store = SQLiteStore("runs/experiment.db")
substrate = LinguisticSubstrate(feature_dim=32)
agents = [Agent(config, store=store) for _ in range(4)]
monitor = StructuralLawMonitor(store, T_monitor=10)
strategist = RecombinationStrategist()
bridge = SubstrateBridgeAgent(substrate, store, T_substrate=20, alpha=0.1, gamma=0.2)
orchestrator = MultiAgentOrchestrator(
    agents, field=field,
    monitor=monitor, strategist=strategist, bridge=bridge
)
```

---

## 5. Testing Strategy

Tests in `tests/substrate/test_bridge.py`. Use a stub `ExternalSubstrate` that returns fixed vectors (no HTTP calls). Use a real `SQLiteStore` via `tmp_path`.

**Stub substrate for tests:**
```python
class StubSubstrate:
    def __init__(self, freq=0.5):
        self._freq = freq
        self.call_count = 0
    def fetch(self, query): return []
    def field_frequency(self, pattern):
        self.call_count += 1
        return self._freq
    def stream(self): return iter([])
```

**Test cases:**

- `test_no_op_on_non_cadence_step` — step_t=1 with T_substrate=10 → returns `{}`, no weight changes
- `test_standard_boost_applied` — Level 3 pattern with f_freq=0.5, alpha=0.1 → weight multiplied by 1.05
- `test_zero_freq_no_boost` — f_freq=0.0 → weight unchanged (multiplier = 1.0)
- `test_below_min_level_skipped` — Level 2 pattern not queried or updated; substrate.call_count == 0
- `test_echo_chamber_penalty_applied` — redundancy > threshold AND f_freq < frequency_low → penalty applied after boost
- `test_echo_chamber_skipped_when_redundancy_none` — `field_quality["redundancy"] = None` → no penalty pass
- `test_echo_chamber_skipped_when_redundancy_low` — redundancy < threshold → no penalty even if f_freq is low
- `test_cache_hit_avoids_substrate_call` — pattern queried at T_substrate, mu unchanged → substrate.call_count == 1 at 2×T_substrate
- `test_cache_miss_on_mu_change` — pattern mu changes beyond threshold → substrate re-queried at 2×T_substrate
- `test_weights_normalised_per_agent` — after boost, each agent's pattern weights sum to 1.0 (within float tolerance)
- `test_multi_agent_normalisation_independent` — two agents with different pattern counts normalised independently
- `test_bridge_report_keys` — cadence step returns dict with `patterns_checked`, `cache_hits`, `echo_chamber_penalty_applied`, `mean_field_frequency`
- `test_no_candidates_returns_zeroed_report` — all patterns below min_bridge_level → `patterns_checked=0`, no errors
- `test_orchestrator_no_bridge` — orchestrator with `bridge=None` returns `bridge_report == {}`
- `test_orchestrator_bridge_integrated` — orchestrator with bridge returns `bridge_report` with correct keys after T_substrate steps

---

## 6. What Is NOT in Scope

- Multi-substrate queries (querying Wikipedia AND Linguistic simultaneously) — single substrate per Bridge Agent in v1
- Substrate-triggered recombination (recombination is the Strategist's domain)
- Persistent frequency cache (cache is in-memory only, resets on restart)
- Pattern label extraction for query strings (uses `getattr(pattern, "label", str(pattern.id))` — same pattern as existing substrate implementations)
- Bridge Agent modifying agent configs (configs are the Strategist's domain; Bridge only touches weights)
