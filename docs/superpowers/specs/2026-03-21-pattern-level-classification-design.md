# Pattern Level Classification Design Spec

**Date:** 2026-03-21

## Goal

Add HPM hierarchical level (1–5) as a stored attribute of every pattern. Level is an emergent property computed during the evaluation phase from structural metrics and pattern density. Once classified, level modifies the per-pattern stickiness bias (`kappa_D`), enabling the weight update to protect deep structural patterns more aggressively than surface patterns. Level also serves as a gate for future operations (Recombination Operator targets Level 4+; externalisation targets Level 5).

## Background

HPM's five-level hierarchy describes the progressive abstraction of patterns from surface feature correlations (Level 1) through relational laws (Level 3–4) to generative primitives (Level 5). The existing evaluator pipeline already captures the signals needed to classify level — connectivity and compress from `GaussianPattern`, and density `D(h_i)` from `PatternDensity`. This spec wires those signals into a persistent `level` attribute on every pattern and feeds the level back into the `MetaPatternRule` weight update via a per-level `kappa_D` table.

Level 5 ("Generative / Expert") is defined by **Critical Density**: the pattern has achieved high structural coherence, evaluator saturation (low change rate), and social confirmation simultaneously — i.e. `D(h_i) > 0.85` — making it a stable candidate for recombination, even before the Recombination Operator is built.

## Level Definitions

| Level | Name | Classification Criterion | Functional State |
|---|---|---|---|
| 1 | Surface-Feature | Default (no other criterion met) | Fragile, context-bound |
| 2 | Local Structural | `connectivity > 0.3` | Recognising local repetitions |
| 3 | Relational | `connectivity > 0.5` AND `compress > 0.4` | Linking disparate features |
| 4 | Abstract Structural | `connectivity > 0.7` AND `compress > 0.6` | High-density structural law |
| 5 | Generative | `density > 0.85` AND `connectivity > 0.8` AND `compress > 0.7` | Generative building block |

Classification is evaluated top-down (Level 5 checked first, Level 1 is the fallback). All thresholds use strict `>` comparisons — a pattern sitting exactly on a threshold stays at the lower level. This is intentional: boundary patterns are ambiguous and should not be elevated prematurely. All thresholds are configurable via `AgentConfig`.

### Temporal stability note

Premature level elevation is naturally resisted by the density formula: `D(h_i)` incorporates saturation (`1 - novelty`) as a component. A pattern that briefly achieves high structural scores but is still changing rapidly will have low saturation → low density → cannot reach the Level 5 density threshold. No additional smoothing is required.

## Computation Sequence in Agent.step()

```
1. epistemic_accs, e_affs, last_capacity() computed (existing)
2. field_freqs, freq_totals, e_socs, e_costs computed (existing)
3. densities computed via PatternDensity.compute() (existing, Phase D1)
4. NEW: for each pattern i: p.level = self.level_classifier.compute_level(p, densities[i])
5. NEW: kappa_d_per_pattern[i] = self.config.kappa_d_levels[p.level - 1]
6. totals computed (existing)
7. MetaPatternRule.step(patterns, weights, totals,
                        densities=densities,
                        kappa_d_per_pattern=kappa_d_per_pattern)
8. Prune + persist (existing) — level preserved through GaussianPattern.update() and to_dict()
9. Field registration (existing)
```

Level is assigned **after** densities are computed and **before** the weight update, so the correct level-specific `kappa_D` is applied in the same step that the pattern earns it.

`level_mean` and `level_distribution` in the return dict are computed from the **pre-prune** `patterns` list (same list used for densities and weight updates), consistent with how `density_mean` and other per-pattern metrics are computed.

## Weight Update Formula

```
w_i(t+1) = w_i(t)
          + eta * (Total_i - Total_bar) * w_i         # replicator (D5)
          + kappa_D(level_i) * D(h_i) * w_i           # level-aware density bias
          - beta_c * sum_j kappa_ij * w_i * w_j        # conflict inhibition (D5)
```

`kappa_D(level_i) = kappa_d_levels[level_i - 1]` — a lookup into a 5-element list. All entries default to `0.0`, so existing behaviour is fully preserved when `kappa_d_levels` is unset.

The scalar `kappa_D` on `MetaPatternRule` remains as the global fallback when `kappa_d_per_pattern` is not supplied.

## Components

### Modified: `hpm/patterns/base.py`

Add `level: int` to the `Pattern` Protocol:

```python
class Pattern(Protocol):
    id: str
    level: int   # HPM hierarchy level 1–5; default 1
    # ... existing methods unchanged
```

### Modified: `hpm/patterns/gaussian.py`

Add `level: int = 1` constructor parameter after the existing `id` parameter, stored as `self.level`. The full updated constructor signature:

```python
def __init__(self, mu: np.ndarray, sigma: np.ndarray, id: str | None = None, level: int = 1):
    self.id = id or str(uuid.uuid4())
    self.mu = np.array(mu, dtype=float)
    self.sigma = np.array(sigma, dtype=float)
    self.level = level
    self._n_obs: int = 0
```

Preserved through `update()` — pass `id=self.id` and `level=self.level` to the new instance:

```python
def update(self, x: np.ndarray) -> 'GaussianPattern':
    n = self._n_obs + 1
    new_mu = (self.mu * self._n_obs + x) / n
    new_p = GaussianPattern(new_mu, self.sigma.copy(), id=self.id, level=self.level)
    new_p._n_obs = n
    return new_p
```

Add `level` to `to_dict()` and restore it in `from_dict()` — this is the persistence path for the SQLite store, which serialises patterns as JSON:

```python
def to_dict(self) -> dict:
    return {
        'type': 'gaussian',
        'id': self.id,
        'mu': self.mu.tolist(),
        'sigma': self.sigma.tolist(),
        'n_obs': self._n_obs,
        'level': self.level,
    }

@classmethod
def from_dict(cls, d: dict) -> 'GaussianPattern':
    p = cls(np.array(d['mu']), np.array(d['sigma']), id=d['id'],
            level=d.get('level', 1))   # default 1 for existing stored patterns
    p._n_obs = d['n_obs']
    return p
```

`d.get('level', 1)` ensures existing stored patterns (without a `level` key) default to Level 1 transparently — no database migration required.

Existing callers `GaussianPattern(mu=..., sigma=...)` default to `level=1` — no test changes needed.

### New: `hpm/patterns/classifier.py`

```python
class HPMLevelClassifier:
    """
    Assigns an HPM level (1–5) from structural metrics and pattern density.

    All thresholds are configurable. Classification is evaluated top-down
    (Level 5 checked first; Level 1 is the default fallback).
    All comparisons use strict > (not >=); patterns on a threshold boundary
    stay at the lower level.
    """

    def __init__(
        self,
        # Level 5 (Generative / Critical Density)
        l5_density: float = 0.85,
        l5_conn: float = 0.80,
        l5_comp: float = 0.70,
        # Level 4 (Abstract Structural)
        l4_conn: float = 0.70,
        l4_comp: float = 0.60,
        # Level 3 (Relational)
        l3_conn: float = 0.50,
        l3_comp: float = 0.40,
        # Level 2 (Local Structural)
        l2_conn: float = 0.30,
    ):
        ...

    def compute_level(self, pattern, density: float) -> int:
        """Return HPM level (1–5) for the given pattern and its current density."""
        conn = pattern.connectivity()
        comp = pattern.compress()
        if density > self.l5_density and conn > self.l5_conn and comp > self.l5_comp:
            return 5
        if conn > self.l4_conn and comp > self.l4_comp:
            return 4
        if conn > self.l3_conn and comp > self.l3_comp:
            return 3
        if conn > self.l2_conn:
            return 2
        return 1
```

### Modified: `hpm/patterns/__init__.py`

Re-export `HPMLevelClassifier` alongside `GaussianPattern`.

### Modified: `hpm/config.py`

Update the import line — `field` is needed for the `kappa_d_levels` default factory:

```python
from dataclasses import dataclass, field
```

New fields (all backward-compatible defaults):

```python
# Level classifier thresholds
l5_density: float = 0.85
l5_conn: float = 0.80
l5_comp: float = 0.70
l4_conn: float = 0.70
l4_comp: float = 0.60
l3_conn: float = 0.50
l3_comp: float = 0.40
l2_conn: float = 0.30
# Per-level kappa_D table (index 0 = Level 1, index 4 = Level 5)
kappa_d_levels: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
```

### Modified: `hpm/dynamics/meta_pattern_rule.py`

Add `kappa_d_per_pattern=None` to `step()`:

```python
def step(self, patterns, weights, totals, densities=None, kappa_d_per_pattern=None) -> np.ndarray:
```

In the per-pattern loop:

```python
if densities is not None:
    kappa_d_i = kappa_d_per_pattern[i] if kappa_d_per_pattern is not None else self.kappa_D
    density_bias = kappa_d_i * densities[i] * weights[i]
else:
    density_bias = 0.0
new_weights[i] = weights[i] + replicator - conflict + density_bias
```

External callers that pass only `densities` (without `kappa_d_per_pattern`) continue to use the scalar `self.kappa_D` — no behaviour change for those callers. `Agent.step()` is a modified caller: it now always constructs and passes `kappa_d_per_pattern` (derived from pattern levels).

### Modified: `hpm/agents/agent.py`

1. Import `HPMLevelClassifier` from `..patterns.classifier`
2. Instantiate in `__init__` from config thresholds:
   ```python
   self.level_classifier = HPMLevelClassifier(
       l5_density=config.l5_density,
       l5_conn=config.l5_conn,
       l5_comp=config.l5_comp,
       l4_conn=config.l4_conn,
       l4_comp=config.l4_comp,
       l3_conn=config.l3_conn,
       l3_comp=config.l3_comp,
       l2_conn=config.l2_conn,
   )
   ```
3. In `step()`, after densities computed:
   ```python
   for p, d in zip(patterns, densities):
       p.level = self.level_classifier.compute_level(p, d)
   kappa_d_per_pattern = [self.config.kappa_d_levels[p.level - 1] for p in patterns]
   ```
4. Pass `kappa_d_per_pattern` to `self.dynamics.step()`
5. Add to return dict (computed from the pre-prune `patterns` list):
   ```python
   'level_mean': float(np.mean([p.level for p in patterns])) if patterns else 0.0,
   'level_distribution': {lvl: sum(1 for p in patterns if p.level == lvl) for lvl in range(1, 6)},
   ```

Note: `p.level` is mutated in place before the prune loop. After `p.update(x)` returns a new instance, `level` is passed via the constructor (`level=self.level`) and written to the store via `to_dict()`.

### No changes required: `hpm/store/sqlite.py`

The SQLite store persists patterns exclusively via `pattern.to_dict()` / `pattern_from_dict()`. Adding `level` to `GaussianPattern.to_dict()` / `from_dict()` is sufficient for full persistence — no schema migration or SQL column addition is needed. Existing stored patterns (without a `level` key in JSON) will default to `level=1` via `d.get('level', 1)` in `from_dict()`.

## New Config Parameters

| Field | Default | Meaning |
|---|---|---|
| `l5_density` | 0.85 | Density threshold for Level 5 |
| `l5_conn` | 0.80 | Connectivity threshold for Level 5 |
| `l5_comp` | 0.70 | Compress threshold for Level 5 |
| `l4_conn` | 0.70 | Connectivity threshold for Level 4 |
| `l4_comp` | 0.60 | Compress threshold for Level 4 |
| `l3_conn` | 0.50 | Connectivity threshold for Level 3 |
| `l3_comp` | 0.40 | Compress threshold for Level 3 |
| `l2_conn` | 0.30 | Connectivity threshold for Level 2 |
| `kappa_d_levels` | [0,0,0,0,0] | Per-level kappa_D weights (index 0=L1, 4=L5) |

## Backward Compatibility

- `kappa_d_levels` all 0.0 → `kappa_d_per_pattern` all 0.0 → density bias zero → existing behaviour unchanged
- `level=1` default on GaussianPattern → all existing tests pass without modification
- `kappa_d_per_pattern=None` in MetaPatternRule → uses scalar `self.kappa_D` as before (for external callers)
- `d.get('level', 1)` in `from_dict()` → existing stored patterns without `level` field default to Level 1 transparently — no migration needed

## Testing

### Unit tests: `tests/patterns/test_level_classifier.py`

- `test_level_1_when_all_metrics_low` — zero connectivity → Level 1
- `test_level_2_from_connectivity_only` — conn > 0.3, comp = 0 → Level 2
- `test_level_3_requires_both_conn_and_comp` — conn > 0.5 AND comp > 0.4 → Level 3
- `test_level_4_thresholds` — conn > 0.7 AND comp > 0.6 → Level 4
- `test_level_5_requires_high_density_plus_structural` — density > 0.85 AND conn > 0.8 AND comp > 0.7 → Level 5
- `test_level_5_not_reached_by_density_alone` — high density but low structural → not Level 5
- `test_level_4_not_upgraded_to_5_without_density` — conn/comp meet L5 structural threshold but density low → Level 4
- `test_custom_thresholds_honoured` — non-default thresholds change classification
- `test_boundary_values_stay_at_lower_level` — conn exactly equal to a threshold (e.g. conn == 0.3) → stays at Level 1, not elevated to Level 2 (strict > not >=)

### Unit tests: `tests/patterns/test_gaussian_level.py`

- `test_default_level_is_1` — new GaussianPattern defaults to level=1
- `test_level_preserved_through_update` — pattern.update(x) inherits level
- `test_level_settable` — pattern.level = 3 persists
- `test_level_round_trips_through_to_dict_from_dict` — pattern with level=4 → to_dict() → from_dict() → level is 4
- `test_from_dict_defaults_level_to_1_when_key_absent` — dict without 'level' key → from_dict() → level is 1

### Integration tests: `tests/agents/test_agent_level.py`

- `test_agent_step_includes_level_mean_and_distribution` — step dict has both new keys
- `test_level_distribution_sums_to_n_patterns` — sanity check: sum of level_distribution values == number of pre-prune patterns
- `test_kappa_d_levels_config_reaches_meta_pattern_rule` — agent with non-zero `kappa_d_levels` produces `kappa_d_per_pattern` consistent with pattern levels
- `test_level_written_to_store_after_step` — after step, patterns queried from store have correct level (exercises the full to_dict/from_dict round-trip via the store)

## What This Is Not

- Not a recombination operator (Gap 3, separate sub-project) — Level 5 marks eligibility only
- Not a change to external substrate logic — externalisation by level is a future phase
- Not a multi-level pattern hierarchy with parent-child links — levels are independent scalar labels
