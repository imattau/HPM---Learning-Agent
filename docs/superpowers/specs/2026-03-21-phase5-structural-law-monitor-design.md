# Phase 5: Structural Law Monitor Design Specification

**Date:** 2026-03-21
**Status:** Draft v2 (post-review)

---

## Overview

The Structural Law Monitor ("Librarian") is a Phase 5 component that observes the shared pattern population across all HPM agents and computes Field Quality Metrics. It provides the first human-readable window into the agent population's learned laws, directly validating the HPM success criteria for Social Field Convergence (§9.5).

The monitor is implemented as a standalone `StructuralLawMonitor` class composed into `MultiAgentOrchestrator`. It requires a shared `SQLiteStore` so that all agents write to a single persistent store queryable by the monitor.

---

## 1. File Structure

```
hpm/store/sqlite.py                    # existing: SQLiteStore (already implemented — no changes needed)
hpm/monitor/__init__.py                # new: exports StructuralLawMonitor
hpm/monitor/structural_law.py          # new: StructuralLawMonitor class
tests/store/test_sqlite.py             # existing: add test_level_persisted, test_query_all_returns_agent_id
tests/monitor/__init__.py              # new: empty
tests/monitor/test_structural_law.py   # new: monitor metric and cadence tests
```

**Existing files modified:**
- `hpm/agents/multi_agent.py` — add `monitor=None` keyword parameter to `MultiAgentOrchestrator`
- `hpm/store/__init__.py` — export `SQLiteStore`

---

## 2. SQLiteStore

### 2.1 Purpose

`hpm/store/sqlite.py` already implements the `PatternStore` protocol using Python's stdlib `sqlite3` — no new dependencies. It is already used in tests and the existing implementation is retained as-is. This section documents its actual interface for implementors of `StructuralLawMonitor`.

### 2.2 Schema (actual, already implemented)

```sql
CREATE TABLE IF NOT EXISTS patterns (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    pattern_json TEXT NOT NULL,   -- JSON via pattern.to_dict() / pattern_from_dict()
    weight      REAL NOT NULL
);
```

`level` is embedded in `pattern_json` (via `GaussianPattern.to_dict()` which includes `"level": int`). No separate column or index is needed; `pattern.level` is accessed after deserialisation.

### 2.3 Constructor

```python
def __init__(self, db_path: str):
    """
    db_path: file path for the SQLite database (e.g. "runs/experiment.db").
    Use a tmp_path fixture for testing.
    Creates the database and table if they do not exist.
    """
```

### 2.4 Protocol Methods

All `PatternStore` protocol methods are implemented:

| Method | Behaviour |
|--------|-----------|
| `save(pattern, weight, agent_id)` | INSERT OR REPLACE into patterns table |
| `load(pattern_id) -> tuple[GaussianPattern, float]` | SELECT by id; raises KeyError if not found |
| `query(agent_id) -> list[tuple[GaussianPattern, float]]` | SELECT WHERE agent_id = ? |
| `delete(pattern_id)` | DELETE WHERE id = ? |
| `update_weight(pattern_id, weight)` | UPDATE weight WHERE id = ? |

**Additional method (monitor-specific, already implemented):**

```python
def query_all(self) -> list[tuple[GaussianPattern, float, str]]:
    """
    Return all patterns across all agents as (pattern, weight, agent_id) triples.
    Used by StructuralLawMonitor to compute population-wide metrics.
    """
```

### 2.5 Serialisation

Patterns serialised to JSON via `pattern.to_dict()` and deserialised via `pattern_from_dict()`. `level` is included in the JSON payload and recovered on load — no separate column required.

---

## 3. StructuralLawMonitor

### 3.1 Constructor

```python
def __init__(
    self,
    store: PatternStore,
    field: PatternField | None = None,
    T_monitor: int = 50,
    log_path: str | None = None,
    conflict_threshold: float = 0.5,
):
```

| Parameter | Role |
|-----------|------|
| `store` | Shared `SQLiteStore` (or any `PatternStore`) queried for population-wide metrics |
| `field` | Optional `PatternField` (reserved for future diversity queries; unused in v1) |
| `T_monitor` | Cadence for heavy metrics and console/log output (default 50 steps) |
| `log_path` | Path for newline-delimited JSON log; `None` disables logging |
| `conflict_threshold` | Threshold above which a pattern pair counts as conflicting |

### 3.2 Main Method

```python
def step(self, step_t: int, agents: list, total_conflict: float) -> dict:
    """
    Called by MultiAgentOrchestrator after each step.
    Returns field_quality dict (light metrics always present; heavy metrics
    present every T_monitor steps, else None).
    """
```

Internal logic:
1. Increment `self._t`
2. Compute **light metrics** from `store.query_all()`
3. If `self._t % T_monitor == 0`: compute **heavy metrics**, call `_print_table()`, call `_log_json()` (if `log_path` set)
4. Return merged dict

---

## 4. Metric Definitions

### 4.1 Light Metrics (every step)

| Key | Type | Definition |
|-----|------|-----------|
| `pattern_count` | int | Total patterns across all agents |
| `level_distribution` | dict[int, int] | Count of patterns at each level 1–5 |
| `level4plus_count` | int | Patterns with `level >= 4` |
| `level4plus_mean_weight` | float | Mean `w_i` for Level 4+ patterns; `0.0` if none |
| `conflict` | float | `total_conflict` passed in from MetaPatternRule |
| `stability_mean` | float | Mean `sigmoid(pattern.level / 5.0)` across all patterns |

`stability_mean` uses `sigmoid(x) = 1 / (1 + exp(-x))` applied to the normalised level as a proxy for pattern stability without recomputing full PatternDensity.

### 4.2 Heavy Metrics (every T_monitor steps)

| Key | Type | Definition |
|-----|------|-----------|
| `diversity` | float | Entropy of weight distribution: `-sum(w_i * log(w_i + 1e-9))` across all patterns |
| `redundancy` | float | Mean pairwise `sym_kl_normalised(p_i, p_j)` for Level 4+ pattern pairs; `0.0` if fewer than 2 such patterns |

`sym_kl_normalised` is imported from `hpm.dynamics.meta_pattern_rule` (already implemented).

**Between heavy-metric steps,** `field_quality["diversity"]` and `field_quality["redundancy"]` are `None`.

---

## 5. Console Output and JSON Log

### 5.1 Console Table (every T_monitor steps)

Printed to stdout via `print_results_table` from `benchmarks.common`:

```
Field Quality Report (step 150)
────────────────────────────────────────────────────────────────────────
Patterns   L4+   L4+ Weight   Diversity   Redundancy   Conflict   Stable
47         12    0.71         1.83        0.23         0.04       0.82
```

### 5.2 JSON Log (every T_monitor steps, if log_path set)

One JSON object per line appended to `log_path`:

```json
{"step": 150, "pattern_count": 47, "level_distribution": {"1": 8, "2": 12, "3": 15, "4": 9, "5": 3}, "level4plus_count": 12, "level4plus_mean_weight": 0.71, "conflict": 0.04, "stability_mean": 0.82, "diversity": 1.83, "redundancy": 0.23}
```

Newline-delimited JSON (NDJSON) — appendable without parsing the full file.

---

## 6. MultiAgentOrchestrator Integration

### 6.1 Constructor Change

```python
def __init__(self, agents, field: PatternField, seed_pattern=None, groups=None, monitor=None):
```

`monitor: StructuralLawMonitor | None = None` is added as a new keyword argument. The existing positional parameters (`field`, `seed_pattern`, `groups`) are unchanged — fully backward compatible. If `monitor` is `None`, monitoring is disabled with zero overhead.

### 6.2 step() Change

After the communication phase, before returning, aggregate `total_conflict` across all agents (sum of per-agent `step_metrics["total_conflict"]` values) and pass it to the monitor:

```python
total_conflict = sum(
    metrics[aid].get("total_conflict", 0.0) for aid in metrics
)
field_quality = (
    self.monitor.step(self._t, self.agents, total_conflict)
    if self.monitor is not None
    else {}
)
```

The orchestrator `step()` return dict gains a top-level `"field_quality"` key alongside the per-agent keys:

```python
return {**metrics, "field_quality": field_quality}
```

Note: agent_ids must not be `"field_quality"` — this is a reserved key in the return dict.

### 6.3 Typical Usage

```python
store = SQLiteStore("runs/experiment_1.db")
agents = [Agent(config, store=store) for _ in range(4)]
monitor = StructuralLawMonitor(store, T_monitor=50, log_path="runs/monitor.jsonl")
orchestrator = MultiAgentOrchestrator(agents, monitor=monitor)
```

---

## 7. Testing Strategy

### SQLiteStore (`tests/store/test_sqlite.py` — existing file, add cases)

`tests/store/test_sqlite.py` already covers `save/load`, `query`, `update_weight`, `delete`, `query_all`, and cross-connection persistence. Add the following cases to the existing file:

- `test_level_persisted`: save a pattern with `level=3`, query by agent_id, verify `pattern.level == 3`
- `test_query_all_returns_agent_id`: `query_all()` triples include the correct `agent_id` string as third element

### StructuralLawMonitor (`tests/monitor/test_structural_law.py`)

- `test_light_metrics_every_step`: after seeding patterns at known levels, step() returns correct level_distribution
- `test_heavy_metrics_at_cadence`: heavy metrics (diversity, redundancy) are None before T_monitor, present at T_monitor
- `test_no_log_when_log_path_none`: step() with log_path=None produces no file
- `test_json_log_appends`: two heavy-metric steps append two lines to log file
- `test_redundancy_zero_with_one_l4_pattern`: redundancy=0.0 when fewer than 2 Level 4+ patterns
- `test_stability_mean_in_range`: stability_mean in [0, 1]
- `test_diversity_zero_single_pattern`: diversity=0.0 with one pattern (weight=1.0, log(1.0)=0)

### MultiAgentOrchestrator (`existing tests/agents/test_multi_agent.py`)

- `test_monitor_none_no_overhead`: orchestrator with monitor=None returns empty field_quality
- `test_monitor_integrated`: orchestrator with monitor returns field_quality dict with light metrics

---

## 8. What Is NOT in Scope

- PostgreSQL store (SQLite covers the shared-store requirement; Postgres deferred)
- Pattern visualisation (rendering pattern mu as human-readable expression)
- Real-time dashboard or web UI
- Monitor-triggered recombination (Recombination Strategist is a separate Phase 5 agent)
- `PatternField` integration in v1 (field parameter reserved but unused)
