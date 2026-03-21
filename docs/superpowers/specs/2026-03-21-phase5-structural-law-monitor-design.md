# Phase 5: Structural Law Monitor Design Specification

**Date:** 2026-03-21
**Status:** Draft v1

---

## Overview

The Structural Law Monitor ("Librarian") is a Phase 5 component that observes the shared pattern population across all HPM agents and computes Field Quality Metrics. It provides the first human-readable window into the agent population's learned laws, directly validating the HPM success criteria for Social Field Convergence (§9.5).

The monitor is implemented as a standalone `StructuralLawMonitor` class composed into `MultiAgentOrchestrator`. It requires a shared `SQLiteStore` so that all agents write to a single persistent store queryable by the monitor.

---

## 1. File Structure

```
hpm/store/sqlite.py                    # new: SQLiteStore (shared persistent PatternStore)
hpm/monitor/__init__.py                # new: exports StructuralLawMonitor
hpm/monitor/structural_law.py          # new: StructuralLawMonitor class
tests/store/test_sqlite_store.py       # new: SQLiteStore protocol compliance tests
tests/monitor/test_structural_law.py   # new: monitor metric and cadence tests
```

**Existing files modified:**
- `hpm/agents/multi_agent.py` — add `monitor` parameter to `MultiAgentOrchestrator`
- `hpm/store/__init__.py` — export `SQLiteStore`

---

## 2. SQLiteStore

### 2.1 Purpose

Implements the existing `PatternStore` protocol using Python's stdlib `sqlite3` — no new dependencies. Enables all agents in a multi-agent run to share one persistent store, which the monitor can query globally.

### 2.2 Schema

```sql
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    agent_id   TEXT NOT NULL,
    mu         BLOB NOT NULL,   -- numpy array serialised with np.save (BytesIO)
    sigma      BLOB NOT NULL,   -- numpy array serialised with np.save (BytesIO)
    weight     REAL NOT NULL,
    level      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_agent ON patterns(agent_id);
CREATE INDEX IF NOT EXISTS idx_level ON patterns(level);
```

### 2.3 Constructor

```python
def __init__(self, path: str):
    """
    path: file path for the SQLite database (e.g. "runs/experiment.db").
    Use ":memory:" for in-process testing.
    Creates the database and table if they do not exist.
    """
```

### 2.4 Protocol Methods

Implements all methods of the `PatternStore` protocol:

| Method | Behaviour |
|--------|-----------|
| `save(pattern, weight, agent_id)` | INSERT OR REPLACE into patterns table |
| `query(agent_id) -> list[tuple[GaussianPattern, float]]` | SELECT WHERE agent_id = ? |
| `delete(pattern_id)` | DELETE WHERE pattern_id = ? |
| `update_weight(pattern_id, weight)` | UPDATE weight WHERE pattern_id = ? |

**Additional method (monitor-specific):**

```python
def query_all(self) -> list[tuple[GaussianPattern, float, str]]:
    """
    Return all patterns across all agents as (pattern, weight, agent_id) triples.
    Used exclusively by StructuralLawMonitor.
    """
```

### 2.5 Serialisation

`mu` and `sigma` serialised with `np.save` to `BytesIO`, stored as `BLOB`. Deserialised with `np.load` from `BytesIO`. `level` stored as `INTEGER`, updated by `save()` from `pattern.level`.

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
def __init__(self, agents, field=None, monitor=None):
```

`monitor: StructuralLawMonitor | None = None`. If `None`, monitoring disabled — zero overhead, fully backward compatible.

### 6.2 step() Change

After the communication phase, before returning:

```python
field_quality = (
    self.monitor.step(self._t, self.agents, total_conflict)
    if self.monitor is not None
    else {}
)
```

Add `"field_quality": field_quality` to the orchestrator step return dict.

### 6.3 Typical Usage

```python
store = SQLiteStore("runs/experiment_1.db")
agents = [Agent(config, store=store) for _ in range(4)]
monitor = StructuralLawMonitor(store, T_monitor=50, log_path="runs/monitor.jsonl")
orchestrator = MultiAgentOrchestrator(agents, monitor=monitor)
```

---

## 7. Testing Strategy

### SQLiteStore (`tests/store/test_sqlite_store.py`)

- `test_save_and_query_roundtrip`: save pattern, query by agent_id, verify mu/sigma/weight/level preserved
- `test_delete_removes_pattern`: save then delete, query returns empty
- `test_update_weight`: save then update_weight, query shows new weight
- `test_query_all_across_agents`: save patterns for two agent_ids, query_all returns all
- `test_memory_db`: `SQLiteStore(":memory:")` works for in-process testing
- `test_level_persisted`: pattern.level stored and recovered correctly

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
