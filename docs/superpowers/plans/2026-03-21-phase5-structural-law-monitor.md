# Phase 5: Structural Law Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `StructuralLawMonitor` — a population-level field quality observer composable into `MultiAgentOrchestrator` that computes light metrics every step and heavy metrics every T_monitor steps, printing a console table and optionally writing NDJSON logs.

**Architecture:** Standalone `StructuralLawMonitor` class in `hpm/monitor/structural_law.py` reads from a shared `SQLiteStore` via `query_all()`. Composed into `MultiAgentOrchestrator` as an optional `monitor=None` keyword parameter — zero overhead when disabled. Light metrics (level distribution, conflict, stability) computed every step; heavy metrics (diversity entropy, redundancy) computed every T_monitor steps.

**Tech Stack:** Python stdlib only (`sqlite3`, `json`, `math`). Imports `sym_kl_normalised` from existing `hpm.dynamics.meta_pattern_rule`. No new dependencies.

---

## File Structure

```
hpm/monitor/__init__.py                # new: exports StructuralLawMonitor
hpm/monitor/structural_law.py          # new: StructuralLawMonitor class
tests/monitor/__init__.py              # new: empty
tests/monitor/test_structural_law.py   # new: metric and cadence tests
tests/store/test_sqlite.py             # existing: add 2 new test cases
hpm/agents/multi_agent.py              # existing: add monitor=None kwarg
hpm/store/__init__.py                  # existing: export SQLiteStore
```

**Key existing files to read before starting:**
- `hpm/store/sqlite.py` — SQLiteStore with `query_all()` already implemented
- `hpm/dynamics/meta_pattern_rule.py` — `sym_kl_normalised(p1, p2)` function
- `hpm/agents/multi_agent.py` — MultiAgentOrchestrator.step() return dict structure

---

### Task 1: SQLiteStore additions and monitor package skeleton

**Files:**
- Modify: `tests/store/test_sqlite.py`
- Create: `hpm/monitor/__init__.py`
- Create: `tests/monitor/__init__.py`

- [ ] **Step 1: Add 2 tests to existing test_sqlite.py**

Read `tests/store/test_sqlite.py` first to understand fixture pattern. Then add at the end:

```python
def test_level_persisted(tmp_path):
    """Pattern.level is stored in JSON and recovered correctly."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    p.level = 3
    store.save(p, 0.5, "agent_a")
    records = store.query("agent_a")
    assert len(records) == 1
    recovered_pattern, _ = records[0]
    assert recovered_pattern.level == 3


def test_query_all_returns_agent_id(tmp_path):
    """query_all() triples include the correct agent_id as third element."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p1 = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    p2 = GaussianPattern(mu=np.ones(4), sigma=np.eye(4))
    store.save(p1, 0.6, "agent_a")
    store.save(p2, 0.4, "agent_b")
    triples = store.query_all()
    agent_ids = {aid for _, _, aid in triples}
    assert agent_ids == {"agent_a", "agent_b"}
```

- [ ] **Step 2: Run new tests to verify they pass**

```bash
uv run pytest tests/store/test_sqlite.py::test_level_persisted tests/store/test_sqlite.py::test_query_all_returns_agent_id -v
```

Expected: both PASS (SQLiteStore already supports these).

If `test_level_persisted` fails: check that `GaussianPattern.to_dict()` includes `"level"`. If not, add `"level": self.level` to the dict in `hpm/patterns/gaussian.py`.

- [ ] **Step 3: Create monitor package**

Create `hpm/monitor/__init__.py`:
```python
from .structural_law import StructuralLawMonitor

__all__ = ["StructuralLawMonitor"]
```

Create `tests/monitor/__init__.py` (empty file).

- [ ] **Step 4: Commit**

```bash
git add tests/store/test_sqlite.py hpm/monitor/__init__.py tests/monitor/__init__.py
git commit -m "feat: add SQLiteStore level/query_all tests and monitor package skeleton"
```

---

### Task 2: StructuralLawMonitor — light metrics

**Files:**
- Create: `hpm/monitor/structural_law.py`
- Create: `tests/monitor/test_structural_law.py`

- [ ] **Step 1: Write failing tests for light metrics**

Create `tests/monitor/test_structural_law.py`:

```python
import math
import json
import numpy as np
import pytest
from hpm.store.sqlite import SQLiteStore
from hpm.patterns.gaussian import GaussianPattern
from hpm.monitor.structural_law import StructuralLawMonitor


def _make_pattern(level: int, dim: int = 4) -> GaussianPattern:
    p = GaussianPattern(mu=np.random.randn(dim), sigma=np.eye(dim))
    p.level = level
    return p


def _seed_store(store, agent_id, levels):
    """Save one pattern per level value into the store."""
    n = len(levels)
    for i, lvl in enumerate(levels):
        p = _make_pattern(lvl)
        store.save(p, 1.0 / n, agent_id)


def test_light_metrics_every_step(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [1, 2, 4, 4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=100)

    result = monitor.step(step_t=1, agents=[], total_conflict=0.1)

    assert result["pattern_count"] == 5
    assert result["level_distribution"][4] == 2
    assert result["level_distribution"][5] == 1
    assert result["level4plus_count"] == 3
    assert 0.0 <= result["level4plus_mean_weight"] <= 1.0
    assert result["conflict"] == pytest.approx(0.1)
    assert 0.0 <= result["stability_mean"] <= 1.0


def test_heavy_metrics_none_before_cadence(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=10)

    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)

    assert result["diversity"] is None
    assert result["redundancy"] is None


def test_heavy_metrics_present_at_cadence(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=1)

    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)

    assert result["diversity"] is not None
    assert result["redundancy"] is not None
    assert result["diversity"] >= 0.0
    assert 0.0 <= result["redundancy"] <= 1.0


def test_stability_mean_in_range(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [1, 2, 3, 4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=100)
    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)
    assert 0.0 <= result["stability_mean"] <= 1.0


def test_redundancy_zero_with_one_l4_pattern(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4])  # only one Level 4+ pattern
    monitor = StructuralLawMonitor(store, T_monitor=1)
    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)
    assert result["redundancy"] == pytest.approx(0.0)


def test_no_log_when_log_path_none(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4])
    monitor = StructuralLawMonitor(store, T_monitor=1, log_path=None)
    monitor.step(step_t=1, agents=[], total_conflict=0.0)
    # No files created other than the DB
    files = list(tmp_path.iterdir())
    assert all(f.suffix == ".db" for f in files)


def test_json_log_appends(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4, 5])
    log_path = str(tmp_path / "monitor.jsonl")
    monitor = StructuralLawMonitor(store, T_monitor=1, log_path=log_path)
    monitor.step(step_t=1, agents=[], total_conflict=0.0)
    monitor.step(step_t=2, agents=[], total_conflict=0.0)
    with open(log_path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert "step" in entry
    assert "diversity" in entry


def test_diversity_zero_single_pattern(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(1)
    store.save(p, 1.0, "a1")
    monitor = StructuralLawMonitor(store, T_monitor=1)
    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)
    # Single pattern weight=1.0 → entropy = -1.0 * log(1.0) = 0.0
    assert result["diversity"] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitor/test_structural_law.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hpm.monitor.structural_law'`

- [ ] **Step 3: Create StructuralLawMonitor with light metrics**

Create `hpm/monitor/structural_law.py`:

```python
"""
hpm/monitor/structural_law.py — Structural Law Monitor ("The Librarian")

Observes the shared PatternStore population and computes Field Quality Metrics.
Integrated into MultiAgentOrchestrator as an optional monitor= parameter.

Light metrics: computed every step (inexpensive).
Heavy metrics: computed every T_monitor steps (O(n²) pairwise operations).
"""

import json
import math
from typing import Any

import numpy as np

from hpm.dynamics.meta_pattern_rule import sym_kl_normalised


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class StructuralLawMonitor:
    """
    Computes population-level Field Quality Metrics from a shared PatternStore.

    Args:
        store:              Shared PatternStore (SQLiteStore recommended).
        field:              Optional PatternField (reserved; unused in v1).
        T_monitor:          Cadence for heavy metrics and console/log output.
        log_path:           NDJSON log path; None disables logging.
        conflict_threshold: Threshold above which pattern pairs count as conflicting.
    """

    def __init__(
        self,
        store,
        field=None,
        T_monitor: int = 50,
        log_path: str | None = None,
        conflict_threshold: float = 0.5,
    ):
        self._store = store
        self._field = field
        self._T_monitor = T_monitor
        self._log_path = log_path
        self._conflict_threshold = conflict_threshold
        self._t = 0

    def step(self, step_t: int, agents: list, total_conflict: float) -> dict:
        """
        Called by MultiAgentOrchestrator after each step.

        Returns field_quality dict:
          - Light metrics always present.
          - Heavy metrics (diversity, redundancy) present every T_monitor steps; else None.
        """
        self._t += 1

        all_records = self._store.query_all()  # list of (pattern, weight, agent_id)
        patterns = [p for p, _, _ in all_records]
        weights = [w for _, w, _ in all_records]

        light = self._compute_light(patterns, weights, total_conflict)

        heavy_diversity = None
        heavy_redundancy = None

        if self._t % self._T_monitor == 0:
            heavy_diversity, heavy_redundancy = self._compute_heavy(patterns, weights)
            self._print_table(step_t, light, heavy_diversity, heavy_redundancy)
            if self._log_path is not None:
                self._log_json(step_t, light, heavy_diversity, heavy_redundancy)

        return {
            **light,
            "diversity": heavy_diversity,
            "redundancy": heavy_redundancy,
        }

    # ------------------------------------------------------------------
    # Light metrics
    # ------------------------------------------------------------------

    def _compute_light(self, patterns, weights, total_conflict: float) -> dict:
        level_dist = {lvl: 0 for lvl in range(1, 6)}
        for p in patterns:
            lvl = getattr(p, "level", 1)
            level_dist[min(max(lvl, 1), 5)] += 1

        l4plus = [(p, w) for p, w in zip(patterns, weights) if getattr(p, "level", 1) >= 4]
        l4plus_count = len(l4plus)
        l4plus_mean_weight = (
            float(np.mean([w for _, w in l4plus])) if l4plus else 0.0
        )

        stability_mean = (
            float(np.mean([_sigmoid(getattr(p, "level", 1) / 5.0) for p in patterns]))
            if patterns else 0.0
        )

        return {
            "pattern_count": len(patterns),
            "level_distribution": level_dist,
            "level4plus_count": l4plus_count,
            "level4plus_mean_weight": l4plus_mean_weight,
            "conflict": float(total_conflict),
            "stability_mean": stability_mean,
        }

    # ------------------------------------------------------------------
    # Heavy metrics
    # ------------------------------------------------------------------

    def _compute_heavy(self, patterns, weights) -> tuple[float, float]:
        # Diversity: entropy of weight distribution
        total_w = sum(weights)
        if total_w > 0:
            norm_weights = [w / total_w for w in weights]
        else:
            norm_weights = [1.0 / len(weights)] * len(weights) if weights else []
        diversity = -sum(w * math.log(w + 1e-9) for w in norm_weights) if norm_weights else 0.0

        # Redundancy: mean pairwise sym_kl_normalised for Level 4+ patterns
        l4plus_patterns = [p for p in patterns if getattr(p, "level", 1) >= 4]
        if len(l4plus_patterns) < 2:
            redundancy = 0.0
        else:
            sims = []
            for i in range(len(l4plus_patterns)):
                for j in range(i + 1, len(l4plus_patterns)):
                    sims.append(sym_kl_normalised(l4plus_patterns[i], l4plus_patterns[j]))
            redundancy = float(np.mean(sims)) if sims else 0.0

        return diversity, redundancy

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _print_table(self, step_t, light, diversity, redundancy):
        title = f"Field Quality Report (step {step_t})"
        cols = ["Patterns", "L4+", "L4+ Weight", "Diversity", "Redundancy", "Conflict", "Stable"]
        row = {
            "Patterns": str(light["pattern_count"]),
            "L4+": str(light["level4plus_count"]),
            "L4+ Weight": f"{light['level4plus_mean_weight']:.2f}",
            "Diversity": f"{diversity:.2f}" if diversity is not None else "—",
            "Redundancy": f"{redundancy:.2f}" if redundancy is not None else "—",
            "Conflict": f"{light['conflict']:.2f}",
            "Stable": f"{light['stability_mean']:.2f}",
        }
        col_widths = {c: max(len(c), len(row[c])) for c in cols}
        sep = "   "
        header = sep.join(c.ljust(col_widths[c]) for c in cols)
        total_width = max(len(title), len(header))
        print()
        print(title)
        print("─" * total_width)
        print(header)
        print("─" * total_width)
        print(sep.join(row[c].ljust(col_widths[c]) for c in cols))
        print()

    def _log_json(self, step_t, light, diversity, redundancy):
        entry = {
            "step": step_t,
            **light,
            "level_distribution": {str(k): v for k, v in light["level_distribution"].items()},
            "diversity": diversity,
            "redundancy": redundancy,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run all monitor tests**

```bash
uv run pytest tests/monitor/test_structural_law.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: 275+ passed, 9 skipped.

- [ ] **Step 6: Commit**

```bash
git add hpm/monitor/structural_law.py tests/monitor/test_structural_law.py hpm/monitor/__init__.py tests/monitor/__init__.py
git commit -m "feat: StructuralLawMonitor with light and heavy field quality metrics"
```

---

### Task 3: MultiAgentOrchestrator integration

**Files:**
- Modify: `hpm/agents/multi_agent.py`
- Modify: `hpm/store/__init__.py`

- [ ] **Step 1: Read existing MultiAgentOrchestrator**

Read `hpm/agents/multi_agent.py` to find:
- Constructor signature and existing parameters
- The `step()` method return dict structure
- Where the communication phase ends

- [ ] **Step 2: Write failing integration tests**

Add to `tests/monitor/test_structural_law.py`:

```python
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.field.field import PatternField


def _make_agent(store, dim=4):
    agent_id = f"agent_{id(store)}"
    config = AgentConfig(feature_dim=dim, agent_id=agent_id)
    return Agent(config, store=store)


def test_monitor_none_returns_empty_field_quality(tmp_path):
    """Orchestrator with monitor=None returns empty field_quality."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field, monitor=None)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("field_quality") == {}


def test_monitor_integrated_returns_light_metrics(tmp_path):
    """Orchestrator with monitor returns field_quality with light metrics."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_agent(store)
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=100)
    orch = MultiAgentOrchestrator([agent], field=field, monitor=monitor)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    fq = result["field_quality"]
    assert "pattern_count" in fq
    assert "level_distribution" in fq
    assert fq["pattern_count"] >= 0
```

Run: `uv run pytest tests/monitor/test_structural_law.py::test_monitor_none_returns_empty_field_quality -v`
Expected: FAIL (MultiAgentOrchestrator has no `monitor` parameter yet).

- [ ] **Step 3: Add monitor parameter to MultiAgentOrchestrator**

In `hpm/agents/multi_agent.py`, find the `__init__` signature and add `monitor=None`. Also add `self._t = 0` and `self.monitor = monitor` to `__init__` body:

```python
def __init__(self, agents, field: PatternField, seed_pattern=None, groups=None, monitor=None):
    ...  # existing body unchanged
    self._t = 0
    self.monitor = monitor
```

In `step()`, at the very start of the method body (before the agents loop), increment the step counter:

```python
self._t += 1
```

Then after the communication phase (just before `return metrics`), add:

```python
# Aggregate total_conflict across all agents
total_conflict_sum = sum(
    metrics[aid].get("total_conflict", 0.0) for aid in metrics
)
field_quality = (
    self.monitor.step(self._t, self.agents, total_conflict_sum)
    if self.monitor is not None
    else {}
)
```

And add to the return dict:
```python
return {**metrics, "field_quality": field_quality}
```

Note: if `step()` currently returns a flat dict keyed by agent_id, just add `"field_quality"` as a new key. Verify `"field_quality"` is not already a reserved key in the existing return dict.

- [ ] **Step 4: Export SQLiteStore from hpm/store/__init__.py**

In `hpm/store/__init__.py`, add:
```python
from .sqlite import SQLiteStore
```

- [ ] **Step 5: Run integration tests**

```bash
uv run pytest tests/monitor/ -v
```

Expected: all tests PASS including the two new integration tests.

- [ ] **Step 6: Run full suite**

```bash
uv run pytest -q
```

Expected: 277+ passed, 9 skipped (no regressions).

- [ ] **Step 7: Commit**

```bash
git add hpm/agents/multi_agent.py hpm/store/__init__.py tests/monitor/test_structural_law.py
git commit -m "feat: integrate StructuralLawMonitor into MultiAgentOrchestrator"
```

---

### Task 4: Smoke test and final commit

**Files:** No new files.

- [ ] **Step 1: Write a smoke test script**

```python
# Inline smoke test — run directly, not via pytest
import numpy as np
from hpm.store.sqlite import SQLiteStore
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.monitor.structural_law import StructuralLawMonitor

import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    db_path = os.path.join(tmp, "smoke.db")
    log_path = os.path.join(tmp, "monitor.jsonl")

    store = SQLiteStore(db_path)
    config = AgentConfig(feature_dim=8, agent_id="smoke_agent")
    agents = [Agent(config, store=store)]
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=10, log_path=log_path)
    orch = MultiAgentOrchestrator(agents, field=field, monitor=monitor)

    rng = np.random.default_rng(42)
    obs = {config.agent_id: None}
    for t in range(30):
        obs[config.agent_id] = rng.normal(0, 1, 8)
        result = orch.step(obs)

    assert "field_quality" in result
    fq = result["field_quality"]
    assert fq["pattern_count"] >= 0
    assert fq["diversity"] is not None  # step 30 is divisible by T_monitor=10

    import json
    with open(log_path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) >= 3  # steps 10, 20, and 30

    print("Smoke test PASSED")
    print(f"  pattern_count={fq['pattern_count']}, L4+={fq['level4plus_count']}")
    div_str = f"{fq['diversity']:.3f}" if fq['diversity'] is not None else "None"
    red_str = f"{fq['redundancy']:.3f}" if fq['redundancy'] is not None else "None"
    print(f"  diversity={div_str}, redundancy={red_str}")
    print(f"  JSON log entries: {len(lines)}")
```

Save as `smoke_monitor.py` in repo root, run:

```bash
uv run python smoke_monitor.py
```

Expected: prints `Smoke test PASSED` and a Field Quality Report console table at steps 10 and 20.

- [ ] **Step 2: Delete smoke script**

```bash
rm smoke_monitor.py
```

- [ ] **Step 3: Run full test suite one final time**

```bash
uv run pytest -q
```

Expected: 277+ passed, 9 skipped.

- [ ] **Step 4: Final commit**

```bash
git commit -m "feat: Phase 5 Structural Law Monitor complete — SQLiteStore extensions, StructuralLawMonitor, MultiAgentOrchestrator integration"
```
