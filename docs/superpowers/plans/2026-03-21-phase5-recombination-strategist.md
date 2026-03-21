# Phase 5: Recombination Strategist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `RecombinationStrategist` — a population-level governor that monitors `field_quality` from `StructuralLawMonitor` and mutates agent configs (conflict_threshold, kappa_0, beta_c) to prevent premature convergence.

**Architecture:** Standalone `RecombinationStrategist` class in `hpm/monitor/recombination_strategist.py`, composed into `MultiAgentOrchestrator` as an optional `strategist=None` kwarg (same pattern as `StructuralLawMonitor`). Three intervention modes: Recombination Burst (temporary conflict_threshold lowering), Adoption Scaling (kappa_0 EMA nudging), Conflict Damping (beta_c one-way ratchet).

**Tech Stack:** Python stdlib only (`math`). No new dependencies. Reads from `field_quality` dict produced by `StructuralLawMonitor`.

---

## File Structure

```
hpm/monitor/recombination_strategist.py         # new: RecombinationStrategist class
tests/monitor/test_recombination_strategist.py  # new: unit + integration tests
hpm/monitor/__init__.py                         # existing: add RecombinationStrategist export
hpm/agents/multi_agent.py                       # existing: add strategist=None kwarg + call
```

**Key existing files to read before starting:**
- `hpm/monitor/structural_law.py` — see how monitor is called by orchestrator (pattern to follow)
- `hpm/agents/multi_agent.py` — current step() return dict structure, where monitor.step() is called
- `hpm/config.py` — AgentConfig fields: `conflict_threshold`, `kappa_0`, `beta_c`

---

### Task 1: RecombinationStrategist class and unit tests

**Files:**
- Create: `hpm/monitor/recombination_strategist.py`
- Create: `tests/monitor/test_recombination_strategist.py`

**Context for implementer:**

The `RecombinationStrategist.step()` is called every orchestrator step with `field_quality` from the monitor. `field_quality["diversity"]` is `None` on non-heavy-metric steps (only computed every T_monitor steps). When `diversity` is `None`, stagnation counting, kappa_0 nudging, and beta_c damping are all skipped.

Agents have a `.agent_id` attribute and a `.config` attribute (AgentConfig dataclass). Config fields to mutate: `conflict_threshold` (float), `kappa_0` (float), `beta_c` (float, may not exist on all configs — use `hasattr`).

- [ ] **Step 1: Write all unit tests**

Create `tests/monitor/test_recombination_strategist.py`:

```python
import pytest
from unittest.mock import MagicMock
from hpm.monitor.recombination_strategist import RecombinationStrategist


def _make_agent(agent_id="a1", conflict_threshold=0.1, kappa_0=0.1, beta_c=1.0):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.config.conflict_threshold = conflict_threshold
    agent.config.kappa_0 = kappa_0
    agent.config.beta_c = beta_c
    return agent


def _heavy_fq(diversity=1.0, conflict=0.0):
    """field_quality dict with heavy metrics present."""
    return {
        "pattern_count": 5,
        "level_distribution": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        "level4plus_count": 2,
        "level4plus_mean_weight": 0.4,
        "conflict": conflict,
        "stability_mean": 0.7,
        "diversity": diversity,
        "redundancy": 0.1,
    }


def _light_fq(conflict=0.0):
    """field_quality dict with heavy metrics absent (non-T_monitor step)."""
    fq = _heavy_fq(conflict=conflict)
    fq["diversity"] = None
    fq["redundancy"] = None
    return fq


# ---- Burst mode ----

def test_no_intervention_when_healthy():
    """High diversity + low conflict → no config mutations."""
    agents = [_make_agent()]
    s = RecombinationStrategist(diversity_low=0.5, conflict_high=0.3, stagnation_window=3)
    result = s.step(1, _heavy_fq(diversity=1.5, conflict=0.1), agents)
    assert result["burst_active"] is False
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)


def test_burst_fires_after_stagnation_window():
    """N consecutive stagnant heavy-metric steps → burst fires."""
    agents = [_make_agent(conflict_threshold=0.1)]
    s = RecombinationStrategist(
        diversity_low=0.5, conflict_high=0.3, stagnation_window=3,
        burst_conflict_threshold=0.01, burst_duration=10
    )
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)
    s.step(2, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)  # not yet
    s.step(3, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.01)  # burst fired


def test_burst_active_flag():
    """burst_active is True while burst is running."""
    agents = [_make_agent()]
    s = RecombinationStrategist(stagnation_window=1, burst_conflict_threshold=0.01, burst_duration=5)
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    result = s.step(1, stagnant, agents)
    assert result["burst_active"] is True


def test_burst_restores_config_after_duration():
    """Original conflict_threshold restored after burst_duration steps."""
    agents = [_make_agent(conflict_threshold=0.15)]
    s = RecombinationStrategist(stagnation_window=1, burst_conflict_threshold=0.01, burst_duration=3)
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)  # burst fires
    assert agents[0].config.conflict_threshold == pytest.approx(0.01)
    s.step(2, _light_fq(), agents)  # step 1 of burst
    s.step(3, _light_fq(), agents)  # step 2 of burst
    s.step(4, _light_fq(), agents)  # step 3 of burst → restored
    assert agents[0].config.conflict_threshold == pytest.approx(0.15)


def test_burst_cooldown_prevents_retrigger():
    """Burst fires; stagnation resumes immediately; burst does NOT re-fire during cooldown."""
    agents = [_make_agent()]
    s = RecombinationStrategist(
        stagnation_window=1, burst_conflict_threshold=0.01,
        burst_duration=2, burst_cooldown=10
    )
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)   # burst fires
    s.step(2, _light_fq(), agents)
    s.step(3, _light_fq(), agents)  # burst ends, cooldown starts
    # Try to retrigger — stagnant heavy step during cooldown
    orig = agents[0].config.conflict_threshold
    s.step(4, stagnant, agents)
    s.step(5, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(orig)  # no burst


def test_stagnation_skipped_when_diversity_none():
    """If diversity is None (light step), stagnation counter does not advance."""
    agents = [_make_agent()]
    s = RecombinationStrategist(stagnation_window=2, burst_conflict_threshold=0.01)
    light = _light_fq(conflict=0.9)
    s.step(1, light, agents)
    s.step(2, light, agents)
    s.step(3, light, agents)
    assert s._stagnation_count == 0
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)


def test_burst_fires_on_stagnation_window_step():
    """Burst fires on the step when stagnation_count reaches window (not the step after).

    The implementation increments the stagnation counter AND fires the burst in the
    same step when count reaches the window threshold. With stagnation_window=2 and
    two consecutive stagnant heavy steps, the burst fires at the end of step 2.
    (The spec test name 'test_burst_fires_on_step_after_stagnation_window_not_during'
    is misleading — 'after' refers to the logic ordering within a single step call,
    not a separate step_t+1 call.)
    """
    agents = [_make_agent(conflict_threshold=0.1)]
    s = RecombinationStrategist(stagnation_window=2, burst_conflict_threshold=0.01, burst_duration=5)
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)  # count=1, not yet
    s.step(2, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.01)  # count=2 → fired


# ---- kappa_0 adoption scaling ----

def test_kappa_0_no_nudge_on_first_heavy_step():
    """First step with non-None diversity sets EMA but does not nudge kappa_0."""
    agents = [_make_agent(kappa_0=0.1)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.2, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=2.0), agents)
    assert agents[0].config.kappa_0 == pytest.approx(0.1)
    assert s._diversity_ema == pytest.approx(2.0)


def test_kappa_0_rises_when_diversity_improving():
    """diversity above EMA → kappa_0 nudged toward kappa_0_max."""
    agents = [_make_agent(kappa_0=0.1)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.5, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=1.0), agents)  # initialise EMA=1.0, no nudge
    s.step(2, _heavy_fq(diversity=2.0), agents)  # new EMA=1.5 < diversity=2.0 → up
    assert agents[0].config.kappa_0 > 0.1


def test_kappa_0_falls_when_diversity_falling():
    """diversity below EMA → kappa_0 nudged toward kappa_0_min."""
    agents = [_make_agent(kappa_0=0.2)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.5, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=2.0), agents)  # EMA=2.0, no nudge
    s.step(2, _heavy_fq(diversity=0.5), agents)  # new EMA=1.25 > diversity=0.5 → down
    assert agents[0].config.kappa_0 < 0.2


def test_kappa_0_clamped_to_bounds():
    """kappa_0 never exceeds kappa_0_max or falls below kappa_0_min."""
    # Upper bound: nudge toward max repeatedly
    agents = [_make_agent(kappa_0=0.28)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.9, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=0.1), agents)  # initialise EMA=0.1, no nudge
    for _ in range(10):
        s.step(2, _heavy_fq(diversity=100.0), agents)  # diversity >> EMA → nudge up
    assert agents[0].config.kappa_0 <= 0.3

    # Lower bound: fresh strategist so EMA is reset, then nudge toward min repeatedly
    agents2 = [_make_agent(kappa_0=0.06)]
    s2 = RecombinationStrategist(kappa_0_ema_alpha=0.9, kappa_0_min=0.05, kappa_0_max=0.3)
    s2.step(1, _heavy_fq(diversity=100.0), agents2)  # initialise EMA=100.0, no nudge
    for _ in range(10):
        s2.step(2, _heavy_fq(diversity=0.0), agents2)  # diversity << EMA → nudge down
    assert agents2[0].config.kappa_0 >= 0.05


# ---- beta_c damping ----

def test_beta_c_damped_when_conflict_persists():
    """conflict > conflict_high for stagnation_window cycles → beta_c reduced."""
    agents = [_make_agent(beta_c=1.0)]
    s = RecombinationStrategist(conflict_high=0.3, stagnation_window=3, beta_c_decay=0.9, beta_c_min=0.1)
    high_conflict = _heavy_fq(diversity=1.0, conflict=0.5)
    s.step(1, high_conflict, agents)
    s.step(2, high_conflict, agents)
    assert agents[0].config.beta_c == pytest.approx(1.0)  # not yet
    s.step(3, high_conflict, agents)
    assert agents[0].config.beta_c == pytest.approx(0.9)


def test_beta_c_floored_at_minimum():
    """Repeated damping → beta_c never falls below beta_c_min."""
    agents = [_make_agent(beta_c=1.0)]
    s = RecombinationStrategist(conflict_high=0.3, stagnation_window=1, beta_c_decay=0.1, beta_c_min=0.1)
    high_conflict = _heavy_fq(diversity=1.0, conflict=0.5)
    for i in range(20):
        s.step(i, high_conflict, agents)
    assert agents[0].config.beta_c >= 0.1


def test_agent_missing_beta_c_skipped():
    """Agent config without beta_c attribute is skipped silently during damping."""
    agent = MagicMock()
    agent.agent_id = "a1"
    agent.config.conflict_threshold = 0.1
    agent.config.kappa_0 = 0.1
    del agent.config.beta_c  # no beta_c
    # hasattr returns False for MagicMock deleted attributes — use spec instead
    agent2 = MagicMock(spec=["agent_id", "config"])
    agent2.agent_id = "a2"
    agent2.config = MagicMock(spec=["conflict_threshold", "kappa_0"])
    agent2.config.conflict_threshold = 0.1
    agent2.config.kappa_0 = 0.1
    s = RecombinationStrategist(conflict_high=0.3, stagnation_window=1, beta_c_min=0.1)
    high = _heavy_fq(diversity=1.0, conflict=0.5)
    # Should not raise
    s.step(1, high, [agent2])


# ---- Interventions dict ----

def test_interventions_dict_always_present():
    """Returned dict always has all expected keys."""
    agents = [_make_agent()]
    s = RecombinationStrategist()
    result = s.step(1, _heavy_fq(), agents)
    assert "burst_active" in result
    assert "kappa_0" in result
    assert "beta_c_scaled" in result
    assert "stagnation_count" in result
    assert "cooldown_remaining" in result


def test_empty_agents_list():
    """step() with agents=[] returns valid dict, no errors."""
    s = RecombinationStrategist()
    result = s.step(1, _heavy_fq(diversity=0.1, conflict=0.9), [])
    assert isinstance(result, dict)
    assert "burst_active" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/monitor/test_recombination_strategist.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hpm.monitor.recombination_strategist'`

- [ ] **Step 3: Implement RecombinationStrategist**

Create `hpm/monitor/recombination_strategist.py`:

```python
"""
hpm/monitor/recombination_strategist.py — Recombination Strategist ("The Innovator")

Population-level governor that mutates per-agent config parameters to prevent
premature convergence and stimulate structural novelty.

Three intervention modes:
  - Recombination Burst: temporarily lowers conflict_threshold on all agents
  - Adoption Scaling: adjusts kappa_0 based on diversity trend (EMA)
  - Conflict Scale Damping: reduces beta_c when conflict persists (one-way ratchet)
"""


class RecombinationStrategist:
    """
    Reads field_quality from StructuralLawMonitor and mutates agent configs.

    Composed into MultiAgentOrchestrator as an optional strategist=None kwarg.
    Degrades gracefully: if field_quality["diversity"] is None, all interventions
    that require heavy metrics are skipped.
    """

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
        self.diversity_low = diversity_low
        self.conflict_high = conflict_high
        self.stagnation_window = stagnation_window
        self.burst_conflict_threshold = burst_conflict_threshold
        self.burst_duration = burst_duration
        self.burst_cooldown = burst_cooldown
        self.kappa_0_min = kappa_0_min
        self.kappa_0_max = kappa_0_max
        self.kappa_0_ema_alpha = kappa_0_ema_alpha
        self.beta_c_min = beta_c_min
        self.beta_c_decay = beta_c_decay

        # Internal state
        self._stagnation_count: int = 0
        self._burst_steps_remaining: int = 0
        self._cooldown_steps_remaining: int = 0
        self._original_conflict_thresholds: dict = {}
        self._diversity_ema: float | None = None
        self._conflict_persistent_cycles: int = 0

    def step(self, step_t: int, field_quality: dict, agents: list) -> dict:
        """
        Called by MultiAgentOrchestrator after monitor.step().

        Args:
            step_t:        Current orchestrator step counter.
            field_quality: Dict from StructuralLawMonitor.step() (or {}).
            agents:        List of Agent instances (mutable config).

        Returns:
            interventions dict with keys:
              burst_active, kappa_0, beta_c_scaled, stagnation_count, cooldown_remaining
        """
        # Step 1: Tick down burst / cooldown (before stagnation check)
        if self._burst_steps_remaining > 0:
            self._burst_steps_remaining -= 1
            if self._burst_steps_remaining == 0:
                self._restore_conflict_thresholds(agents)
                self._cooldown_steps_remaining = self.burst_cooldown
        elif self._cooldown_steps_remaining > 0:
            self._cooldown_steps_remaining -= 1

        diversity = field_quality.get("diversity")  # None on light steps
        conflict = float(field_quality.get("conflict", 0.0))

        kappa_0_applied = None
        beta_c_scaled = False

        if diversity is not None:
            # --- Stagnation check + burst fire (only outside burst/cooldown) ---
            if self._burst_steps_remaining == 0 and self._cooldown_steps_remaining == 0:
                if diversity < self.diversity_low and conflict > self.conflict_high:
                    self._stagnation_count += 1
                else:
                    self._stagnation_count = 0

                if self._stagnation_count >= self.stagnation_window:
                    self._fire_burst(agents)

            # --- Adoption Scaling (kappa_0) ---
            kappa_0_applied = self._update_kappa_0(diversity, agents)

            # --- Conflict Scale Damping (beta_c) ---
            beta_c_scaled = self._update_beta_c(conflict, agents)

        return {
            "burst_active": self._burst_steps_remaining > 0,
            "kappa_0": kappa_0_applied,
            "beta_c_scaled": beta_c_scaled,
            "stagnation_count": self._stagnation_count,
            "cooldown_remaining": self._cooldown_steps_remaining,
        }

    # ------------------------------------------------------------------
    # Burst
    # ------------------------------------------------------------------

    def _fire_burst(self, agents: list) -> None:
        for agent in agents:
            self._original_conflict_thresholds[agent.agent_id] = agent.config.conflict_threshold
            agent.config.conflict_threshold = self.burst_conflict_threshold
        self._burst_steps_remaining = self.burst_duration
        self._stagnation_count = 0

    def _restore_conflict_thresholds(self, agents: list) -> None:
        for agent in agents:
            if agent.agent_id in self._original_conflict_thresholds:
                agent.config.conflict_threshold = self._original_conflict_thresholds[agent.agent_id]
        self._original_conflict_thresholds = {}

    # ------------------------------------------------------------------
    # Adoption Scaling
    # ------------------------------------------------------------------

    def _update_kappa_0(self, diversity: float, agents: list) -> float | None:
        if self._diversity_ema is None:
            self._diversity_ema = diversity
            return None  # No nudge on first heavy step

        self._diversity_ema = (
            self.kappa_0_ema_alpha * diversity
            + (1 - self.kappa_0_ema_alpha) * self._diversity_ema
        )

        if diversity == self._diversity_ema:
            return None

        kappa_0_result = None
        for agent in agents:
            current = agent.config.kappa_0
            if diversity > self._diversity_ema:
                new_k0 = current + self.kappa_0_ema_alpha * (self.kappa_0_max - current)
            else:
                new_k0 = current - self.kappa_0_ema_alpha * (current - self.kappa_0_min)
            agent.config.kappa_0 = max(self.kappa_0_min, min(self.kappa_0_max, new_k0))
            kappa_0_result = agent.config.kappa_0

        return kappa_0_result

    # ------------------------------------------------------------------
    # Conflict Scale Damping
    # ------------------------------------------------------------------

    def _update_beta_c(self, conflict: float, agents: list) -> bool:
        if conflict > self.conflict_high:
            self._conflict_persistent_cycles += 1
        else:
            self._conflict_persistent_cycles = 0

        if self._conflict_persistent_cycles >= self.stagnation_window:
            for agent in agents:
                if hasattr(agent.config, "beta_c"):
                    agent.config.beta_c = max(self.beta_c_min, agent.config.beta_c * self.beta_c_decay)
            return True

        return False
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/monitor/test_recombination_strategist.py -v
```

Expected: all tests PASS. If `test_agent_missing_beta_c_skipped` is flaky due to MagicMock spec behaviour, simplify: create a plain object with only the needed attributes instead of MagicMock.

- [ ] **Step 5: Commit**

```bash
git add hpm/monitor/recombination_strategist.py tests/monitor/test_recombination_strategist.py
git commit -m "feat: RecombinationStrategist with burst, kappa_0 scaling, and beta_c damping"
```

---

### Task 2: MultiAgentOrchestrator integration + __init__ export

**Files:**
- Modify: `hpm/agents/multi_agent.py`
- Modify: `hpm/monitor/__init__.py`
- Modify: `tests/monitor/test_recombination_strategist.py` (add integration tests)

- [ ] **Step 1: Read hpm/agents/multi_agent.py**

Find:
- The `__init__` signature (current params: `agents, field, seed_pattern=None, groups=None, monitor=None`)
- Where `monitor.step()` is called and `field_quality` is computed
- The final return statement

- [ ] **Step 2: Write failing integration tests**

Append to `tests/monitor/test_recombination_strategist.py`:

```python
import numpy as np
from hpm.store.sqlite import SQLiteStore
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.monitor.structural_law import StructuralLawMonitor
from hpm.monitor.recombination_strategist import RecombinationStrategist


def _make_real_agent(store, dim=4):
    config = AgentConfig(feature_dim=dim, agent_id=f"agent_{id(store)}")
    return Agent(config, store=store)


def test_orchestrator_with_no_strategist(tmp_path):
    """Orchestrator with strategist=None returns interventions == {}."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field, monitor=None, strategist=None)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("interventions") == {}


def test_orchestrator_strategist_integrated(tmp_path):
    """Orchestrator with monitor + strategist returns both field_quality and interventions."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=100)
    strategist = RecombinationStrategist()
    orch = MultiAgentOrchestrator([agent], field=field, monitor=monitor, strategist=strategist)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert "field_quality" in result
    assert "interventions" in result
    fq = result["interventions"]
    assert "burst_active" in fq
```

Run: `uv run pytest tests/monitor/test_recombination_strategist.py::test_orchestrator_with_no_strategist -v`
Expected: FAIL (`MultiAgentOrchestrator` has no `strategist` parameter yet)

- [ ] **Step 3: Add strategist to MultiAgentOrchestrator**

In `hpm/agents/multi_agent.py`:

Find the `__init__` signature and add `strategist=None`:
```python
def __init__(self, agents, field: PatternField, seed_pattern=None, groups=None, monitor=None, strategist=None):
```

Add to `__init__` body:
```python
self.strategist = strategist
```

In `step()`, after the `field_quality` computation (after `monitor.step()` call), add:

```python
interventions = (
    self.strategist.step(self._t, field_quality, self.agents)
    if self.strategist is not None
    else {}
)
```

Update the return statement at the end of `step()`. Replace:
```python
        return {**metrics, "field_quality": field_quality}
```
With:
```python
        return {**metrics, "field_quality": field_quality, "interventions": interventions}
```

- [ ] **Step 4: Export RecombinationStrategist from hpm/monitor/__init__.py**

In `hpm/monitor/__init__.py`, add:
```python
from .recombination_strategist import RecombinationStrategist

__all__ = ["StructuralLawMonitor", "RecombinationStrategist"]
```

- [ ] **Step 5: Run all monitor tests**

```bash
uv run pytest tests/monitor/ -v
```

Expected: all tests PASS including the two new integration tests.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -q
```

Expected: 300+ passed, 9 skipped (no regressions).

- [ ] **Step 7: Commit**

```bash
git add hpm/agents/multi_agent.py hpm/monitor/__init__.py tests/monitor/test_recombination_strategist.py
git commit -m "feat: integrate RecombinationStrategist into MultiAgentOrchestrator"
```

---

### Task 3: Smoke test and final commit

**Files:** No new files.

- [ ] **Step 1: Write and run smoke test**

```python
# Save as smoke_strategist.py, delete after
import numpy as np
from hpm.store.sqlite import SQLiteStore
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.monitor.structural_law import StructuralLawMonitor
from hpm.monitor.recombination_strategist import RecombinationStrategist
import tempfile, os

with tempfile.TemporaryDirectory() as tmp:
    store = SQLiteStore(os.path.join(tmp, "smoke.db"))
    config = AgentConfig(feature_dim=8, agent_id="smoke_agent")
    agents = [Agent(config, store=store)]
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=10)
    strategist = RecombinationStrategist(stagnation_window=2, burst_duration=5)
    orch = MultiAgentOrchestrator(agents, field=field, monitor=monitor, strategist=strategist)

    rng = np.random.default_rng(42)
    for t in range(30):
        obs = {config.agent_id: rng.normal(0, 1, 8)}
        result = orch.step(obs)

    assert "field_quality" in result
    assert "interventions" in result
    iv = result["interventions"]
    assert "burst_active" in iv
    assert "stagnation_count" in iv
    print("Smoke test PASSED")
    print(f"  burst_active={iv['burst_active']}, stagnation_count={iv['stagnation_count']}")
    print(f"  kappa_0={iv['kappa_0']}, beta_c_scaled={iv['beta_c_scaled']}")
```

Run: `uv run python smoke_strategist.py`
Expected: prints `Smoke test PASSED`

- [ ] **Step 2: Delete smoke script**

```bash
rm smoke_strategist.py
```

- [ ] **Step 3: Run full test suite one final time**

```bash
uv run pytest -q
```

Expected: 300+ passed, 9 skipped.

- [ ] **Step 4: Final commit**

All implementation files were committed in Tasks 1 and 2. The smoke script was deleted and never committed. There is nothing to add or commit here — this step is a no-op confirmation checkpoint.

```bash
git log --oneline -3  # confirm the two feature commits are present
```

Expected output includes:
```
feat: integrate RecombinationStrategist into MultiAgentOrchestrator
feat: RecombinationStrategist with burst, kappa_0 scaling, and beta_c damping
```
