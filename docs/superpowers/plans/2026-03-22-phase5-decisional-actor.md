# Phase 5: Decisional RL Actor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `DecisionalActor` ("The Actor") — the final Phase 5 agent that sits at the end of `MultiAgentOrchestrator.step()` and makes two kinds of learned decisions: external action selection (via top-pattern log-prob prior + Q-values) and internal meta-action dispatch (EXPLOIT/EXPLORE/REGROUND, learned from Δfield_quality reward).

**Architecture:** Two Q-value heads (`ExternalHead`, `InternalHead`) live inside `DecisionalActor` in `hpm/agents/actor.py`. ExternalHead selects from discrete action candidates using the top pattern's log-prob as a prior, updated by environment reward each step. InternalHead fires on high redundancy or fragility, executes a meta-action mutating orchestrator component state, and updates Q-values from Δfield_quality on the next trigger. Both use TD(0) bandit updates with no neural networks.

**Tech Stack:** Python stdlib, NumPy, pytest, GaussianPattern (for log_prob scoring in ExternalHead), SQLiteStore (integration tests only)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `hpm/agents/actor.py` | Create | `_softmax`, `_field_score`, `ExternalHead`, `InternalHead`, `DecisionalActor` |
| `tests/agents/test_actor.py` | Create | 14 tests (12 unit + 2 integration) |
| `hpm/agents/multi_agent.py` | Modify | Add `actor=None` kwarg; call after `forecast_report`; add `actor_report` to return |
| `hpm/agents/__init__.py` | Modify | Export `DecisionalActor` |

---

## Task 1: DecisionalActor class + unit tests

**Files:**
- Create: `tests/agents/test_actor.py`
- Create: `hpm/agents/actor.py`

---

- [ ] **Step 1.1 — Write the failing test file**

Create `tests/agents/test_actor.py`:

```python
import numpy as np
import pytest
from hpm.agents.actor import DecisionalActor
from hpm.patterns.gaussian import GaussianPattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_forecast(prediction=None, fragility_flag=False, top_pattern_id=None):
    return {
        "prediction": prediction,
        "fragility_flag": fragility_flag,
        "top_pattern_id": top_pattern_id,
        "top_pattern_level": None,
        "prediction_error": None,
        "fragility_score": None,
        "delta_nll": None,
    }


def _make_field_quality(redundancy=0.0, level4plus_count=0):
    return {"redundancy": redundancy, "level4plus_count": level4plus_count}


class StubStore:
    """Minimal store stub: returns a fixed list of (pattern, weight, agent_id)."""
    def __init__(self, records):
        self._records = records  # list of (pattern, weight, agent_id)

    def query_all(self):
        return self._records


class StubForecaster:
    """Stub with the two attributes DecisionalActor needs."""
    def __init__(self, patterns=None, min_bridge_level=4):
        self.min_bridge_level = min_bridge_level
        self._store = StubStore(patterns or [])


class StubBridge:
    """Stub with the two attributes DecisionalActor needs."""
    def __init__(self, T_substrate=20, t=0):
        self.T_substrate = T_substrate
        self._t = t


# ---------------------------------------------------------------------------
# Test 1: External action selected by log-prob
# ---------------------------------------------------------------------------

def test_external_action_selected_by_logprob():
    """Pattern centred on action_vectors[2]; temperature→0 → selects action 2."""
    action_vectors = np.eye(4)
    mu = action_vectors[2]  # [0, 0, 1, 0]
    p = GaussianPattern(mu=mu, sigma=np.eye(4))
    p.level = 4

    forecaster = StubForecaster(patterns=[(p, 1.0, "a1")])
    actor = DecisionalActor(
        action_vectors=action_vectors,
        forecaster=forecaster,
        temperature=1e-9,
    )

    result = actor.step(
        1,
        _make_field_quality(),
        _make_forecast(prediction=mu, top_pattern_id=p.id),
    )
    assert result["external_action"] == 2


# ---------------------------------------------------------------------------
# Test 2: Q-values update on reward
# ---------------------------------------------------------------------------

def test_q_values_update_on_reward():
    """After reward=1.0 with alpha_ext=1.0, Q[selected_action] == 1.0."""
    action_vectors = np.eye(3)
    actor = DecisionalActor(action_vectors=action_vectors, alpha_ext=1.0)

    np.random.seed(42)
    result = actor.step(
        1,
        _make_field_quality(),
        _make_forecast(prediction=action_vectors[0]),
        external_reward=1.0,
    )
    selected = result["external_action"]
    # Q[selected] = 0 + 1.0*(1.0 - 0) = 1.0
    assert result["external_q_values"][selected] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 3: No action when prediction is None
# ---------------------------------------------------------------------------

def test_no_action_when_no_prediction():
    """prediction=None → external_action=None."""
    actor = DecisionalActor(action_vectors=np.eye(3))
    result = actor.step(1, _make_field_quality(), _make_forecast(prediction=None))
    assert result["external_action"] is None


# ---------------------------------------------------------------------------
# Test 4: Internal action triggered on redundancy
# ---------------------------------------------------------------------------

def test_internal_action_triggered_on_redundancy():
    """redundancy > threshold → internal_action is not None."""
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        redundancy_threshold=0.3,
        temperature=1e-9,
    )
    result = actor.step(1, _make_field_quality(redundancy=0.5), _make_forecast())
    assert result["internal_action"] is not None


# ---------------------------------------------------------------------------
# Test 5: Internal action triggered on fragility_flag
# ---------------------------------------------------------------------------

def test_internal_action_triggered_on_fragility_flag():
    """fragility_flag=True with low redundancy → internal_action is not None."""
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        redundancy_threshold=0.3,
        temperature=1e-9,
    )
    result = actor.step(
        1,
        _make_field_quality(redundancy=0.0),
        _make_forecast(fragility_flag=True),
    )
    assert result["internal_action"] is not None


# ---------------------------------------------------------------------------
# Test 6: Internal action not triggered below threshold
# ---------------------------------------------------------------------------

def test_internal_action_not_triggered_below_threshold():
    """redundancy=0, fragility_flag=False → internal_action=None."""
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        redundancy_threshold=0.3,
    )
    result = actor.step(
        1,
        _make_field_quality(redundancy=0.0),
        _make_forecast(fragility_flag=False),
    )
    assert result["internal_action"] is None


# ---------------------------------------------------------------------------
# Test 7: Internal Q-values update on delayed reward
# ---------------------------------------------------------------------------

def test_internal_q_update_on_delayed_reward():
    """
    Trigger at t0 (field_score=1.0) with EXPLOIT forced.
    Trigger at t1 (field_score=2.0).
    reward = 2.0 - 1.0 = 1.0; alpha_int=1.0 → Q[EXPLOIT] = 1.0.
    """
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        alpha_int=1.0,
        redundancy_threshold=0.0,   # always triggers
        temperature=1e-9,
    )
    # Force EXPLOIT (action 0) at first trigger
    actor._internal_head.q_values = np.array([10.0, 0.0, 0.0])

    fq0 = _make_field_quality(level4plus_count=1, redundancy=0.0)
    result0 = actor.step(1, fq0, _make_forecast(fragility_flag=True))
    assert result0["internal_action"] == "EXPLOIT"

    # Second trigger: score=2.0; delayed reward = 2.0 - 1.0 = 1.0
    # Q[0] = 10.0 + 1.0*(1.0 - 10.0) = 10.0 - 9.0 = 1.0
    fq1 = _make_field_quality(level4plus_count=2, redundancy=0.0)
    result1 = actor.step(2, fq1, _make_forecast(fragility_flag=True))
    assert result1["internal_q_values"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 8: EXPLOIT raises min_bridge_level
# ---------------------------------------------------------------------------

def test_exploit_raises_min_bridge_level():
    """EXPLOIT action increments forecaster.min_bridge_level by 1."""
    forecaster = StubForecaster(min_bridge_level=4)
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        forecaster=forecaster,
        redundancy_threshold=0.0,
        temperature=1e-9,
        min_bridge_level_bounds=(2, 6),
    )
    actor._internal_head.q_values = np.array([10.0, 0.0, 0.0])  # force EXPLOIT

    actor.step(1, _make_field_quality(redundancy=1.0), _make_forecast())
    assert forecaster.min_bridge_level == 5


# ---------------------------------------------------------------------------
# Test 9: EXPLORE lowers min_bridge_level
# ---------------------------------------------------------------------------

def test_explore_lowers_min_bridge_level():
    """EXPLORE action decrements forecaster.min_bridge_level by 1."""
    forecaster = StubForecaster(min_bridge_level=4)
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        forecaster=forecaster,
        redundancy_threshold=0.0,
        temperature=1e-9,
        min_bridge_level_bounds=(2, 6),
    )
    actor._internal_head.q_values = np.array([0.0, 10.0, 0.0])  # force EXPLORE

    actor.step(1, _make_field_quality(redundancy=1.0), _make_forecast())
    assert forecaster.min_bridge_level == 3


# ---------------------------------------------------------------------------
# Test 10: EXPLORE respects lower bound
# ---------------------------------------------------------------------------

def test_explore_respects_lower_bound():
    """EXPLORE at lower bound → min_bridge_level unchanged."""
    forecaster = StubForecaster(min_bridge_level=2)
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        forecaster=forecaster,
        redundancy_threshold=0.0,
        temperature=1e-9,
        min_bridge_level_bounds=(2, 6),
    )
    actor._internal_head.q_values = np.array([0.0, 10.0, 0.0])  # force EXPLORE

    actor.step(1, _make_field_quality(redundancy=1.0), _make_forecast())
    assert forecaster.min_bridge_level == 2  # unchanged


# ---------------------------------------------------------------------------
# Test 11: REGROUND resets bridge._t
# ---------------------------------------------------------------------------

def test_reground_resets_bridge_t():
    """REGROUND sets bridge._t = bridge.T_substrate - 1."""
    bridge = StubBridge(T_substrate=20, t=5)
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        bridge=bridge,
        redundancy_threshold=0.0,
        temperature=1e-9,
    )
    actor._internal_head.q_values = np.array([0.0, 0.0, 10.0])  # force REGROUND

    actor.step(1, _make_field_quality(redundancy=1.0), _make_forecast())
    assert bridge._t == 19  # T_substrate - 1


# ---------------------------------------------------------------------------
# Test 12: All 4 report keys always present
# ---------------------------------------------------------------------------

def test_actor_report_keys_always_present():
    """All 4 keys present whether gate fires or not."""
    actor = DecisionalActor(action_vectors=np.eye(3))
    expected = {"external_action", "internal_action", "external_q_values", "internal_q_values"}

    # No prediction, no trigger
    r1 = actor.step(1, _make_field_quality(), _make_forecast())
    assert set(r1.keys()) == expected

    # Prediction present, trigger fires
    r2 = actor.step(
        2,
        _make_field_quality(redundancy=1.0),
        _make_forecast(prediction=np.zeros(3), fragility_flag=True),
    )
    assert set(r2.keys()) == expected
```

- [ ] **Step 1.2 — Run tests to verify failure**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && pytest tests/agents/test_actor.py -v 2>&1 | head -20
```

Expected: `ImportError` — `hpm/agents/actor.py` does not exist yet.

- [ ] **Step 1.3 — Implement `hpm/agents/actor.py`**

Create `hpm/agents/actor.py`:

```python
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _field_score(field_quality: dict) -> float:
    """Scalar summarising field health: high when many deep patterns, low redundancy."""
    count = field_quality.get("level4plus_count", 0) or 0
    redundancy = field_quality.get("redundancy") or 0.0
    return float(count) * (1.0 - float(redundancy))


_INTERNAL_ACTION_NAMES = ["EXPLOIT", "EXPLORE", "REGROUND"]


# ---------------------------------------------------------------------------
# ExternalHead
# ---------------------------------------------------------------------------

class ExternalHead:
    """
    Selects a discrete external action every step.

    Uses the top pattern's log_prob scores as a prior over actions,
    combined with learned Q-values. Updated by external reward via TD(0).
    """

    def __init__(
        self,
        n_actions: int,
        alpha_ext: float,
        beta: float,
        temperature: float,
    ):
        self.q_values = np.zeros(n_actions)
        self.alpha_ext = alpha_ext
        self.beta = beta
        self.temperature = temperature
        self._last_action: int | None = None

    def select(
        self,
        action_vectors: np.ndarray,
        prediction,
        forecaster,
        top_pattern_id: str | None,
    ) -> int | None:
        """Return action index, or None if prediction is None."""
        if prediction is None:
            return None

        n_actions = len(action_vectors)

        # Look up top pattern from forecaster's store
        top_pattern = None
        if forecaster is not None and top_pattern_id is not None:
            for p, _w, _aid in forecaster._store.query_all():
                if p.id == top_pattern_id:
                    top_pattern = p
                    break

        if top_pattern is not None:
            log_prob_scores = np.array([
                float(top_pattern.log_prob(action_vectors[i]))
                for i in range(n_actions)
            ])
            logits = self.beta * log_prob_scores + self.q_values
        else:
            # Fallback: Q-values only (no pattern available)
            logits = self.q_values.copy()

        probs = _softmax(logits / self.temperature)
        action = int(np.random.choice(n_actions, p=probs))
        self._last_action = action
        return action

    def update(self, external_reward: float) -> None:
        """TD(0) Q-update for the last selected action."""
        if self._last_action is not None:
            q = self.q_values[self._last_action]
            self.q_values[self._last_action] = q + self.alpha_ext * (external_reward - q)


# ---------------------------------------------------------------------------
# InternalHead
# ---------------------------------------------------------------------------

class InternalHead:
    """
    Selects a meta-action (EXPLOIT / EXPLORE / REGROUND) when triggered.

    Trigger: high redundancy OR fragility_flag.
    Reward: Δfield_score at the *next* trigger (delayed reward).
    """

    def __init__(
        self,
        alpha_int: float,
        temperature: float,
        redundancy_threshold: float,
        min_bridge_level_bounds: tuple,
    ):
        self.q_values = np.zeros(3)
        self.alpha_int = alpha_int
        self.temperature = temperature
        self.redundancy_threshold = redundancy_threshold
        self.min_bridge_level_bounds = min_bridge_level_bounds
        self._last_action: int | None = None
        self._baseline_score: float | None = None

    def _triggered(self, field_quality: dict, forecast_report: dict) -> bool:
        redundancy = field_quality.get("redundancy", 0.0) or 0.0
        fragility_flag = forecast_report.get("fragility_flag", False)
        return redundancy > self.redundancy_threshold or bool(fragility_flag)

    def step(
        self,
        field_quality: dict,
        forecast_report: dict,
        forecaster,
        bridge,
    ) -> str | None:
        """
        Check trigger, apply delayed reward, select and execute meta-action.
        Returns action name or None if not triggered.
        """
        if not self._triggered(field_quality, forecast_report):
            return None

        current_score = _field_score(field_quality)

        # Apply delayed reward from the previous trigger
        if self._last_action is not None and self._baseline_score is not None:
            reward = current_score - self._baseline_score
            q = self.q_values[self._last_action]
            self.q_values[self._last_action] = q + self.alpha_int * (reward - q)

        # Select new action
        probs = _softmax(self.q_values / self.temperature)
        action = int(np.random.choice(3, p=probs))
        self._last_action = action
        self._baseline_score = current_score

        # Execute
        self._execute(action, forecaster, bridge)
        return _INTERNAL_ACTION_NAMES[action]

    def _execute(self, action: int, forecaster, bridge) -> None:
        lo, hi = self.min_bridge_level_bounds
        if action == 0:  # EXPLOIT
            if forecaster is not None:
                forecaster.min_bridge_level = min(forecaster.min_bridge_level + 1, hi)
        elif action == 1:  # EXPLORE
            if forecaster is not None:
                forecaster.min_bridge_level = max(forecaster.min_bridge_level - 1, lo)
        elif action == 2:  # REGROUND
            if bridge is not None:
                bridge._t = bridge.T_substrate - 1


# ---------------------------------------------------------------------------
# DecisionalActor
# ---------------------------------------------------------------------------

class DecisionalActor:
    """
    HPM Phase 5 — Decisional RL Actor ("The Actor").

    Sits at the end of MultiAgentOrchestrator.step(). Makes two learned
    decisions each step:

    - ExternalHead: discrete action selection using top-pattern log-prob prior
      + learned Q-values; updated by environment reward each step.
    - InternalHead: meta-action dispatch (EXPLOIT/EXPLORE/REGROUND) triggered
      by field_quality; updated by Δfield_quality reward on next trigger.

    Composed into MultiAgentOrchestrator as optional actor=None kwarg.
    """

    def __init__(
        self,
        action_vectors: np.ndarray,       # shape (n_actions, feature_dim)
        forecaster=None,                   # ref to PredictiveSynthesisAgent
        bridge=None,                       # ref to SubstrateBridgeAgent
        alpha_ext: float = 0.1,
        alpha_int: float = 0.1,
        beta: float = 1.0,
        temperature: float = 1.0,
        redundancy_threshold: float = 0.3,
        min_bridge_level_bounds: tuple = (2, 6),
    ):
        self._action_vectors = action_vectors
        self._forecaster = forecaster
        self._bridge = bridge
        self._external_head = ExternalHead(
            n_actions=len(action_vectors),
            alpha_ext=alpha_ext,
            beta=beta,
            temperature=temperature,
        )
        self._internal_head = InternalHead(
            alpha_int=alpha_int,
            temperature=temperature,
            redundancy_threshold=redundancy_threshold,
            min_bridge_level_bounds=min_bridge_level_bounds,
        )

    def step(
        self,
        step_t: int,
        field_quality: dict,
        forecast_report: dict,
        external_reward: float = 0.0,
    ) -> dict:
        """
        Called by MultiAgentOrchestrator every step.

        Step flow:
        1. ExternalHead: select action using top-pattern prior + Q; update Q on reward
        2. InternalHead: check trigger; apply delayed reward; select + execute meta-action
        3. Return actor_report with all 4 keys

        Args:
            step_t:          Current orchestrator step counter.
            field_quality:   Dict from StructuralLawMonitor (or {}).
            forecast_report: Dict from PredictiveSynthesisAgent (or {}).
            external_reward: Reward from environment this step (default 0.0).

        Returns:
            actor_report dict — all 4 keys always present.
        """
        prediction = forecast_report.get("prediction")
        top_pattern_id = forecast_report.get("top_pattern_id")

        # 1. ExternalHead
        external_action = self._external_head.select(
            self._action_vectors, prediction, self._forecaster, top_pattern_id
        )
        self._external_head.update(external_reward)

        # 2. InternalHead
        internal_action = self._internal_head.step(
            field_quality, forecast_report, self._forecaster, self._bridge
        )

        # 3. Return
        return {
            "external_action": external_action,
            "internal_action": internal_action,
            "external_q_values": self._external_head.q_values.tolist(),
            "internal_q_values": self._internal_head.q_values.tolist(),
        }
```

- [ ] **Step 1.4 — Run tests to verify pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && pytest tests/agents/test_actor.py -v -k "not orchestrator"
```

Expected: 12 tests pass (the 2 orchestrator integration tests aren't written yet).

- [ ] **Step 1.5 — Commit**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm/agents/actor.py tests/agents/test_actor.py
git commit -m "$(cat <<'EOF'
feat: add DecisionalActor with ExternalHead and InternalHead (Phase 5)

Implements The Actor in hpm/agents/actor.py:
- ExternalHead: Q-bandit with top-pattern log-prob prior, TD(0) update
- InternalHead: EXPLOIT/EXPLORE/REGROUND meta-actions, delayed reward
- 12 unit tests covering all paths

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: MultiAgentOrchestrator integration + __init__ export

**Files:**
- Append to: `tests/agents/test_actor.py` (2 integration tests)
- Modify: `hpm/agents/multi_agent.py`
- Modify: `hpm/agents/__init__.py`

---

- [ ] **Step 2.1 — Write failing integration tests**

Read `tests/agents/test_actor.py`, then append these tests to the end of the file:

```python
# ---------------------------------------------------------------------------
# Integration tests: MultiAgentOrchestrator + DecisionalActor
# ---------------------------------------------------------------------------

from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.field.field import PatternField
from hpm.store.sqlite import SQLiteStore
import uuid


def _make_agent(store, dim=4):
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"
    config = AgentConfig(feature_dim=dim, agent_id=agent_id)
    return Agent(config, store=store)


def test_orchestrator_no_actor(tmp_path):
    """actor=None (default) → actor_report == {} in step result."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field)  # no actor kwarg
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("actor_report") == {}


def test_orchestrator_actor_integrated(tmp_path):
    """With actor wired in, step returns actor_report with all 4 keys."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_agent(store)
    field = PatternField()

    action_vectors = np.eye(4)
    actor = DecisionalActor(action_vectors=action_vectors)

    orch = MultiAgentOrchestrator([agent], field=field, actor=actor)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)

    assert "actor_report" in result
    ar = result["actor_report"]
    expected_keys = {"external_action", "internal_action", "external_q_values", "internal_q_values"}
    assert set(ar.keys()) == expected_keys
```

- [ ] **Step 2.2 — Run integration tests to verify failure**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && pytest tests/agents/test_actor.py::test_orchestrator_no_actor tests/agents/test_actor.py::test_orchestrator_actor_integrated -v 2>&1 | head -20
```

Expected: `TypeError` (unexpected keyword argument `actor`) or `KeyError` (`actor_report` missing).

- [ ] **Step 2.3 — Modify `hpm/agents/multi_agent.py`**

**Change 1 — Add `actor=None` to `__init__`:**

In the `__init__` signature, add `actor=None` after `forecaster=None`:
```python
def __init__(
    self,
    agents: list,
    field: PatternField,
    seed_pattern: GaussianPattern | None = None,
    groups: dict | None = None,
    monitor=None,
    strategist=None,
    bridge=None,
    forecaster=None,
    actor=None,          # ← add this
):
```

In the constructor body, add after `self.forecaster = forecaster`:
```python
    self.actor = actor
```

**Change 2 — Call actor in `step()` and add to return:**

After the `forecast_report = (...)` block, add:
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

Change the return statement from:
```python
        return {
            **metrics,
            "field_quality": field_quality,
            "interventions": interventions,
            "bridge_report": bridge_report,
            "forecast_report": forecast_report,
        }
```
to:
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

- [ ] **Step 2.4 — Modify `hpm/agents/__init__.py`**

Add the export:
```python
from .actor import DecisionalActor

__all__ = ["DecisionalActor"]
```

(If the file already exports other names, append `DecisionalActor` to the existing `__all__`.)

- [ ] **Step 2.5 — Run all 14 tests**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && pytest tests/agents/test_actor.py -v
```

Expected: all 14 tests pass.

Also run the full suite to check for regressions:
```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && pytest --tb=short -q
```

Expected: no regressions.

- [ ] **Step 2.6 — Commit**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm/agents/multi_agent.py hpm/agents/__init__.py tests/agents/test_actor.py
git commit -m "$(cat <<'EOF'
feat: wire DecisionalActor into MultiAgentOrchestrator + export

- Add actor=None kwarg to MultiAgentOrchestrator.__init__()
- Call actor.step() after forecast_report in step()
- Return actor_report key in step() result dict
- Export DecisionalActor from hpm/agents/__init__.py
- 2 orchestrator integration tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
