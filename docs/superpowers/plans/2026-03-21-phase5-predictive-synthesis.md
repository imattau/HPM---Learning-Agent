# Phase 5: Predictive Synthesis Agent — Implementation Plan

**Goal:** Implement `PredictiveSynthesisAgent` ("The Forecaster") — a monitor-layer component that uses the highest-weight Level 4+ `GaussianPattern` μ as the system's primary structural prediction, then validates that prediction's robustness via an additive-noise Far Transfer probe. Wire it into `MultiAgentOrchestrator` as an optional `forecaster=None` kwarg.

**Architecture:**
- `hpm/monitor/predictive_synthesis.py` — new standalone class
- `tests/monitor/test_predictive_synthesis.py` — 11 unit + integration tests
- `hpm/monitor/__init__.py` — add export
- `hpm/agents/multi_agent.py` — add `forecaster=None` kwarg + call + return key

**Tech Stack:** Python stdlib (`collections.deque`), NumPy, pytest, SQLiteStore (real DB via `tmp_path`), GaussianPattern

---

## Task 1: PredictiveSynthesisAgent class + unit tests

**Files:**
- `tests/monitor/test_predictive_synthesis.py` (new — write first)
- `hpm/monitor/predictive_synthesis.py` (new — implement after tests)

---

### Step 1.1 — Write failing tests

```
pytest tests/monitor/test_predictive_synthesis.py
# Expected: collection error (file doesn't exist yet)
```

Create `tests/monitor/test_predictive_synthesis.py`:

```python
import numpy as np
import pytest
from hpm.store.sqlite import SQLiteStore
from hpm.patterns.gaussian import GaussianPattern
from hpm.monitor.predictive_synthesis import PredictiveSynthesisAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern(level, mu, sigma_scale=1.0, dim=None):
    """Create a GaussianPattern with the given level and mu."""
    mu_arr = np.array(mu, dtype=float)
    if dim is None:
        dim = len(mu_arr)
    p = GaussianPattern(mu=mu_arr, sigma=np.eye(dim) * sigma_scale)
    p.level = level
    return p


def _seed_store(store, agent_id, patterns_weights):
    """Save (pattern, weight) pairs into the store under agent_id."""
    for p, w in patterns_weights:
        store.save(p, w, agent_id)


# ---------------------------------------------------------------------------
# Test 1: Fast gate — no Level 4+ patterns
# ---------------------------------------------------------------------------

def test_uncertainty_when_no_l4_patterns(tmp_path):
    """When field_quality['level4plus_count'] == 0, return empty report immediately."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    # Seed only Level 2 patterns — store is queried only if gate passes
    p = _make_pattern(level=2, mu=[1.0, 2.0, 3.0, 4.0])
    store.save(p, 1.0, "a1")

    agent = PredictiveSynthesisAgent(store)
    report = agent.step(
        step_t=1,
        current_obs={"a1": np.zeros(4)},
        field_quality={"level4plus_count": 0},
    )

    assert report["prediction"] is None
    assert report["prediction_error"] is None
    assert report["fragility_score"] is None
    assert report["delta_nll"] is None
    assert report["fragility_flag"] is False
    assert report["top_pattern_level"] is None
    assert report["top_pattern_id"] is None


# ---------------------------------------------------------------------------
# Test 2: Prediction equals μ of highest-weight Level 4+ pattern
# ---------------------------------------------------------------------------

def test_prediction_is_top_pattern_mu(tmp_path):
    """With two Level 4 patterns, prediction == mu of higher-weight pattern."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    mu_low  = [1.0, 1.0, 1.0, 1.0]
    mu_high = [9.0, 9.0, 9.0, 9.0]
    p_low  = _make_pattern(level=4, mu=mu_low)
    p_high = _make_pattern(level=4, mu=mu_high)
    store.save(p_low,  0.2, "a1")
    store.save(p_high, 0.8, "a1")

    agent = PredictiveSynthesisAgent(store)
    report = agent.step(
        step_t=1,
        current_obs={"a1": np.zeros(4)},
        field_quality={"level4plus_count": 2},
    )

    np.testing.assert_array_almost_equal(report["prediction"], np.array(mu_high))
    assert report["top_pattern_level"] == 4


# ---------------------------------------------------------------------------
# Test 3: Fallback to Level 3 when no Level 4+ in store
# ---------------------------------------------------------------------------

def test_fallback_to_level3(tmp_path):
    """When store has no Level 4+ patterns, falls back to highest-weight Level 3."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    mu3 = [5.0, 5.0, 5.0, 5.0]
    p3 = _make_pattern(level=3, mu=mu3)
    store.save(p3, 0.9, "a1")

    # field_quality says level4plus_count=1 (timing discrepancy — gate passes)
    agent = PredictiveSynthesisAgent(store)
    report = agent.step(
        step_t=1,
        current_obs={"a1": np.zeros(4)},
        field_quality={"level4plus_count": 1},
    )

    assert report["top_pattern_level"] == 3
    np.testing.assert_array_almost_equal(report["prediction"], np.array(mu3))


# ---------------------------------------------------------------------------
# Test 4: prediction_error equals top_pattern.log_prob(x_obs)
# ---------------------------------------------------------------------------

def test_prediction_error_computed(tmp_path):
    """prediction_error matches direct log_prob call on top pattern."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    mu = [1.0, 2.0, 3.0, 4.0]
    p = _make_pattern(level=4, mu=mu, sigma_scale=1.0)
    store.save(p, 1.0, "a1")

    obs = np.array([1.5, 2.5, 3.5, 4.5])
    agent = PredictiveSynthesisAgent(store)
    report = agent.step(
        step_t=1,
        current_obs={"a1": obs},
        field_quality={"level4plus_count": 1},
    )

    expected_error = float(p.log_prob(obs))
    assert report["prediction_error"] == pytest.approx(expected_error)


# ---------------------------------------------------------------------------
# Test 5: Probe returns None before probe_k observations accumulated
# ---------------------------------------------------------------------------

def test_probe_none_before_k_obs(tmp_path):
    """fragility_score and delta_nll are None until probe_k observations are seen."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(level=4, mu=[0.0, 0.0, 0.0, 0.0])
    store.save(p, 1.0, "a1")

    probe_k = 5
    agent = PredictiveSynthesisAgent(store, probe_k=probe_k)

    for i in range(probe_k - 1):
        report = agent.step(
            step_t=i + 1,
            current_obs={"a1": np.random.randn(4)},
            field_quality={"level4plus_count": 1},
        )
        assert report["fragility_score"] is None, f"Expected None at step {i+1}"
        assert report["delta_nll"] is None, f"Expected None at step {i+1}"


# ---------------------------------------------------------------------------
# Test 6: Robust pattern (wide sigma, high threshold) not flagged
# ---------------------------------------------------------------------------

def test_robust_pattern_not_flagged(tmp_path):
    """Wide-sigma pattern with high fragility_threshold → fragility_flag=False."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    # Large sigma → log_prob is nearly flat; noise barely changes NLL
    p = _make_pattern(level=4, mu=[0.0, 0.0, 0.0, 0.0], sigma_scale=100.0)
    store.save(p, 1.0, "a1")

    probe_k = 3
    agent = PredictiveSynthesisAgent(
        store,
        probe_k=probe_k,
        probe_n=5,
        probe_sigma_scale=0.5,
        fragility_threshold=1000.0,  # very high — never triggered
    )

    obs = np.zeros(4)
    report = None
    for i in range(probe_k):
        report = agent.step(
            step_t=i + 1,
            current_obs={"a1": obs},
            field_quality={"level4plus_count": 1},
        )

    assert report["fragility_flag"] is False


# ---------------------------------------------------------------------------
# Test 7: Fragile pattern (narrow sigma, low threshold) flagged
# ---------------------------------------------------------------------------

def test_fragile_pattern_flagged(tmp_path):
    """Narrow-sigma pattern with low fragility_threshold → fragility_flag=True."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    # Very small sigma → log_prob is a sharp spike; small noise destroys fit
    p = _make_pattern(level=4, mu=[0.0, 0.0, 0.0, 0.0], sigma_scale=0.001)
    store.save(p, 1.0, "a1")

    probe_k = 3
    agent = PredictiveSynthesisAgent(
        store,
        probe_k=probe_k,
        probe_n=5,
        probe_sigma_scale=2.0,   # large noise relative to tiny sigma
        fragility_threshold=0.0,  # flag on any positive delta
    )

    obs = np.zeros(4)
    report = None
    for i in range(probe_k):
        report = agent.step(
            step_t=i + 1,
            current_obs={"a1": obs},
            field_quality={"level4plus_count": 1},
        )

    assert report["fragility_flag"] is True
    assert report["fragility_score"] is not None
    assert report["delta_nll"] is not None


# ---------------------------------------------------------------------------
# Test 8: probe_sigma_scale=0 → zero noise → delta_nll == 0
# ---------------------------------------------------------------------------

def test_noise_scaled_to_pattern_sigma(tmp_path):
    """With probe_sigma_scale=0, noise is zero; fragility_score equals prediction_error."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(level=4, mu=[1.0, 2.0, 3.0, 4.0], sigma_scale=1.0)
    store.save(p, 1.0, "a1")

    probe_k = 3
    agent = PredictiveSynthesisAgent(
        store,
        probe_k=probe_k,
        probe_n=5,
        probe_sigma_scale=0.0,  # zero noise
    )

    obs = np.array([1.0, 2.0, 3.0, 4.0])
    report = None
    for i in range(probe_k):
        report = agent.step(
            step_t=i + 1,
            current_obs={"a1": obs},
            field_quality={"level4plus_count": 1},
        )

    assert report["delta_nll"] == pytest.approx(0.0, abs=1e-9)
    assert report["fragility_score"] == pytest.approx(report["prediction_error"], abs=1e-9)


# ---------------------------------------------------------------------------
# Test 9: All 7 keys always present in forecast_report
# ---------------------------------------------------------------------------

def test_forecast_report_keys_always_present(tmp_path):
    """All 7 keys must be present whether or not the fast gate fires."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = PredictiveSynthesisAgent(store)

    expected_keys = {
        "prediction",
        "prediction_error",
        "fragility_score",
        "delta_nll",
        "fragility_flag",
        "top_pattern_level",
        "top_pattern_id",
    }

    # Gate fires (no level4+ count)
    report_empty = agent.step(
        step_t=1,
        current_obs={"a1": np.zeros(4)},
        field_quality={"level4plus_count": 0},
    )
    assert set(report_empty.keys()) == expected_keys

    # Gate does not fire (but store is empty — fallback path)
    report_fallback = agent.step(
        step_t=2,
        current_obs={"a1": np.zeros(4)},
        field_quality={"level4plus_count": 1},
    )
    assert set(report_fallback.keys()) == expected_keys
```

### Step 1.2 — Run tests to verify they fail

```bash
pytest tests/monitor/test_predictive_synthesis.py -v 2>&1 | head -40
```

Expected: `ImportError` or `ModuleNotFoundError` (file does not exist yet).

### Step 1.3 — Implement `hpm/monitor/predictive_synthesis.py`

Create `hpm/monitor/predictive_synthesis.py`:

```python
from collections import deque

import numpy as np


def _empty_report() -> dict:
    """Return the canonical empty forecast_report with all 7 keys."""
    return {
        "prediction": None,
        "prediction_error": None,
        "fragility_score": None,
        "delta_nll": None,
        "fragility_flag": False,
        "top_pattern_level": None,
        "top_pattern_id": None,
    }


class PredictiveSynthesisAgent:
    """
    HPM Phase 5 — Predictive Synthesis Agent ("The Forecaster").

    Uses the highest-weight Level 4+ GaussianPattern μ as the system's
    primary structural prediction, then measures robustness via a Far
    Transfer probe (additive noise + NLL scoring).

    Called every orchestrator step; returns a forecast_report dict.
    """

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

    def step(self, step_t: int, current_obs: dict, field_quality: dict) -> dict:
        """
        Called by MultiAgentOrchestrator every step.

        Args:
            step_t:        Current orchestrator step counter.
            current_obs:   Dict {agent_id: np.ndarray} of current observations.
            field_quality: Dict from StructuralLawMonitor.step() (or {}).

        Returns:
            forecast_report dict — all 7 keys always present.
        """
        # --- 3.1 Fast gate ---
        if field_quality.get("level4plus_count", 0) == 0:
            return _empty_report()

        # --- 3.2 Query store and select top pattern ---
        all_records = self._store.query_all()  # list of (pattern, weight, agent_id)

        # Filter to min_bridge_level (default 4)
        eligible = [
            (p, w) for p, w, _ in all_records
            if p.level >= self.min_bridge_level
        ]

        # Fallback to level 3 if no level 4+ found (timing discrepancy)
        fallback_level = None
        if not eligible:
            eligible = [
                (p, w) for p, w, _ in all_records
                if p.level >= 3
            ]
            if eligible:
                fallback_level = 3

        if not eligible:
            return _empty_report()

        # Select the highest-weight pattern
        top_pattern, _ = max(eligible, key=lambda pw: pw[1])
        top_level = fallback_level if fallback_level is not None else self.min_bridge_level
        # Refine level to the actual pattern level (handles fallback correctly)
        top_level = top_pattern.level

        # --- 3.3 Extract observation ---
        x_obs = next(iter(current_obs.values()))
        # Defensive trim/pad to match pattern dimension
        pat_dim = top_pattern.mu.shape[0]
        if x_obs.shape[0] > pat_dim:
            x_obs = x_obs[:pat_dim]
        elif x_obs.shape[0] < pat_dim:
            x_obs = np.concatenate([x_obs, np.zeros(pat_dim - x_obs.shape[0])])

        # --- 3.4 Predict and score ---
        prediction = top_pattern.mu.copy()
        prediction_error = float(top_pattern.log_prob(x_obs))

        # --- 3.5 Update obs history ---
        self._obs_history.append(x_obs.copy())

        # --- 3.6 Far Transfer probe ---
        fragility_score = None
        delta_nll = None
        fragility_flag = False

        if len(self._obs_history) >= self.probe_k:
            sigma_diag_mean = float(np.mean(np.diag(top_pattern.sigma)))
            sigma_probe = self.probe_sigma_scale * np.sqrt(max(sigma_diag_mean, 1e-9))

            rng = np.random.default_rng()
            nlls = []
            for x_hist in self._obs_history:
                for _ in range(self.probe_n):
                    noise = rng.normal(0.0, sigma_probe, size=x_hist.shape)
                    x_perturbed = x_hist + noise
                    nlls.append(float(top_pattern.log_prob(x_perturbed)))

            fragility_score = float(np.mean(nlls))
            delta_nll = fragility_score - prediction_error
            fragility_flag = delta_nll > self.fragility_threshold

        return {
            "prediction": prediction,
            "prediction_error": prediction_error,
            "fragility_score": fragility_score,
            "delta_nll": delta_nll,
            "fragility_flag": fragility_flag,
            "top_pattern_level": top_level,
            "top_pattern_id": top_pattern.id,
        }
```

### Step 1.4 — Run tests to verify they pass

```bash
pytest tests/monitor/test_predictive_synthesis.py -v
```

Expected: all 9 unit tests pass (green).

### Step 1.5 — Commit

```bash
git add hpm/monitor/predictive_synthesis.py tests/monitor/test_predictive_synthesis.py
git commit -m "feat: add PredictiveSynthesisAgent with Far Transfer probe (Phase 5)

Implements Forecaster class in hpm/monitor/predictive_synthesis.py:
- Fast gate on field_quality['level4plus_count']
- Top-weight Level 4+ pattern μ as prediction
- Fallback to Level 3 on timing discrepancy
- Additive-noise Far Transfer probe for fragility scoring
- 9 unit tests covering all paths

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: MultiAgentOrchestrator integration + `__init__` export

**Files:**
- `tests/monitor/test_predictive_synthesis.py` (append 2 integration tests — write first)
- `hpm/agents/multi_agent.py` (modify)
- `hpm/monitor/__init__.py` (modify)

---

### Step 2.1 — Write failing integration tests

Append to `tests/monitor/test_predictive_synthesis.py`:

```python
# ---------------------------------------------------------------------------
# Integration tests: MultiAgentOrchestrator + PredictiveSynthesisAgent
# ---------------------------------------------------------------------------

from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.field.field import PatternField


def _make_agent(store, dim=4):
    agent_id = f"agent_{id(store)}"
    config = AgentConfig(feature_dim=dim, agent_id=agent_id)
    return Agent(config, store=store)


def test_orchestrator_no_forecaster(tmp_path):
    """forecaster=None (default) → forecast_report == {} in step result."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field)  # no forecaster kwarg
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("forecast_report") == {}


def test_orchestrator_forecaster_integrated(tmp_path):
    """With a forecaster wired in, step returns forecast_report with all 7 keys."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    # Seed a Level 4 pattern so the gate passes
    p = _make_pattern(level=4, mu=[1.0, 2.0, 3.0, 4.0])
    store.save(p, 1.0, "a1")

    agent = _make_agent(store)
    field = PatternField()
    from hpm.monitor.structural_law import StructuralLawMonitor
    monitor = StructuralLawMonitor(store, T_monitor=1)
    forecaster = PredictiveSynthesisAgent(store, probe_k=10)

    orch = MultiAgentOrchestrator(
        [agent], field=field, monitor=monitor, forecaster=forecaster
    )
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)

    assert "forecast_report" in result
    fr = result["forecast_report"]
    expected_keys = {
        "prediction",
        "prediction_error",
        "fragility_score",
        "delta_nll",
        "fragility_flag",
        "top_pattern_level",
        "top_pattern_id",
    }
    assert set(fr.keys()) == expected_keys
```

### Step 2.2 — Run integration tests to verify they fail

```bash
pytest tests/monitor/test_predictive_synthesis.py::test_orchestrator_no_forecaster tests/monitor/test_predictive_synthesis.py::test_orchestrator_forecaster_integrated -v
```

Expected: `TypeError` (unexpected keyword argument `forecaster`) or `KeyError` (`forecast_report` not in result).

### Step 2.3 — Modify `hpm/agents/multi_agent.py`

**Change 1:** Add `forecaster=None` to the constructor signature:

```python
# Old:
def __init__(
    self,
    agents: list,
    field: PatternField,
    seed_pattern: GaussianPattern | None = None,
    groups: dict | None = None,
    monitor=None,
    strategist=None,
    bridge=None,
):
    self.agents = agents
    self._groups = groups
    self._group_fields: dict[str, PatternField] = {}
    self._t = 0
    self.monitor = monitor
    self.strategist = strategist
    self.bridge = bridge

# New:
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
):
    self.agents = agents
    self._groups = groups
    self._group_fields: dict[str, PatternField] = {}
    self._t = 0
    self.monitor = monitor
    self.strategist = strategist
    self.bridge = bridge
    self.forecaster = forecaster
```

**Change 2:** Add forecaster call after `bridge_report` in `step()`, and add to return dict:

```python
# Old return block at end of step():
        bridge_report = (
            self.bridge.step(self._t, field_quality)
            if self.bridge is not None
            else {}
        )

        return {**metrics, "field_quality": field_quality, "interventions": interventions, "bridge_report": bridge_report}

# New:
        bridge_report = (
            self.bridge.step(self._t, field_quality)
            if self.bridge is not None
            else {}
        )

        forecast_report = (
            self.forecaster.step(self._t, observations, field_quality)
            if self.forecaster is not None
            else {}
        )

        return {
            **metrics,
            "field_quality": field_quality,
            "interventions": interventions,
            "bridge_report": bridge_report,
            "forecast_report": forecast_report,
        }
```

### Step 2.4 — Modify `hpm/monitor/__init__.py`

```python
# Old:
from .structural_law import StructuralLawMonitor
from .recombination_strategist import RecombinationStrategist

__all__ = ["StructuralLawMonitor", "RecombinationStrategist"]

# New:
from .structural_law import StructuralLawMonitor
from .recombination_strategist import RecombinationStrategist
from .predictive_synthesis import PredictiveSynthesisAgent

__all__ = ["StructuralLawMonitor", "RecombinationStrategist", "PredictiveSynthesisAgent"]
```

### Step 2.5 — Run all tests to verify they pass

```bash
pytest tests/monitor/test_predictive_synthesis.py -v
```

Expected: all 11 tests pass (9 unit + 2 integration).

Also run the full suite to check for regressions:

```bash
pytest --tb=short -q
```

Expected: no regressions (all previously passing tests still pass).

### Step 2.6 — Commit

```bash
git add hpm/agents/multi_agent.py hpm/monitor/__init__.py tests/monitor/test_predictive_synthesis.py
git commit -m "feat: wire PredictiveSynthesisAgent into MultiAgentOrchestrator + export

- Add forecaster=None kwarg to MultiAgentOrchestrator.__init__()
- Call forecaster.step() after bridge_report in step()
- Return forecast_report key in step() result dict
- Export PredictiveSynthesisAgent from hpm/monitor/__init__.py
- 2 orchestrator integration tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
