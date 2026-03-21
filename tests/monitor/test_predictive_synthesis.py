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
# Test 2: Prediction equals mu of highest-weight Level 4+ pattern
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

    # Simulate timing discrepancy: gate passes (level4plus_count=1) but
    # store has no Level 4+ patterns yet — should silently fall back to Level 3.
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
    """Wide-sigma pattern with high fragility_threshold -> fragility_flag=False."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(level=4, mu=[0.0, 0.0, 0.0, 0.0], sigma_scale=100.0)
    store.save(p, 1.0, "a1")

    probe_k = 3
    agent = PredictiveSynthesisAgent(
        store,
        probe_k=probe_k,
        probe_n=5,
        probe_sigma_scale=0.5,
        fragility_threshold=1000.0,
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
    """Narrow-sigma pattern with low fragility_threshold -> fragility_flag=True."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(level=4, mu=[0.0, 0.0, 0.0, 0.0], sigma_scale=0.001)
    store.save(p, 1.0, "a1")

    probe_k = 3
    agent = PredictiveSynthesisAgent(
        store,
        probe_k=probe_k,
        probe_n=5,
        probe_sigma_scale=2.0,
        fragility_threshold=0.0,
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
# Test 8: probe_sigma_scale=0 -> zero noise -> delta_nll == 0
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
        probe_sigma_scale=0.0,
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

    report_empty = agent.step(
        step_t=1,
        current_obs={"a1": np.zeros(4)},
        field_quality={"level4plus_count": 0},
    )
    assert set(report_empty.keys()) == expected_keys

    report_fallback = agent.step(
        step_t=2,
        current_obs={"a1": np.zeros(4)},
        field_quality={"level4plus_count": 1},
    )
    assert set(report_fallback.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration tests: MultiAgentOrchestrator + PredictiveSynthesisAgent
# ---------------------------------------------------------------------------

from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.agent import Agent
from hpm.field.field import PatternField


def _make_agent(store, dim=4):
    from hpm.config import AgentConfig
    import uuid
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"
    config = AgentConfig(feature_dim=dim, agent_id=agent_id)
    return Agent(config, store=store)


def test_orchestrator_no_forecaster(tmp_path):
    """forecaster=None (default) -> forecast_report == {} in step result."""
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
