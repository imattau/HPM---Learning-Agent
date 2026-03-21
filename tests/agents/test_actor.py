import numpy as np
import pytest
from hpm.agents.actor import DecisionalActor
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.field.field import PatternField
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
    """Pattern centred on action_vectors[2]; temperature->0 -> selects action 2."""
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
    """prediction=None -> external_action=None."""
    actor = DecisionalActor(action_vectors=np.eye(3))
    result = actor.step(1, _make_field_quality(), _make_forecast(prediction=None))
    assert result["external_action"] is None


# ---------------------------------------------------------------------------
# Test 4: Internal action triggered on redundancy
# ---------------------------------------------------------------------------

def test_internal_action_triggered_on_redundancy():
    """redundancy > threshold -> internal_action is not None."""
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
    """fragility_flag=True with low redundancy -> internal_action is not None."""
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
    """redundancy=0, fragility_flag=False -> internal_action=None."""
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
    reward = 2.0 - 1.0 = 1.0; alpha_int=1.0 -> Q[EXPLOIT] = 1.0.
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
    # Q[0] = 10.0 + 1.0*(1.0 - 10.0) = 1.0
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
    """EXPLORE at lower bound -> min_bridge_level unchanged."""
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


def test_exploit_respects_upper_bound():
    """EXPLOIT at upper bound leaves min_bridge_level unchanged."""
    forecaster = StubForecaster(min_bridge_level=6)  # already at upper bound
    actor = DecisionalActor(
        action_vectors=np.eye(3),
        forecaster=forecaster,
        alpha_int=1.0,
        temperature=1e-9,
        redundancy_threshold=0.0,
        min_bridge_level_bounds=(2, 6),
    )
    # Force EXPLOIT (index 0) by setting q_values
    actor._internal_head.q_values[:] = [10.0, 0.0, 0.0]
    fq = _make_field_quality(redundancy=0.5)
    fr = _make_forecast()
    actor.step(0, fq, fr)
    assert forecaster.min_bridge_level == 6  # clamped, not incremented to 7


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


# ---------------------------------------------------------------------------
# Integration Test 13: Orchestrator with no actor → actor_report == {}
# ---------------------------------------------------------------------------

def test_orchestrator_no_actor():
    """actor=None -> actor_report == {}"""
    field = PatternField()
    agents = [Agent(AgentConfig(agent_id="a0", feature_dim=4), field=field)]
    orch = MultiAgentOrchestrator(agents, field)  # no actor kwarg
    obs = {"a0": np.zeros(4)}
    result = orch.step(obs)
    assert result["actor_report"] == {}


# ---------------------------------------------------------------------------
# Integration Test 14: Orchestrator with actor wired in → all 4 keys present
# ---------------------------------------------------------------------------

def test_orchestrator_actor_integrated():
    """actor wired in -> step returns actor_report with all 4 keys."""
    field = PatternField()
    agents = [Agent(AgentConfig(agent_id="a0", feature_dim=4), field=field)]
    actor = DecisionalActor(action_vectors=np.eye(3))
    orch = MultiAgentOrchestrator(agents, field, actor=actor)
    obs = {"a0": np.zeros(4)}
    result = orch.step(obs)
    expected_keys = {"external_action", "internal_action", "external_q_values", "internal_q_values"}
    assert set(result["actor_report"].keys()) == expected_keys
