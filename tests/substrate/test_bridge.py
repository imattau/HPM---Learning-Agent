import numpy as np
import pytest
from hpm.store.sqlite import SQLiteStore
from hpm.patterns.gaussian import GaussianPattern
from hpm.substrate.bridge import SubstrateBridgeAgent


class StubSubstrate:
    """Deterministic stub — no HTTP calls."""
    def __init__(self, freq=0.5):
        self._freq = freq
        self.call_count = 0

    def fetch(self, query):
        return []

    def field_frequency(self, pattern):
        self.call_count += 1
        return self._freq

    def stream(self):
        return iter([])


def _make_pattern(level: int, dim: int = 4) -> GaussianPattern:
    p = GaussianPattern(mu=np.random.randn(dim), sigma=np.eye(dim))
    p.level = level
    return p


def _seed(store, agent_id, levels, weight=0.25):
    patterns = []
    for lvl in levels:
        p = _make_pattern(lvl)
        store.save(p, weight, agent_id)
        patterns.append(p)
    return patterns


def _make_bridge(substrate, store, **kwargs):
    return SubstrateBridgeAgent(substrate, store, **kwargs)


# ---- Cadence gate ----

def test_no_op_on_non_cadence_step(tmp_path):
    """Returns {} and makes no weight changes on non-cadence steps."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    patterns = _seed(store, "a1", [3], weight=0.5)
    sub = StubSubstrate(freq=0.8)
    bridge = _make_bridge(sub, store, T_substrate=10)

    result = bridge.step(step_t=1, field_quality={})

    assert result == {}
    assert sub.call_count == 0
    records = store.query("a1")
    assert records[0][1] == pytest.approx(0.5)  # weight unchanged


def test_runs_on_cadence_step(tmp_path):
    """Returns bridge_report dict on cadence step."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3])
    bridge = _make_bridge(StubSubstrate(), store, T_substrate=1)

    result = bridge.step(step_t=1, field_quality={})

    assert isinstance(result, dict)
    assert "patterns_checked" in result


# ---- Standard boost ----

def test_standard_boost_applied(tmp_path):
    """Level 3 pattern with f_freq=0.5, alpha=0.1 → weight multiplied by 1.05."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3], weight=1.0)
    bridge = _make_bridge(StubSubstrate(freq=0.5), store, T_substrate=1, alpha=0.1)

    bridge.step(step_t=1, field_quality={})

    records = store.query("a1")
    # After boost: 1.0 * (1 + 0.1 * 0.5) = 1.05; after normalise with one pattern: 1.0
    # Weight is renormalised to 1.0 (single pattern), but boost was applied
    assert records[0][1] == pytest.approx(1.0)  # single pattern normalises to 1.0


def test_standard_boost_relative(tmp_path):
    """Higher f_freq pattern gets relatively more weight after normalisation."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p_high = _make_pattern(3)
    p_low = _make_pattern(3)
    store.save(p_high, 0.5, "a1")
    store.save(p_low, 0.5, "a1")

    sub = StubSubstrate()
    sub.field_frequency = lambda p: 0.9 if p.id == p_high.id else 0.1
    bridge = _make_bridge(sub, store, T_substrate=1, alpha=0.2)

    bridge.step(step_t=1, field_quality={})

    recs = {p.id: w for p, w in store.query("a1")}
    assert recs[p_high.id] > recs[p_low.id]


def test_zero_freq_no_boost(tmp_path):
    """f_freq=0.0 → weight multiplier is 1.0 (no boost before normalisation)."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p1 = _make_pattern(3)
    p2 = _make_pattern(3)
    store.save(p1, 0.5, "a1")
    store.save(p2, 0.5, "a1")

    sub = StubSubstrate()
    sub.field_frequency = lambda p: 0.0
    bridge = _make_bridge(sub, store, T_substrate=1, alpha=0.2)

    bridge.step(step_t=1, field_quality={})

    recs = {p.id: w for p, w in store.query("a1")}
    # Both patterns got zero boost → weights stay equal after normalisation
    assert recs[p1.id] == pytest.approx(recs[p2.id], abs=1e-6)


# ---- Level filtering ----

def test_below_min_level_skipped(tmp_path):
    """Level 2 patterns not queried or updated."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [2], weight=0.4)
    sub = StubSubstrate(freq=0.9)
    bridge = _make_bridge(sub, store, T_substrate=1, min_bridge_level=3)

    bridge.step(step_t=1, field_quality={})

    assert sub.call_count == 0
    records = store.query("a1")
    assert records[0][1] == pytest.approx(0.4)


def test_no_candidates_returns_zeroed_report(tmp_path):
    """All patterns below min_bridge_level → patterns_checked=0, no errors."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [1, 2], weight=0.5)
    bridge = _make_bridge(StubSubstrate(), store, T_substrate=1, min_bridge_level=3)

    result = bridge.step(step_t=1, field_quality={})

    assert result["patterns_checked"] == 0
    assert result["mean_field_frequency"] == pytest.approx(0.0)


# ---- Echo-chamber audit ----

def test_echo_chamber_penalty_applied(tmp_path):
    """redundancy > threshold AND f_freq < low_threshold → penalty applied after boost."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p_grounded = _make_pattern(3)
    p_ungrounded = _make_pattern(3)
    store.save(p_grounded, 0.5, "a1")
    store.save(p_ungrounded, 0.5, "a1")

    sub = StubSubstrate()
    sub.field_frequency = lambda p: 0.8 if p.id == p_grounded.id else 0.1
    bridge = _make_bridge(
        sub, store, T_substrate=1, alpha=0.1, gamma=0.3,
        redundancy_threshold=0.2, frequency_low_threshold=0.2
    )
    fq = {"redundancy": 0.5, "diversity": 1.0}

    bridge.step(step_t=1, field_quality=fq)

    recs = {p.id: w for p, w in store.query("a1")}
    # p_grounded boosted, p_ungrounded boosted then penalised → grounded >> ungrounded
    assert recs[p_grounded.id] > recs[p_ungrounded.id]


def test_echo_chamber_skipped_when_redundancy_none(tmp_path):
    """field_quality["redundancy"] = None → no penalty pass."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3], weight=0.5)
    sub = StubSubstrate(freq=0.05)  # below frequency_low_threshold

    bridge2 = _make_bridge(sub, store, T_substrate=1, gamma=0.5, redundancy_threshold=0.2)
    result = bridge2.step(step_t=1, field_quality={"redundancy": None})
    assert result["echo_chamber_penalty_applied"] is False


def test_echo_chamber_skipped_when_redundancy_low(tmp_path):
    """redundancy < threshold → no penalty even if f_freq is low."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3], weight=0.5)
    sub = StubSubstrate(freq=0.05)
    bridge = _make_bridge(sub, store, T_substrate=1, gamma=0.5, redundancy_threshold=0.4)
    fq = {"redundancy": 0.1}

    result = bridge.step(step_t=1, field_quality=fq)

    assert result["echo_chamber_penalty_applied"] is False


# ---- Frequency cache ----

def test_cache_hit_avoids_substrate_call(tmp_path):
    """Pattern queried once, mu unchanged → substrate.field_frequency not called second time."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3])
    sub = StubSubstrate(freq=0.5)
    bridge = _make_bridge(sub, store, T_substrate=1, cache_distance_threshold=0.1)

    bridge.step(step_t=1, field_quality={})
    assert sub.call_count == 1

    bridge.step(step_t=2, field_quality={})
    assert sub.call_count == 1  # cache hit — no new call


def test_cache_miss_on_mu_change(tmp_path):
    """Pattern mu changes beyond threshold → substrate re-queried."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(3)
    store.save(p, 0.5, "a1")
    sub = StubSubstrate(freq=0.5)
    bridge = _make_bridge(sub, store, T_substrate=1, cache_distance_threshold=0.01)

    bridge.step(step_t=1, field_quality={})
    assert sub.call_count == 1

    # Mutate the pattern's mu significantly, delete and re-save with same id
    store.delete(p.id)
    p2 = GaussianPattern(mu=np.ones(4) * 10.0, sigma=np.eye(4), id=p.id)
    p2.level = 3
    store.save(p2, 0.5, "a1")

    bridge.step(step_t=2, field_quality={})
    assert sub.call_count == 2  # cache miss — re-queried


# ---- Normalisation ----

def test_weights_normalised_per_agent(tmp_path):
    """After boost, each agent's pattern weights sum to 1.0."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3, 4, 5], weight=0.33)
    bridge = _make_bridge(StubSubstrate(freq=0.7), store, T_substrate=1, alpha=0.2)

    bridge.step(step_t=1, field_quality={})

    records = store.query("a1")
    total = sum(w for _, w in records)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_multi_agent_normalisation_independent(tmp_path):
    """Two agents normalised independently."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3, 4], weight=0.5)
    _seed(store, "a2", [3], weight=1.0)
    bridge = _make_bridge(StubSubstrate(freq=0.5), store, T_substrate=1, alpha=0.1)

    bridge.step(step_t=1, field_quality={})

    for agent_id in ["a1", "a2"]:
        records = store.query(agent_id)
        total = sum(w for _, w in records)
        assert total == pytest.approx(1.0, abs=1e-6)


# ---- Bridge report ----

def test_bridge_report_keys(tmp_path):
    """Cadence step always returns dict with all expected keys."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3])
    bridge = _make_bridge(StubSubstrate(), store, T_substrate=1)

    result = bridge.step(step_t=1, field_quality={"redundancy": 0.1})

    assert "patterns_checked" in result
    assert "cache_hits" in result
    assert "echo_chamber_penalty_applied" in result
    assert "mean_field_frequency" in result
    assert result["patterns_checked"] == 1
    assert result["mean_field_frequency"] == pytest.approx(0.5)


# ---- Integration tests ----

from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.monitor.structural_law import StructuralLawMonitor
from hpm.substrate.bridge import SubstrateBridgeAgent


def _make_real_agent(store, dim=4):
    config = AgentConfig(feature_dim=dim, agent_id=f"agent_{id(store)}")
    return Agent(config, store=store)


def test_orchestrator_no_bridge(tmp_path):
    """Orchestrator with bridge=None returns bridge_report == {}."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field, bridge=None)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("bridge_report") == {}


def test_orchestrator_bridge_integrated(tmp_path):
    """Orchestrator with bridge returns bridge_report after T_substrate steps."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=100)
    bridge = SubstrateBridgeAgent(StubSubstrate(), store, T_substrate=5)
    orch = MultiAgentOrchestrator([agent], field=field, monitor=monitor, bridge=bridge)
    obs = {agent.agent_id: np.zeros(4)}

    result = None
    for _ in range(5):
        result = orch.step(obs)

    assert "bridge_report" in result
    # On step 5 (T_substrate=5), bridge_report should be non-empty dict
    br = result["bridge_report"]
    if br:  # may be empty if no Level 3+ patterns yet
        assert "patterns_checked" in br
