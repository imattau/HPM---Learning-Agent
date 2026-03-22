import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.agents.hierarchical import LevelBundle, encode_bundle
from hpm.agents.hierarchical import extract_bundle
from hpm.agents.hierarchical import (
    HierarchicalOrchestrator, make_hierarchical_orchestrator
)
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.store.memory import InMemoryStore
from hpm.field.field import PatternField


def test_encode_bundle_shape():
    bundle = LevelBundle(agent_id="a", mu=np.zeros(16), weight=0.5, epistemic_loss=0.1)
    encoded = encode_bundle(bundle)
    assert encoded.shape == (18,)


def test_encode_bundle_values():
    mu = np.ones(4)
    bundle = LevelBundle(agent_id="a", mu=mu, weight=0.7, epistemic_loss=0.3)
    encoded = encode_bundle(bundle)
    np.testing.assert_allclose(encoded[:4], mu)
    assert encoded[4] == pytest.approx(0.7)
    assert encoded[5] == pytest.approx(0.3)


def test_level_bundle_fields():
    b = LevelBundle(agent_id="x", mu=np.zeros(8), weight=1.0, epistemic_loss=0.0)
    assert b.agent_id == "x"
    assert b.mu.shape == (8,)
    assert b.weight == 1.0
    assert b.epistemic_loss == 0.0


def _make_agent(feature_dim=8, agent_id="test_agent"):
    cfg = AgentConfig(agent_id=agent_id, feature_dim=feature_dim)
    store = InMemoryStore()
    field = PatternField()
    return Agent(cfg, store=store, field=field)


def test_extract_bundle_empty_store_returns_zeros():
    agent = _make_agent(feature_dim=8)
    # Manually clear store to trigger empty-store guard
    agent.store._data.clear()
    bundle = extract_bundle(agent)
    assert bundle.agent_id == agent.agent_id
    assert bundle.mu.shape == (8,)
    np.testing.assert_allclose(bundle.mu, np.zeros(8))
    assert bundle.weight == 0.0
    assert bundle.epistemic_loss == 1.0


def test_extract_bundle_populated_returns_top_pattern():
    agent = _make_agent(feature_dim=4)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.standard_normal(4)
        agent.step(obs)
    bundle = extract_bundle(agent)
    assert bundle.agent_id == agent.agent_id
    assert bundle.mu.shape == (4,)
    assert isinstance(bundle.weight, float)
    assert isinstance(bundle.epistemic_loss, float)
    assert np.isfinite(bundle.mu).all()


def test_extract_bundle_top_is_highest_weight():
    agent = _make_agent(feature_dim=4)
    rng = np.random.default_rng(42)
    for _ in range(20):
        agent.step(rng.standard_normal(4))
    bundle = extract_bundle(agent)
    records = agent.store.query(agent.agent_id)
    max_weight = max(w for _, w in records)
    assert bundle.weight == pytest.approx(max_weight)


def test_hierarchical_orchestrator_cadence_k1():
    """K=1: Level 2 steps on every Level 1 step."""
    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=1,
    )
    rng = np.random.default_rng(0)
    l2_call_count = 0
    for _ in range(10):
        result = h_orch.step(rng.standard_normal(8))
        if result["level2"]:
            l2_call_count += 1
    assert l2_call_count == 10


def test_hierarchical_orchestrator_cadence_k5():
    """K=5: Level 2 steps only at t=5,10,15,20."""
    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=5,
    )
    rng = np.random.default_rng(0)
    l2_call_count = 0
    for _ in range(20):
        result = h_orch.step(rng.standard_normal(8))
        if result["level2"]:
            l2_call_count += 1
    assert l2_call_count == 4  # steps 5, 10, 15, 20


def test_hierarchical_orchestrator_no_l2_on_noncadence():
    """level2 key is {} on non-cadence steps."""
    h_orch, _, _ = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=5,
    )
    rng = np.random.default_rng(0)
    result = h_orch.step(rng.standard_normal(8))  # t=1, not a cadence step
    assert result["level2"] == {}


def test_hierarchical_orchestrator_k_larger_than_steps():
    """K > n_steps: Level 2 never steps, no error."""
    h_orch, _, _ = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=100,
    )
    rng = np.random.default_rng(0)
    for _ in range(10):
        result = h_orch.step(rng.standard_normal(8))
    assert result["level2"] == {}


def test_hierarchical_orchestrator_l2_bundle_shape():
    """Level 2 receives bundles of shape (l1_feature_dim + 2,)."""
    l1_dim = 8
    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=l1_dim, K=1,
    )
    rng = np.random.default_rng(0)
    h_orch.step(rng.standard_normal(l1_dim))
    assert l2_agents[0].config.feature_dim == l1_dim + 2


def test_hierarchical_orchestrator_returns_t():
    """step() return dict includes 't' counter."""
    h_orch, _, _ = make_hierarchical_orchestrator(
        n_l1_agents=1, n_l2_agents=1, l1_feature_dim=4, K=1,
    )
    for i in range(1, 4):
        result = h_orch.step(np.zeros(4))
        assert result["t"] == i
