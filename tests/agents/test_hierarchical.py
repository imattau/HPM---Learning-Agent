import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.agents.hierarchical import LevelBundle, encode_bundle
from hpm.agents.hierarchical import extract_bundle
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
