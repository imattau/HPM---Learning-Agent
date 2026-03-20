import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.memory import InMemoryStore


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def pattern(dim):
    return GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))


def test_save_and_load(store, pattern):
    store.save(pattern, weight=0.8, agent_id="agent1")
    loaded_p, loaded_w = store.load(pattern.id)
    assert loaded_p.id == pattern.id
    assert loaded_w == pytest.approx(0.8)


def test_query_returns_agent_patterns(store, pattern, dim):
    other = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    store.save(pattern, 0.6, "agent1")
    store.save(other, 0.4, "agent2")
    results = store.query("agent1")
    assert len(results) == 1
    assert results[0][0].id == pattern.id


def test_update_weight(store, pattern):
    store.save(pattern, 0.5, "agent1")
    store.update_weight(pattern.id, 0.9)
    _, w = store.load(pattern.id)
    assert w == pytest.approx(0.9)


def test_delete(store, pattern):
    store.save(pattern, 1.0, "agent1")
    store.delete(pattern.id)
    assert store.query("agent1") == []


def test_query_all(store, pattern, dim):
    other = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    store.save(pattern, 0.6, "agent1")
    store.save(other, 0.4, "agent2")
    all_records = store.query_all()
    assert len(all_records) == 2
