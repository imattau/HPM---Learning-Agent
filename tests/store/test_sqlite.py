import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.sqlite import SQLiteStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_patterns.db")


@pytest.fixture
def store(db_path):
    return SQLiteStore(db_path)


@pytest.fixture
def pattern():
    return GaussianPattern(mu=np.array([1.0, 2.0]), sigma=np.eye(2))


def test_save_and_load(store, pattern):
    store.save(pattern, weight=0.7, agent_id="agent1")
    loaded_p, loaded_w = store.load(pattern.id)
    assert loaded_p.id == pattern.id
    assert loaded_w == pytest.approx(0.7)
    assert np.allclose(loaded_p.mu, pattern.mu)
    assert np.allclose(loaded_p.sigma, pattern.sigma)


def test_query_returns_agent_patterns(store, pattern):
    other = GaussianPattern(mu=np.array([5.0, 5.0]), sigma=np.eye(2))
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


def test_query_all(store, pattern):
    other = GaussianPattern(mu=np.array([3.0, 4.0]), sigma=np.eye(2))
    store.save(pattern, 0.6, "agent1")
    store.save(other, 0.4, "agent2")
    all_records = store.query_all()
    assert len(all_records) == 2


def test_persistence_across_connections(db_path, pattern):
    """Patterns saved in one connection are readable in a new one."""
    store1 = SQLiteStore(db_path)
    store1.save(pattern, 0.8, "agent1")

    store2 = SQLiteStore(db_path)
    results = store2.query("agent1")
    assert len(results) == 1
    assert results[0][0].id == pattern.id
