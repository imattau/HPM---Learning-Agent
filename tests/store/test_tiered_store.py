import numpy as np
import pytest
from hpm.store.tiered_store import TieredStore
from hpm.patterns.gaussian import GaussianPattern


def _pat(seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(4)
    return GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)


def test_save_goes_to_tier1_during_context():
    store = TieredStore()
    store.begin_context("task_0")
    p = _pat()
    store.save(p, 1.0, "agent_a")
    assert store.query_tier2("agent_a") == []
    assert len(store.query("agent_a")) == 1


def test_save_without_context_goes_to_tier2():
    store = TieredStore()
    p = _pat()
    store.save(p, 1.0, "agent_a")
    assert len(store.query_tier2("agent_a")) == 1


def test_update_weight_does_not_mutate_tier2():
    store = TieredStore()
    p = _pat()
    store.save(p, 1.0, "agent_a")          # goes to tier2
    store.begin_context("task_0")
    store.update_weight(p.id, 0.01)        # must NOT change tier2 weight
    store.end_context("task_0", correct=False)
    tier2 = store.query_tier2("agent_a")
    assert tier2[0][1] == pytest.approx(1.0)  # unchanged


def test_query_returns_tier1_plus_tier2_during_context():
    store = TieredStore()
    p2 = _pat(seed=0)
    store.save(p2, 0.8, "agent_a")         # tier2
    store.begin_context("task_0")
    p1 = _pat(seed=1)
    store.save(p1, 1.0, "agent_a")         # tier1
    results = store.query("agent_a")
    assert len(results) == 2


def test_end_context_clears_tier1():
    store = TieredStore()
    store.begin_context("task_0")
    store.save(_pat(), 1.0, "agent_a")
    store.end_context("task_0", correct=False)
    assert store.query("agent_a") == []    # tier1 gone, tier2 empty too


def test_delete_removes_from_tier1():
    store = TieredStore()
    store.begin_context("task_0")
    p = _pat()
    store.save(p, 1.0, "agent_a")
    store.delete(p.id)
    assert store.query("agent_a") == []
