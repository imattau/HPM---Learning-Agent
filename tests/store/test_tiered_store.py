import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.tiered_store import TieredStore


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


def test_delete_does_not_remove_from_tier2():
    store = TieredStore()
    p = _pat()
    store.save(p, 1.0, "agent_a")          # goes to tier2
    store.begin_context("task_0")
    store.delete(p.id)                      # should NOT affect tier2
    store.end_context("task_0", correct=False)
    tier2 = store.query_tier2("agent_a")
    assert len(tier2) == 1                  # tier2 pattern still there


def test_begin_context_raises_if_already_active():
    store = TieredStore()
    store.begin_context("task_0")
    with pytest.raises(RuntimeError, match="already active"):
        store.begin_context("task_0")


def test_similarity_merge_promotes_novel_pattern_to_tier2():
    store = TieredStore()
    store.begin_context("task_0")
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
    store.save(p, 0.9, "agent_a")
    store.end_context("task_0", correct=True)

    # Novel pattern (no tier2 yet) should be promoted at half weight
    tier2 = store.query_tier2("agent_a")
    assert len(tier2) == 1
    assert tier2[0][1] == pytest.approx(0.45)   # half of 0.9


def test_similarity_merge_boosts_matching_tier2_pattern():
    store = TieredStore()

    # Seed a Tier 2 pattern at mu=[1,0,0,0]
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    p_t2 = GaussianPattern(mu=mu.copy(), sigma=np.eye(4) * 0.1)
    store.save(p_t2, 0.5, "agent_a")   # no context → tier2

    # Task produces very similar pattern (same direction, tiny offset)
    store.begin_context("task_0")
    p_t1 = GaussianPattern(mu=mu.copy() + 0.001, sigma=np.eye(4) * 0.1)
    store.save(p_t1, 0.8, "agent_a")
    store.end_context("task_0", correct=True)

    # Tier 2 weight should have been boosted
    tier2 = store.query_tier2("agent_a")
    assert tier2[0][1] > 0.5


def test_similarity_merge_skipped_on_incorrect_task():
    store = TieredStore()
    store.begin_context("task_0")
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
    store.save(p, 0.9, "agent_a")
    store.end_context("task_0", correct=False)  # task failed

    # Nothing promoted to tier2
    assert store.query_tier2("agent_a") == []


def test_similarity_merge_respects_max_tier2_patterns():
    """When Tier 2 is at cap, novel patterns are not promoted."""
    store = TieredStore()
    # Fill tier2 to cap of 2
    for i in range(2):
        mu = np.zeros(4)
        mu[i] = 1.0
        p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
        store.save(p, 1.0, "agent_a")   # no context → tier2

    # Task has a novel pattern (orthogonal to existing ones)
    store.begin_context("task_0")
    mu_novel = np.array([0.0, 0.0, 1.0, 0.0])
    p_novel = GaussianPattern(mu=mu_novel, sigma=np.eye(4) * 0.1)
    store.save(p_novel, 0.8, "agent_a")
    store.end_context("task_0", correct=True)

    # With max_tier2_patterns=2, novel pattern should NOT be promoted
    # But we need to call similarity_merge directly with the cap
    # Reset and redo with explicit cap
    store2 = TieredStore()
    for i in range(2):
        mu = np.zeros(4)
        mu[i] = 1.0
        p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
        store2.save(p, 1.0, "agent_a")
    store2.begin_context("task_x")
    p_novel2 = GaussianPattern(mu=np.array([0.0, 0.0, 1.0, 0.0]), sigma=np.eye(4) * 0.1)
    store2.save(p_novel2, 0.8, "agent_a")
    # Manually call similarity_merge with cap=2
    store2.similarity_merge("task_x", max_tier2_patterns=2)
    store2._tier1.pop("task_x", None)

    tier2 = store2.query_tier2("agent_a")
    assert len(tier2) == 2  # cap enforced, no new pattern added
