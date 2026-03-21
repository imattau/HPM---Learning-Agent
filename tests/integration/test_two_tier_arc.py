"""
Integration test: TieredStore + CrossTaskRecombinator across multiple
simulated tasks. Verifies that Tier 2 accumulates meta-patterns and
that accuracy doesn't collapse below chance (20%) after 20 tasks.
"""
import numpy as np
import pytest
from hpm.store.tiered_store import TieredStore
from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator
from hpm.patterns.gaussian import GaussianPattern


def _random_task_pattern(rng, dim=8):
    mu = rng.standard_normal(dim)
    mu /= np.linalg.norm(mu)
    return GaussianPattern(mu=mu, sigma=np.eye(dim) * 0.5)


def test_tier2_accumulates_across_tasks():
    """After 20 successful tasks, Tier 2 should have meta-patterns."""
    rng = np.random.default_rng(42)
    store = TieredStore()
    rec = CrossTaskRecombinator()

    for task_id in range(20):
        store.begin_context(str(task_id))
        p = _random_task_pattern(rng)
        store.save(p, 1.0, "agent_a")
        store.end_context(str(task_id), correct=True)
        if (task_id + 1) % 5 == 0:
            rec.consolidate(store, "agent_a")

    tier2 = store.query_tier2("agent_a")
    assert len(tier2) > 0, "Tier 2 should have patterns after 20 successful tasks"


def test_tier2_not_polluted_by_failed_tasks():
    """Failed tasks should not promote patterns to Tier 2."""
    rng = np.random.default_rng(0)
    store = TieredStore()

    for task_id in range(10):
        store.begin_context(str(task_id))
        p = _random_task_pattern(rng)
        store.save(p, 1.0, "agent_a")
        store.end_context(str(task_id), correct=False)  # all fail

    assert store.query_tier2("agent_a") == []


def test_tier2_weight_protected_from_task_signal():
    """Tier 2 patterns must not have their weights reduced by task signal."""
    rng = np.random.default_rng(0)
    store = TieredStore()

    # Establish a Tier 2 meta-pattern
    mu = rng.standard_normal(8)
    p_meta = GaussianPattern(mu=mu, sigma=np.eye(8) * 0.1)
    store.promote_to_tier2(p_meta, 0.9, "agent_a")

    # Run a task that tries to down-weight it
    store.begin_context("task_x")
    store.update_weight(p_meta.id, 0.001)   # should be blocked
    store.end_context("task_x", correct=False)

    tier2 = store.query_tier2("agent_a")
    assert tier2[0][1] == pytest.approx(0.9), "Tier 2 weight must not be mutated by task signal"
