import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.tiered_store import TieredStore
from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator


def _make_store_with_tier2(n_patterns=2):
    """Two Tier 2 patterns at orthogonal positions."""
    store = TieredStore()
    for i in range(n_patterns):
        mu = np.zeros(4)
        mu[i] = 1.0
        p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
        store.save(p, 1.0, "agent_a")
    return store


def test_consolidate_creates_meta_pattern():
    # Orthogonal patterns have cosine sim=0.0; use similarity_lo=-0.5 to include them
    store = _make_store_with_tier2(2)
    rec = CrossTaskRecombinator(similarity_lo=-0.5, similarity_hi=0.9)
    rec.consolidate(store, agent_id="agent_a")
    # Should have promoted a recombinant: now 3 tier2 patterns
    tier2 = store.query_tier2("agent_a")
    assert len(tier2) == 3


def test_consolidate_skips_identical_patterns():
    """Patterns with cosine sim > similarity_hi are too similar — skip."""
    store = TieredStore()
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(2):
        p = GaussianPattern(mu=mu.copy(), sigma=np.eye(4) * 0.1)
        store.save(p, 1.0, "agent_a")
    rec = CrossTaskRecombinator(similarity_lo=-0.5, similarity_hi=0.9)
    rec.consolidate(store, agent_id="agent_a")
    # Identical patterns (sim=1.0 > similarity_hi=0.9) → no recombinant
    assert len(store.query_tier2("agent_a")) == 2


def test_consolidate_skips_too_dissimilar():
    """Patterns with cosine sim < similarity_lo are too different — skip."""
    store = TieredStore()
    mus = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0, 0.0])]
    for mu in mus:
        p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
        store.save(p, 1.0, "agent_a")
    rec = CrossTaskRecombinator(similarity_lo=0.1, similarity_hi=0.9)
    # Opposite vectors have sim=-1.0 < similarity_lo=0.1 → no recombinant
    rec.consolidate(store, agent_id="agent_a")
    assert len(store.query_tier2("agent_a")) == 2


def test_consolidate_returns_count_of_promoted():
    store = _make_store_with_tier2(2)
    rec = CrossTaskRecombinator(similarity_lo=-0.5, similarity_hi=0.9)
    count = rec.consolidate(store, agent_id="agent_a")
    assert count == 1


def test_consolidate_respects_max_recombinants():
    """With max_recombinants=0, nothing is promoted."""
    store = _make_store_with_tier2(2)
    rec = CrossTaskRecombinator(similarity_lo=-0.5, similarity_hi=0.9, max_recombinants=0)
    count = rec.consolidate(store, agent_id="agent_a")
    assert count == 0
    assert len(store.query_tier2("agent_a")) == 2
