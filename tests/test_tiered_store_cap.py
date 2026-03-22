"""
Tests for TieredStore tier2 cap enforcement.

Verifies that:
1. similarity_merge respects max_tier2_patterns (existing behaviour)
2. promote_to_tier2 respects max_tier2_patterns (bug fix)
3. promote_to_tier2 evicts the lowest-weight pattern when at capacity
"""
import numpy as np
import pytest
from hpm.store.tiered_store import TieredStore
from hpm.patterns.gaussian import GaussianPattern


def make_pattern(dim=4, seed=None):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(dim)
    mu /= np.linalg.norm(mu) + 1e-8
    return GaussianPattern(mu=mu, sigma=np.eye(dim))


class TestTier2CapViaSimilarityMerge:
    """similarity_merge already had cap logic — verify it still works."""

    def test_cap_respected_on_similarity_merge(self):
        store = TieredStore()
        cap = 5

        for i in range(cap + 3):
            ctx = f"ctx_{i}"
            store.begin_context(ctx)
            p = make_pattern(seed=i)
            store.save(p, 1.0, "agent_a")
            store.similarity_merge(ctx, max_tier2_patterns=cap)
            store._tier1.pop(ctx, None)

        assert len(store.query_tier2_all()) <= cap


class TestTier2CapViaPromote:
    """promote_to_tier2 must respect the cap (was the bug)."""

    def test_promote_respects_cap(self):
        store = TieredStore()
        cap = 5

        for i in range(cap + 10):
            p = make_pattern(seed=i)
            store.promote_to_tier2(p, float(i + 1), "agent_a",
                                   max_tier2_patterns=cap)

        assert len(store.query_tier2_all()) <= cap

    def test_promote_evicts_lowest_weight(self):
        store = TieredStore()
        cap = 3

        # Fill to cap with known weights
        patterns = []
        for i in range(cap):
            p = make_pattern(seed=i)
            patterns.append(p)
            store.promote_to_tier2(p, float(i + 1), "agent_a",
                                   max_tier2_patterns=cap)

        # At capacity: weights are 1.0, 2.0, 3.0
        assert len(store.query_tier2_all()) == cap

        # Promote one more — should evict weight=1.0 pattern
        new_p = make_pattern(seed=100)
        store.promote_to_tier2(new_p, 5.0, "agent_a", max_tier2_patterns=cap)

        assert len(store.query_tier2_all()) == cap

        # The lowest-weight (weight=1.0, seed=0) should be gone
        ids_present = {rec[0].id for rec in store.query_tier2_all()}
        assert patterns[0].id not in ids_present  # evicted
        assert new_p.id in ids_present             # promoted

    def test_promote_below_cap_does_not_evict(self):
        store = TieredStore()
        cap = 10

        for i in range(5):
            p = make_pattern(seed=i)
            store.promote_to_tier2(p, 1.0, "agent_a", max_tier2_patterns=cap)

        # No eviction — all 5 should survive
        assert len(store.query_tier2_all()) == 5


class TestTier2NegativeCap:
    """negative_merge cap is enforced; direct save via agent should also be capped."""

    def test_negative_merge_cap(self):
        store = TieredStore()
        cap = 3

        # Seed Tier 2 with a positive pattern so negative_merge has something to compare against
        pos = make_pattern(seed=0)
        store._tier2.save(pos, 1.0, "agent_a")

        for i in range(cap + 5):
            ctx = f"ctx_{i}"
            store.begin_context(ctx)
            # Use same direction as the positive pattern so cosine sim > threshold
            p = GaussianPattern(mu=pos.mu.copy(), sigma=np.eye(len(pos.mu)))
            store.save(p, 1.0, "agent_a")
            store.negative_merge(ctx, neg_conflict_threshold=0.5,
                                 max_tier2_negative=cap)
            store._tier1.pop(ctx, None)

        assert len(store.query_tier2_negative_all()) <= cap
