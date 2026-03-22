import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.tiered_store import TieredStore


def _pat(mu, dim=4):
    return GaussianPattern(mu=np.array(mu, dtype=float), sigma=np.eye(dim) * 0.1)


# ── Test 1: empty by default ────────────────────────────────────────────────

def test_query_negative_empty_by_default():
    store = TieredStore()
    assert store.query_negative("agent_a") == []
    assert store.query_tier2_negative_all() == []


# ── Test 2: end_context(correct=False) promotes conflicting pattern ──────────

def test_negative_merge_promotes_conflicting():
    store = TieredStore()
    # Seed a positive Tier 2 pattern at mu=[1,0,0,0]
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # no context → tier2

    # Failed task: Tier 1 pattern highly similar to the positive (cosine ~0.9998)
    store.begin_context("task_fail")
    p_neg = _pat([1.0, 0.05, 0.0, 0.0])
    store.save(p_neg, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    results = store.query_negative("agent_a")
    assert len(results) == 1
    pattern, weight = results[0]
    assert weight == pytest.approx(0.4)  # 0.8 * 0.5


# ── Test 3: end_context(correct=False) discards non-conflicting pattern ──────

def test_negative_merge_discards_non_conflicting():
    store = TieredStore()
    # Seed a positive Tier 2 pattern at mu=[1,0,0,0]
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    # Failed task: Tier 1 pattern orthogonal to positive (cosine = 0.0)
    store.begin_context("task_fail")
    p_ortho = _pat([0.0, 1.0, 0.0, 0.0])
    store.save(p_ortho, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    # Orthogonal → no conflict detected → discarded
    assert store.query_negative("agent_a") == []


# ── Test 4: max_tier2_negative cap is respected ──────────────────────────────

def test_negative_merge_respects_cap():
    store = TieredStore()
    # Seed one positive Tier 2 pattern
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")

    cap = 3

    # Fill _tier2_negative to the cap via direct saves (bypassing merge)
    for i in range(cap):
        dummy = _pat([1.0, 0.0, 0.0, 0.0])
        store._tier2_negative.save(dummy, 0.5, "agent_a")

    assert len(store.query_tier2_negative_all()) == cap

    # Now a failed task with a conflicting pattern — cap already reached
    store.begin_context("task_overflow")
    p_conflict = _pat([1.0, 0.02, 0.0, 0.0])
    store.save(p_conflict, 0.9, "agent_a")
    store.end_context("task_overflow", correct=False,
                      max_tier2_negative=cap)

    # Still exactly cap patterns — new one silently dropped
    assert len(store.query_tier2_negative_all()) == cap


# ── Test 5: end_context(correct=True) does NOT populate negative store ───────

def test_end_context_correct_no_negative_merge():
    store = TieredStore()
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    store.begin_context("task_ok")
    p_t1 = _pat([1.0, 0.01, 0.0, 0.0])
    store.save(p_t1, 0.8, "agent_a")
    store.end_context("task_ok", correct=True)

    assert store.query_negative("agent_a") == []


# ── Test 6: query() (positive) does NOT return negative patterns ─────────────

def test_positive_query_excludes_negative_patterns():
    store = TieredStore()
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    store.begin_context("task_fail")
    p_neg = _pat([1.0, 0.05, 0.0, 0.0])
    store.save(p_neg, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    positive_results = store.query("agent_a")
    neg_results = store.query_negative("agent_a")

    assert len(neg_results) == 1
    neg_id = neg_results[0][0].id
    positive_ids = {p.id for p, _ in positive_results}
    assert neg_id not in positive_ids


# ── Test 7: Tier 2 positive patterns unaffected by negative_merge ────────────

def test_tier2_positive_unaffected_by_negative_merge():
    store = TieredStore()
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    store.begin_context("task_fail")
    p_t1 = _pat([1.0, 0.02, 0.0, 0.0])
    store.save(p_t1, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    # Tier 2 positive weight must remain unchanged at 1.0
    tier2_pos = store.query_tier2("agent_a")
    assert len(tier2_pos) == 1
    assert tier2_pos[0][1] == pytest.approx(1.0)
