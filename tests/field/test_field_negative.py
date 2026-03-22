import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.field.field import PatternField


def _pat(seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(4)
    return GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)


# ── Test 1: broadcast stores pattern; another agent can pull it ──────────────

def test_broadcast_negative_stores_pattern():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 1.0)
    assert len(pulled) == 1
    pulled_pattern, pulled_weight = pulled[0]
    assert pulled_pattern is p
    assert pulled_weight == pytest.approx(1.0)


# ── Test 2: agent does NOT pull its own patterns back ────────────────────────

def test_pull_negative_excludes_own():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    # agent_a pulls — should not see its own pattern
    pulled = field.pull_negative("agent_a", 1.0)
    assert pulled == []


# ── Test 3: gamma_neg attenuates weight ──────────────────────────────────────

def test_pull_negative_attenuates_weight():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 0.3)
    assert len(pulled) == 1
    _, weight = pulled[0]
    assert weight == pytest.approx(0.3)


# ── Test 4: multiple agents broadcast; third agent pulls all ─────────────────

def test_multiple_agents_broadcast():
    field = PatternField()
    p_a = _pat(0)
    p_b = _pat(1)
    field.broadcast_negative(p_a, 1.0, "agent_a")
    field.broadcast_negative(p_b, 0.8, "agent_b")

    pulled_by_c = field.pull_negative("agent_c", 0.5)
    assert len(pulled_by_c) == 2
    weights = sorted(w for _, w in pulled_by_c)
    assert weights[0] == pytest.approx(0.4)   # 0.8 * 0.5
    assert weights[1] == pytest.approx(0.5)   # 1.0 * 0.5


# ── Test 5: gamma_neg=0 returns patterns with zero weight ───────────────────

def test_pull_negative_gamma_zero():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 0.0)
    assert len(pulled) == 1
    _, weight = pulled[0]
    assert weight == pytest.approx(0.0)


# ── Test 6: _negative channel is independent from positive channel ───────────

def test_negative_channel_independent_from_positive():
    field = PatternField()
    p = _pat(0)
    # Register positive pattern
    field.register("agent_a", [(p.id, 1.0)])
    # Broadcast negative pattern (different object)
    p_neg = _pat(1)
    field.broadcast_negative(p_neg, 1.0, "agent_a")

    # Positive pull (via freq) is unaffected
    assert field.freq(p.id) > 0.0
    # Negative channel returns exactly one entry
    pulled = field.pull_negative("agent_b", 1.0)
    assert len(pulled) == 1
    assert pulled[0][0] is p_neg


# ── Test 7: clearing _negative before re-broadcast prevents accumulation ─────

def test_negative_not_cumulative_after_reset():
    field = PatternField()
    p1 = _pat(0)
    p2 = _pat(1)

    # Step 1: agent_a broadcasts p1
    field._negative["agent_a"] = []
    field.broadcast_negative(p1, 1.0, "agent_a")

    # Step 2: agent_a resets and broadcasts p2 only
    field._negative["agent_a"] = []
    field.broadcast_negative(p2, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 1.0)
    assert len(pulled) == 1
    assert pulled[0][0] is p2  # only p2, not p1 + p2
