import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.store.tiered_store import TieredStore
from hpm.patterns.gaussian import GaussianPattern


def _cfg(agent_id, gamma_neg=0.3):
    return AgentConfig(
        agent_id=agent_id,
        feature_dim=4,
        gamma_neg=gamma_neg,
        gamma_soc=0.5,
    )


def _obs(seed=0):
    return np.random.default_rng(seed).standard_normal(4)


# ── Test 1: Agent with InMemoryStore does NOT crash (backward compatible) ─────

def test_step_backward_compatible_no_tiered_store():
    """Agents using InMemoryStore (no _tier2_negative) must still work."""
    from hpm.store.memory import InMemoryStore
    cfg = _cfg("agent_a")
    agent = Agent(cfg, store=InMemoryStore())
    # No field — pure single agent, no negative channel
    result = agent.step(_obs(0))
    assert "t" in result
    assert result["t"] == 1


# ── Test 2: Agent pulls negative patterns from field into _tier2_negative ─────

def test_agent_pulls_negative_from_field():
    field = PatternField()
    store_a = TieredStore()
    store_b = TieredStore()

    agent_a = Agent(_cfg("agent_a"), store=store_a, field=field)
    agent_b = Agent(_cfg("agent_b"), store=store_b, field=field)

    # Seed agent_a's negative store directly
    p_taboo = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]),
                              sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p_taboo, 0.8, "agent_a")

    # agent_a steps: broadcasts its negative to field
    agent_a.step(_obs(0))

    # agent_b steps: pulls negative from field into its own _tier2_negative
    agent_b.step(_obs(1))

    neg_b = store_b.query_negative("agent_b")
    assert len(neg_b) >= 1
    # Weight should be attenuated by gamma_neg=0.3: 0.8 * 0.3 = 0.24
    weights = [w for _, w in neg_b]
    assert any(abs(w - 0.24) < 0.01 for w in weights)


# ── Test 3: Agent does NOT pull its own negative patterns back ────────────────

def test_agent_does_not_pull_own_negative():
    field = PatternField()
    store_a = TieredStore()

    agent_a = Agent(_cfg("agent_a"), store=store_a, field=field)

    p_taboo = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]),
                              sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p_taboo, 0.8, "agent_a")

    before_count = len(store_a.query_negative("agent_a"))
    agent_a.step(_obs(0))  # broadcasts own patterns; should NOT pull own back

    after_count = len(store_a.query_negative("agent_a"))
    # After step, count should be same or less (not doubled)
    # pull_negative excludes own agent_id, so no self-import occurs
    assert after_count == before_count


# ── Test 4: gamma_neg=0 prevents any taboo import ────────────────────────────

def test_gamma_neg_zero_prevents_taboo_import():
    field = PatternField()
    store_a = TieredStore()
    store_b = TieredStore()

    agent_a = Agent(_cfg("agent_a", gamma_neg=0.0), store=store_a, field=field)
    agent_b = Agent(_cfg("agent_b", gamma_neg=0.0), store=store_b, field=field)

    p_taboo = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]),
                              sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p_taboo, 0.8, "agent_a")

    agent_a.step(_obs(0))   # broadcasts
    agent_b.step(_obs(1))   # pulls with gamma_neg=0.0

    # Patterns arrive with weight 0.0 — still saved (weight=0), or none saved.
    # Key invariant: no non-zero weight patterns added.
    neg_b = store_b.query_negative("agent_b")
    assert all(w == pytest.approx(0.0) for _, w in neg_b)


# ── Test 5: field _negative cleared and re-broadcast each step (not cumulative)

def test_negative_broadcast_not_cumulative():
    field = PatternField()
    store_a = TieredStore()

    agent_a = Agent(_cfg("agent_a"), store=store_a, field=field)

    p1 = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]), sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p1, 0.8, "agent_a")

    agent_a.step(_obs(0))   # step 1: broadcasts p1
    count_after_step1 = len(field._negative.get("agent_a", []))

    agent_a.step(_obs(1))   # step 2: clears and re-broadcasts p1 (only once)
    count_after_step2 = len(field._negative.get("agent_a", []))

    # Must not grow each step
    assert count_after_step2 == count_after_step1
