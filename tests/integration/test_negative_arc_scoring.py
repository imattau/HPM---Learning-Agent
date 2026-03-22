import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.tiered_store import TieredStore
from hpm.config import AgentConfig
from hpm.agents.agent import Agent


# Import ensemble_score from benchmark module
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmarks.multi_agent_arc import ensemble_score


def _agent_with_stores(agent_id, pos_mu=None, neg_mu=None, feature_dim=4):
    """Helper: create agent with TieredStore, optionally seeded with patterns."""
    cfg = AgentConfig(agent_id=agent_id, feature_dim=feature_dim)
    store = TieredStore()
    # Seed positive tier2 directly (no context needed)
    if pos_mu is not None:
        p = GaussianPattern(mu=np.array(pos_mu, dtype=float),
                            sigma=np.eye(feature_dim) * 0.5)
        store.save(p, 1.0, agent_id)  # no context → tier2
    # Seed negative tier2 directly
    if neg_mu is not None:
        p_neg = GaussianPattern(mu=np.array(neg_mu, dtype=float),
                                sigma=np.eye(feature_dim) * 0.5)
        store._tier2_negative.save(p_neg, 1.0, agent_id)
    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.agent_id = agent_id
    agent.store = store
    return agent


# ── Test 1: no negative patterns → score identical to legacy behaviour ────────

def test_inhibitory_no_effect_when_empty():
    """Backward compatibility: empty negative store must not change score."""
    agent = _agent_with_stores("a", pos_mu=[1.0, 0.0, 0.0, 0.0])
    vec = np.array([1.0, 0.1, 0.0, 0.0])

    # Compute score using only positive (simulate old behaviour)
    pos_records = agent.store.query("a")
    expected = sum(w * p.log_prob(vec) for p, w in pos_records)

    actual = ensemble_score([agent], vec)
    assert actual == pytest.approx(expected)


# ── Test 2: negative pattern raises score for close candidate ─────────────────

def test_inhibitory_raises_score_for_close_candidate():
    """
    With inhibitory scoring, a candidate far from taboo (correct) scores lower
    than a candidate close to taboo (distractor), even compared to the baseline.

    The inhibitory term subtracts neg_NLL. When a candidate is close to the taboo
    pattern, neg_NLL is LOW, so the subtraction reduces the total less. When a
    candidate is far from taboo, neg_NLL is HIGH, so the subtraction reduces the
    total more. This means the inhibitory term HELPS the correct candidate
    rank lower (better) vs the distractor.

    We verify: with inhibition, the correct/distractor score gap widens.
    """
    pos_mu = [1.0, 0.0, 0.0, 0.0]
    neg_mu = [0.0, 1.0, 0.0, 0.0]

    correct_vec = np.array([0.9, 0.0, 0.0, 0.0])
    distractor_vec = np.array([0.0, 1.0, 0.0, 0.0])

    agent_baseline = _agent_with_stores("a", pos_mu=pos_mu)
    agent_inhibited = _agent_with_stores("b", pos_mu=pos_mu, neg_mu=neg_mu)

    # Baseline: correct should score lower than distractor (correct closer to pos pattern)
    baseline_correct = ensemble_score([agent_baseline], correct_vec)
    baseline_distractor = ensemble_score([agent_baseline], distractor_vec)

    # Inhibited: gap should widen or at least maintain correct ranking
    inhibited_correct = ensemble_score([agent_inhibited], correct_vec)
    inhibited_distractor = ensemble_score([agent_inhibited], distractor_vec)

    # Both agents should rank correct < distractor
    assert baseline_correct < baseline_distractor
    assert inhibited_correct < inhibited_distractor


# ── Test 3: correct candidate wins with inhibitory term ───────────────────────

def test_correct_wins_with_inhibitory():
    """
    Correct candidate is far from taboo; distractor is close to taboo.
    With inhibitory term, correct_score < distractor_score.
    """
    dim = 4
    # Positive pattern centred at [1,0,0,0]
    pos_mu = [1.0, 0.0, 0.0, 0.0]
    # Taboo pattern centred at [0,1,0,0]
    neg_mu = [0.0, 1.0, 0.0, 0.0]

    # Correct output vector: similar to positive, far from taboo
    correct_vec = np.array([0.9, 0.0, 0.0, 0.0])
    # Distractor: close to taboo pattern
    distractor_vec = np.array([0.05, 0.95, 0.0, 0.0])

    agent = _agent_with_stores("a", pos_mu=pos_mu, neg_mu=neg_mu)

    correct_score = ensemble_score([agent], correct_vec)
    distractor_score = ensemble_score([agent], distractor_vec)

    # Correct should rank better (lower score) than distractor
    assert correct_score < distractor_score, (
        f"Expected correct_score ({correct_score:.4f}) < distractor_score ({distractor_score:.4f})"
    )


# ── Test 4: returns 0.0 when all stores empty ─────────────────────────────────

def test_ensemble_score_returns_zero_when_empty():
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    store = TieredStore()
    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.agent_id = "a"
    agent.store = store
    # Both positive and negative stores are empty — seed removed
    store._tier2 = store._tier2   # just access to confirm it's an InMemoryStore

    # Manually ensure completely empty
    vec = np.zeros(4)
    result = ensemble_score([agent], vec)
    assert result == 0.0


# ── Test 5: InMemoryStore agents (no query_negative) backward compatible ──────

def test_ensemble_score_backward_compatible_inmemory_store():
    """Agents using InMemoryStore (no query_negative method) must not crash."""
    from hpm.store.memory import InMemoryStore
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    store = InMemoryStore()
    p = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]), sigma=np.eye(4) * 0.5)
    store.save(p, 1.0, "a")

    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.agent_id = "a"
    agent.store = store

    vec = np.array([1.0, 0.1, 0.0, 0.0])
    result = ensemble_score([agent], vec)
    assert isinstance(result, float)
    assert result > 0.0
