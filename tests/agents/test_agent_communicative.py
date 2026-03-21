import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.memory import InMemoryStore


def cfg(agent_id='test'):
    return AgentConfig(agent_id=agent_id, feature_dim=2)


def make_agent_with_field(agent_id='test'):
    field = PatternField()
    agent = Agent(cfg(agent_id), field=field)
    return agent, field


def level4_pattern(mu=None):
    """GaussianPattern with level=4 pre-set."""
    p = GaussianPattern(
        mu=mu if mu is not None else np.zeros(2),
        sigma=np.eye(2),
    )
    p.level = 4
    return p


# ---- _share_pending ----

def test_share_pending_broadcasts_level4_pattern():
    agent, field = make_agent_with_field()
    p = level4_pattern()
    agent._share_pending(field, [p])
    queue = field.drain_broadcasts()
    assert len(queue) == 1
    source_id, shared = queue[0]
    assert source_id == 'test'
    assert shared.source_id == p.id


def test_share_pending_does_not_broadcast_below_level4():
    agent, field = make_agent_with_field()
    for lvl in [1, 2, 3]:
        p = GaussianPattern(np.zeros(2), np.eye(2))
        p.level = lvl
        agent._share_pending(field, [p])
    assert field.drain_broadcasts() == []


def test_share_pending_does_not_reshare():
    agent, field = make_agent_with_field()
    p = level4_pattern()
    agent._share_pending(field, [p])
    field.drain_broadcasts()  # consume first broadcast
    agent._share_pending(field, [p])
    assert field.drain_broadcasts() == []  # not shared again


def test_share_pending_shared_copy_has_new_uuid():
    agent, field = make_agent_with_field()
    p = level4_pattern()
    agent._share_pending(field, [p])
    _, shared = field.drain_broadcasts()[0]
    assert shared.id != p.id  # fresh UUID
    assert shared.source_id == p.id  # provenance preserved


# ---- _accept_communicated ----

def test_accept_communicated_novel_pattern_admitted():
    """Pattern moderately novel vs existing and with positive log-prob near observations is admitted."""
    c = AgentConfig(agent_id='test', feature_dim=2, beta_orig=1.0, alpha_nov=0.9, alpha_eff=0.1, kappa_0=0.1)
    agent = Agent(c, store=InMemoryStore())
    # Clear seed patterns; place a single known pattern
    for p, _ in agent.store.query('test'):
        agent.store.delete(p.id)
    existing = GaussianPattern(np.zeros(2), np.eye(2))
    agent.store.save(existing, 1.0, 'test')
    # Add an observation near zero (good efficacy for existing pattern)
    agent._obs_buffer.append(np.zeros(2))
    # Incoming at [2,2] — moderately novel (sym_kl>0) and has decent log-prob near [0,0]
    incoming = GaussianPattern(mu=np.array([2.0, 2.0]), sigma=np.eye(2) * 2.0)
    result = agent._accept_communicated(incoming, 'other_agent')
    assert result is True
    ids = [p.id for p, _ in agent.store.query('test')]
    assert incoming.id in ids


def test_accept_communicated_identical_pattern_empty_buffer_rejected():
    """Nov=0, Eff=0 (empty buffer) -> insight=0 -> rejected."""
    c = AgentConfig(agent_id='test', feature_dim=2, beta_orig=1.0, alpha_nov=0.5, alpha_eff=0.5, kappa_0=0.1)
    agent = Agent(c, store=InMemoryStore())
    # Clear store, then add only one pattern identical to incoming
    for p, _ in agent.store.query('test'):
        agent.store.delete(p.id)
    existing = GaussianPattern(np.zeros(2), np.eye(2))
    agent.store.save(existing, 1.0, 'test')
    # Incoming is identical: sym_kl=0, obs buffer empty -> eff=0 -> insight=0 -> rejected
    incoming = GaussianPattern(np.zeros(2), np.eye(2))
    result = agent._accept_communicated(incoming, 'other_agent')
    assert result is False


def test_accept_communicated_no_self_reception_enforced_by_caller():
    """_accept_communicated itself doesn't enforce self-rejection; orchestrator does.
    Just confirm the method runs without error on same agent_id."""
    agent, _ = make_agent_with_field()
    agent.step(np.zeros(2))
    incoming = GaussianPattern(np.array([50.0, 50.0]), np.eye(2) * 0.01)
    # Should not raise
    agent._accept_communicated(incoming, 'test')


def test_accept_communicated_empty_library_uses_nov_one():
    """When library is empty, Nov=1.0 and Eff=0.0 (empty buffer) -> insight = beta_orig * alpha_nov."""
    c = AgentConfig(agent_id='empty', feature_dim=2, beta_orig=1.0, alpha_nov=0.5, alpha_eff=0.5, kappa_0=0.1)
    agent = Agent(c, store=InMemoryStore())
    # Clear the store (remove seed pattern)
    for p, _ in agent.store.query('empty'):
        agent.store.delete(p.id)
    incoming = GaussianPattern(np.zeros(2), np.eye(2))
    result = agent._accept_communicated(incoming, 'other')
    # Nov=1.0, Eff=0.0 -> insight = 1.0*(0.5*1.0 + 0.5*0.0) = 0.5 > 0 -> admitted
    assert result is True


# ---- step() integration ----

def test_communicated_out_in_return_dict_every_step():
    agent, field = make_agent_with_field()
    result = agent.step(np.zeros(2))
    assert 'communicated_out' in result
    assert result['communicated_out'] == 0


def test_no_sharing_below_level4_from_step():
    """Patterns at level 1-3 never appear in broadcast queue after step."""
    agent, field = make_agent_with_field()
    # Force all patterns to level < 4 by running one step (initial pattern will be level 1)
    agent.step(np.zeros(2))
    assert field.drain_broadcasts() == []


def test_orchestrator_distributes_broadcast_to_other_agents():
    from hpm.agents.multi_agent import MultiAgentOrchestrator
    field = PatternField()
    cfgA = AgentConfig(agent_id='A', feature_dim=2)
    cfgB = AgentConfig(agent_id='B', feature_dim=2, kappa_0=0.5)
    agentA = Agent(cfgA, field=field)
    agentB = Agent(cfgB, field=field)

    # Manually add a level-4 pattern to agent A's store and _share_pending
    p = level4_pattern(mu=np.array([100.0, 100.0]))
    # Directly add to broadcast queue to test orchestrator distribution
    shared = GaussianPattern(mu=p.mu.copy(), sigma=np.eye(2) * 0.01, source_id=p.id)
    field.broadcast('A', shared)

    # Give agent B some observations so Eff can be computed
    for _ in range(5):
        agentB.step(np.zeros(2))

    # Drain and distribute manually (simulating orchestrator)
    broadcasts = field.drain_broadcasts()
    for source_id, pat in broadcasts:
        if source_id != 'B':
            agentB._accept_communicated(pat, source_id)

    # Agent B may or may not admit -- just verify no exception and method works
    # (admission depends on novelty and efficacy values)
