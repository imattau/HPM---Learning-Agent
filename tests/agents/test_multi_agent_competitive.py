import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern


def make_agents(n, feature_dim=2):
    return [Agent(AgentConfig(agent_id=f'agent_{i}', feature_dim=feature_dim))
            for i in range(n)]


def test_groups_assign_separate_field_objects():
    agents = make_agents(4)
    groups = {'agent_0': 'A', 'agent_1': 'A', 'agent_2': 'B', 'agent_3': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)
    field_A = agents[0].field
    field_B = agents[2].field
    assert field_A is not field_B
    assert agents[0].field is agents[1].field
    assert agents[2].field is agents[3].field


def test_in_group_patterns_visible_cross_agent():
    """Shared UUID registered by agent_0 appears in agent_1 field freq (same group)."""
    agents = make_agents(2)
    groups = {'agent_0': 'A', 'agent_1': 'A'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)

    shared_p = GaussianPattern(np.zeros(2), np.eye(2))
    agents[0].field.register('agent_0', [(shared_p.id, 0.8)])
    freq = agents[1].field.freq(shared_p.id)
    assert freq > 0.0


def test_out_group_patterns_not_visible():
    """Agent in group B cannot see agent A's patterns via field."""
    agents = make_agents(2)
    groups = {'agent_0': 'A', 'agent_1': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)

    p = GaussianPattern(np.zeros(2), np.eye(2))
    agents[0].field.register('agent_0', [(p.id, 0.9)])
    freq = agents[1].field.freq(p.id)
    assert freq == 0.0


def test_group_field_quality_keyed_by_group():
    agents = make_agents(4)
    groups = {'agent_0': 'A', 'agent_1': 'A', 'agent_2': 'B', 'agent_3': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)
    quality = orch.group_field_quality()
    assert set(quality.keys()) == {'A', 'B'}
    for gid, q in quality.items():
        assert 'diversity' in q
        assert 'redundancy' in q


def test_backward_compat_no_groups():
    """groups=None: behaviour identical to before; group_field_quality() returns {}."""
    agents = make_agents(2)
    field = PatternField()
    orch = MultiAgentOrchestrator(agents, field, groups=None)
    # Both agents share the same field (the one passed in)
    assert agents[0].field is field
    assert agents[1].field is field
    assert orch.group_field_quality() == {}


def test_competitive_broadcast_within_group_only():
    """In competitive mode, broadcasts only reach within-group agents."""
    agents = make_agents(2)
    groups = {'agent_0': 'A', 'agent_1': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)

    # Manually broadcast on agent_0's group field
    p = GaussianPattern(np.array([100.0, 100.0]), np.eye(2) * 0.01)
    agents[0].field.broadcast('agent_0', p)

    # Drain agent_0's field -- agent_1 is in a different group so its field is separate
    queue_A = agents[0].field.drain_broadcasts()
    queue_B = agents[1].field.drain_broadcasts()
    assert len(queue_A) == 1
    assert len(queue_B) == 0
