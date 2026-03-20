import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField


def _cfg(agent_id="test", gamma_soc=0.5, rho=1.0):
    return AgentConfig(agent_id=agent_id, feature_dim=4, gamma_soc=gamma_soc, rho=rho)


def test_agent_accepts_field_parameter():
    field = PatternField()
    agent = Agent(_cfg(), field=field)
    assert agent.field is field


def test_agent_registers_with_field_after_step():
    field = PatternField()
    agent = Agent(_cfg(agent_id="a1"), field=field)
    agent.step(np.zeros(4))
    assert "a1" in field._agent_patterns


def test_step_returns_e_soc_mean_in_metrics():
    field = PatternField()
    agent = Agent(_cfg(), field=field)
    result = agent.step(np.zeros(4))
    assert "e_soc_mean" in result


def test_gamma_soc_zero_gives_zero_social_signal():
    config = AgentConfig(agent_id="solo", feature_dim=4, gamma_soc=0.0)
    field = PatternField()
    agent = Agent(config, field=field)
    result = agent.step(np.zeros(4))
    assert result["e_soc_mean"] == pytest.approx(0.0)


def test_no_field_gives_zero_social_signal():
    config = AgentConfig(agent_id="solo", feature_dim=4, gamma_soc=1.0)
    agent = Agent(config)  # no field
    result = agent.step(np.zeros(4))
    assert result["e_soc_mean"] == pytest.approx(0.0)


def test_two_agents_shared_uuid_get_nonzero_cross_agent_freq():
    """
    Agents sharing a pattern UUID observe each other's weights in the field.
    Both agents are seeded with the same initial GaussianPattern (same UUID).
    After one agent steps, the other sees a non-trivial freq for the shared UUID.
    """
    from hpm.patterns.gaussian import GaussianPattern
    from hpm.store.memory import InMemoryStore

    field = PatternField()
    shared = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))

    store1 = InMemoryStore()
    store1.save(shared, 1.0, "a1")
    store2 = InMemoryStore()
    store2.save(shared, 1.0, "a2")

    cfg1 = AgentConfig(agent_id="a1", feature_dim=4, gamma_soc=1.0, rho=1.0)
    cfg2 = AgentConfig(agent_id="a2", feature_dim=4, gamma_soc=1.0, rho=1.0)

    agent1 = Agent(cfg1, store=store1, field=field)
    agent2 = Agent(cfg2, store=store2, field=field)

    x = np.zeros(4)
    agent1.step(x)
    result2 = agent2.step(x)
    assert result2["e_soc_mean"] > 0.0


def test_backward_compat_no_field_no_gamma_soc():
    """Existing single-agent usage with defaults unchanged."""
    config = AgentConfig(agent_id="compat", feature_dim=4)
    agent = Agent(config)
    result = agent.step(np.ones(4) * 0.5)
    assert "t" in result
    assert "mean_accuracy" in result
    assert result["e_soc_mean"] == pytest.approx(0.0)
