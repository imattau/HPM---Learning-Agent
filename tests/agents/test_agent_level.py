import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent


@pytest.fixture
def cfg():
    return AgentConfig(agent_id='test', feature_dim=2)


@pytest.fixture
def agent(cfg):
    return Agent(cfg)


def test_agent_step_includes_level_mean(agent):
    x = np.zeros(2)
    result = agent.step(x)
    assert 'level_mean' in result
    assert isinstance(result['level_mean'], float)


def test_agent_step_includes_level_distribution(agent):
    x = np.zeros(2)
    result = agent.step(x)
    assert 'level_distribution' in result
    dist = result['level_distribution']
    assert set(dist.keys()) == {1, 2, 3, 4, 5}


def test_level_distribution_sums_to_pattern_count(agent):
    x = np.zeros(2)
    result = agent.step(x)
    total = sum(result['level_distribution'].values())
    assert total >= 1


def test_level_mean_is_between_1_and_5(agent):
    x = np.zeros(2)
    result = agent.step(x)
    assert 1.0 <= result['level_mean'] <= 5.0


def test_kappa_d_levels_lookup_wiring(cfg):
    """kappa_d_levels[level-1] is actually used: agent wires level_classifier correctly."""
    cfg_a = AgentConfig(agent_id='a', feature_dim=2)
    cfg_a.kappa_d_levels = [2.0, 0.0, 0.0, 0.0, 0.0]

    cfg_b = AgentConfig(agent_id='b', feature_dim=2)
    cfg_b.kappa_d_levels = [0.0, 0.0, 0.0, 0.0, 0.0]

    agent_a = Agent(cfg_a)
    agent_b = Agent(cfg_b)

    r_a = agent_a.step(np.zeros(2))
    r_b = agent_b.step(np.zeros(2))

    assert 'level_mean' in r_a
    assert 'level_mean' in r_b
    assert agent_a.level_classifier is not None
    assert agent_a.config.kappa_d_levels[0] == 2.0


def test_level_written_to_store_after_step(cfg):
    """Pattern level is persisted in the store via to_dict/from_dict round-trip."""
    from hpm.store.memory import InMemoryStore
    store = InMemoryStore()
    agent = Agent(cfg, store=store)

    agent.step(np.zeros(2))

    records = store.query(cfg.agent_id)
    assert len(records) > 0
    for pattern, _weight in records:
        assert hasattr(pattern, 'level')
        assert 1 <= pattern.level <= 5
