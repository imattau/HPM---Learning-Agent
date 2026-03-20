import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig


@pytest.fixture
def config():
    return AgentConfig(agent_id="test_agent", feature_dim=4, eta=0.05, lambda_L=0.2)


@pytest.fixture
def agent(config):
    return Agent(config)


def test_agent_initialises_with_one_pattern(agent):
    from hpm.store.memory import InMemoryStore
    records = agent.store.query("test_agent")
    assert len(records) == 1


def test_step_returns_metrics(agent):
    x = np.zeros(4)
    result = agent.step(x)
    assert 't' in result
    assert 'n_patterns' in result
    assert 'mean_accuracy' in result
    assert result['t'] == 1


def test_accuracy_generally_improves(agent):
    """After many steps on consistent observations, accuracy should improve."""
    x = np.zeros(4)
    results = [agent.step(x) for _ in range(50)]
    early = np.mean([r['mean_accuracy'] for r in results[:10]])
    late = np.mean([r['mean_accuracy'] for r in results[40:]])
    assert late >= early   # accuracy improves (less negative) over time


def test_patterns_are_updated_after_step(agent):
    x = np.ones(4)
    initial_records = agent.store.query("test_agent")
    initial_mu = initial_records[0][0].mu.copy()
    agent.step(x)
    updated_records = agent.store.query("test_agent")
    updated_mu = updated_records[0][0].mu
    assert not np.allclose(initial_mu, updated_mu)
