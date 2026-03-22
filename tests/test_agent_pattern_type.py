import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.store.memory import InMemoryStore
from hpm.patterns.laplace import LaplacePattern
from hpm.patterns.gaussian import GaussianPattern


def _make_agent(pattern_type="gaussian"):
    cfg = AgentConfig(agent_id="test", feature_dim=4, pattern_type=pattern_type)
    return Agent(cfg, store=InMemoryStore(), field=None)


def test_agent_creates_gaussian_by_default():
    agent = _make_agent("gaussian")
    records = agent.store.query("test")
    assert len(records) >= 1
    p, _ = records[0]
    assert isinstance(p, GaussianPattern)


def test_agent_creates_laplace_pattern():
    agent = _make_agent("laplace")
    records = agent.store.query("test")
    assert len(records) >= 1
    p, _ = records[0]
    assert isinstance(p, LaplacePattern)


def test_agent_step_with_laplace_no_crash():
    agent = _make_agent("laplace")
    result = agent.step(np.ones(4))
    assert result.get("n_patterns", 0) >= 1
