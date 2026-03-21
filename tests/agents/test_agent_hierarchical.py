import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig


def _make_agent(beta_comp=0.0):
    cfg = AgentConfig(agent_id='test', feature_dim=4, beta_comp=beta_comp)
    return Agent(cfg)


def test_beta_comp_zero_compress_mean_is_zero():
    """With beta_comp=0, compress_mean must be 0.0 (compress() is not called)."""
    agent = _make_agent(beta_comp=0.0)
    x = np.random.default_rng(0).standard_normal(4)
    result = agent.step(x)
    assert 'compress_mean' in result
    assert result['compress_mean'] == 0.0


def test_beta_comp_nonzero_compress_mean_in_range():
    """With beta_comp>0, compress_mean should be in (0, 1]."""
    agent = _make_agent(beta_comp=1.0)
    x = np.random.default_rng(1).standard_normal(4)
    result = agent.step(x)
    assert 'compress_mean' in result
    assert 0.0 < result['compress_mean'] <= 1.0


def test_beta_comp_nonzero_affects_relative_weights():
    """Patterns with higher compress() should gain relative weight advantage when beta_comp>0."""
    rng = np.random.default_rng(42)
    agent_flat = _make_agent(beta_comp=0.0)
    agent_comp = _make_agent(beta_comp=1.0)
    for _ in range(50):
        x = rng.standard_normal(4)
        agent_flat.step(x)
        agent_comp.step(x)
    result_flat = agent_flat.step(rng.standard_normal(4))
    result_comp = agent_comp.step(rng.standard_normal(4))
    assert result_comp['compress_mean'] >= result_flat['compress_mean']


def test_beta_comp_config_defaults_to_zero():
    cfg = AgentConfig(agent_id='x', feature_dim=3)
    assert cfg.beta_comp == 0.0
