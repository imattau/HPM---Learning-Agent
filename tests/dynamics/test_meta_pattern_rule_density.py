import numpy as np
import pytest
from hpm.dynamics.meta_pattern_rule import MetaPatternRule
from hpm.patterns.gaussian import GaussianPattern


def _two_patterns():
    p1 = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    p2 = GaussianPattern(mu=np.ones(2) * 3.0, sigma=np.eye(2))
    return [p1, p2]


def test_kappa_d_zero_unchanged_from_baseline():
    """kappa_D=0 with densities=[0.9, 0.1] produces identical output to no-densities call."""
    patterns = _two_patterns()
    weights = np.array([0.6, 0.4])
    totals = np.array([1.0, 0.5])

    rule_baseline = MetaPatternRule(eta=0.1, beta_c=0.1, epsilon=1e-4, kappa_D=0.0)
    rule_with = MetaPatternRule(eta=0.1, beta_c=0.1, epsilon=1e-4, kappa_D=0.0)

    rule_baseline._rng = np.random.default_rng(42)
    rule_with._rng = np.random.default_rng(42)

    w_baseline = rule_baseline.step(patterns, weights.copy(), totals)
    w_with = rule_with.step(patterns, weights.copy(), totals, densities=[0.9, 0.1])

    np.testing.assert_allclose(w_baseline, w_with, atol=1e-12)


def test_density_bias_increases_high_density_pattern_weight():
    """High-density pattern gains relatively more weight than low-density under kappa_D > 0."""
    patterns = _two_patterns()
    weights = np.array([0.5, 0.5])
    totals = np.array([1.0, 1.0])
    densities = [0.9, 0.1]

    rule = MetaPatternRule(eta=0.1, beta_c=0.0, epsilon=1e-4, kappa_D=0.5)
    rule._rng = np.random.default_rng(0)
    new_w = rule.step(patterns, weights.copy(), totals, densities=densities)

    assert new_w[0] > new_w[1]


def test_renormalisation_holds_after_density_bias():
    """Weights sum to 1.0 after step with density bias."""
    patterns = _two_patterns()
    weights = np.array([0.5, 0.5])
    totals = np.array([1.0, 0.8])
    densities = [0.7, 0.3]

    rule = MetaPatternRule(eta=0.05, beta_c=0.1, epsilon=1e-4, kappa_D=0.2)
    rule._rng = np.random.default_rng(1)
    new_w = rule.step(patterns, weights.copy(), totals, densities=densities)

    assert abs(new_w.sum() - 1.0) < 1e-9


def test_densities_none_behaves_identically():
    """Calling step() with densities=None is identical to omitting the argument."""
    patterns = _two_patterns()
    weights = np.array([0.6, 0.4])
    totals = np.array([1.0, 0.5])

    rule1 = MetaPatternRule(eta=0.1, kappa_D=0.3)
    rule2 = MetaPatternRule(eta=0.1, kappa_D=0.3)
    rule1._rng = np.random.default_rng(5)
    rule2._rng = np.random.default_rng(5)

    w1 = rule1.step(patterns, weights.copy(), totals)
    w2 = rule2.step(patterns, weights.copy(), totals, densities=None)

    np.testing.assert_allclose(w1, w2, atol=1e-12)


from hpm.config import AgentConfig
from hpm.agents.agent import Agent


def test_agent_step_includes_density_mean():
    """Agent with kappa_D > 0 returns density_mean in step dict."""
    config = AgentConfig(agent_id="test", feature_dim=4, kappa_D=0.1)
    agent = Agent(config)
    result = agent.step(np.zeros(4))
    assert "density_mean" in result
    assert isinstance(result["density_mean"], float)
    assert 0.0 <= result["density_mean"] <= 1.0


def test_agent_step_includes_density_mean_with_kappa_d_zero():
    """density_mean is present even when kappa_D=0 (default)."""
    config = AgentConfig(agent_id="test2", feature_dim=4)
    agent = Agent(config)
    result = agent.step(np.zeros(4))
    assert "density_mean" in result


def test_kappa_d_zero_density_mean_in_range():
    """Agent with kappa_D=0 (default) still computes and returns a valid density_mean."""
    config = AgentConfig(agent_id="c", feature_dim=4, kappa_D=0.0)
    agent = Agent(config)
    result = agent.step(np.zeros(4))
    assert "density_mean" in result
    assert 0.0 <= result["density_mean"] <= 1.0
    assert result["n_patterns"] >= 1
