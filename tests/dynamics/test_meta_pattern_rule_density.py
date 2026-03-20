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
