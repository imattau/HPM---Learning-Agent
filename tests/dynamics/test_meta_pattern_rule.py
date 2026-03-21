import numpy as np
import pytest
from hpm.dynamics.meta_pattern_rule import MetaPatternRule, sym_kl_normalised
from hpm.patterns.gaussian import GaussianPattern


def test_weights_sum_to_one_after_step(dim):
    """Weights remain normalised after dynamics step."""
    mpr = MetaPatternRule(eta=0.1, beta_c=0.05, epsilon=1e-4)
    patterns = [
        GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim)),
        GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim)),
    ]
    weights = np.array([0.6, 0.4])
    totals = np.array([-1.0, -0.5])   # second pattern has higher total score
    new_w = mpr.step(patterns, weights, totals).weights
    assert new_w.sum() == pytest.approx(1.0, abs=1e-6)


def test_higher_total_gains_weight(dim):
    """Pattern with above-average total score gains weight."""
    mpr = MetaPatternRule(eta=0.1, beta_c=0.0, epsilon=1e-4)  # no conflict
    patterns = [
        GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim)),
        GaussianPattern(mu=np.ones(dim) * 5, sigma=np.eye(dim)),
    ]
    weights = np.array([0.5, 0.5])
    totals = np.array([-2.0, -0.5])   # pattern 1 has higher score
    new_w = mpr.step(patterns, weights, totals).weights
    assert new_w[1] > new_w[0]


def test_floor_prevents_empty_library(dim):
    """If all weights collapse, best pattern retained at weight 1.0."""
    mpr = MetaPatternRule(eta=100.0, beta_c=100.0, epsilon=0.1)
    patterns = [GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))]
    weights = np.array([1.0])
    totals = np.array([-999.0])
    new_w = mpr.step(patterns, weights, totals).weights
    assert len(new_w) == 1
    assert new_w[0] == pytest.approx(1.0)


def test_conflict_excludes_self_inhibition(dim):
    """D5 sum excludes j=i: a single pattern should not self-inhibit."""
    mpr = MetaPatternRule(eta=0.0, beta_c=1.0, epsilon=1e-4)   # only conflict term
    patterns = [GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))]
    weights = np.array([1.0])
    totals = np.array([-1.0])
    new_w = mpr.step(patterns, weights, totals).weights
    # Single pattern: no j != i pairs, so no conflict inhibition
    assert new_w[0] == pytest.approx(1.0, abs=1e-6)


def test_sym_kl_same_pattern_is_zero(dim):
    """Symmetric KL of identical patterns should be near zero."""
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    kl = sym_kl_normalised(p, p)
    assert kl == pytest.approx(0.0, abs=0.05)


def test_sym_kl_different_patterns_positive(dim):
    """Distant patterns have positive incompatibility."""
    p1 = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    p2 = GaussianPattern(mu=np.ones(dim) * 10, sigma=np.eye(dim))
    kl = sym_kl_normalised(p1, p2)
    assert kl > 0.0
    assert kl <= 1.0
