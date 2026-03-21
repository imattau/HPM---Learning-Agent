import numpy as np
from hpm.patterns.gaussian import GaussianPattern
from hpm.dynamics.meta_pattern_rule import MetaPatternRule, StepResult


def make_pattern(mu_val, dim=2):
    return GaussianPattern(mu=np.full(dim, float(mu_val)), sigma=np.eye(dim))


def test_step_returns_step_result():
    rule = MetaPatternRule()
    p = make_pattern(0.0)
    result = rule.step([p], np.array([1.0]), np.array([0.5]))
    assert isinstance(result, StepResult)
    assert hasattr(result, 'weights')
    assert hasattr(result, 'total_conflict')


def test_step_result_weights_normalised():
    rule = MetaPatternRule()
    p = make_pattern(0.0)
    result = rule.step([p], np.array([1.0]), np.array([0.5]))
    assert abs(result.weights.sum() - 1.0) < 1e-9


def test_total_conflict_zero_for_single_pattern():
    rule = MetaPatternRule(beta_c=0.1)
    p = make_pattern(0.0)
    result = rule.step([p], np.array([1.0]), np.array([0.5]))
    assert result.total_conflict == 0.0


def test_total_conflict_positive_for_incompatible_patterns():
    rule = MetaPatternRule(beta_c=0.1)
    p1 = make_pattern(0.0)
    p2 = make_pattern(100.0)
    result = rule.step([p1, p2], np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert result.total_conflict > 0.0


def test_total_conflict_zero_for_empty_patterns():
    rule = MetaPatternRule()
    result = rule.step([], np.array([]), np.array([]))
    assert isinstance(result, StepResult)
    assert result.total_conflict == 0.0
