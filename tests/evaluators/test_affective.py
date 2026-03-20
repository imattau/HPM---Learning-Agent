import numpy as np
import pytest
from hpm.evaluators.affective import AffectiveEvaluator
from hpm.patterns.gaussian import GaussianPattern


def test_e_aff_non_negative(dim):
    """Affective evaluator output is always non-negative."""
    aff = AffectiveEvaluator(k=1.0, c_opt=5.0, sigma_c=3.0)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    score = aff.update(pattern, current_accuracy=-2.0)
    assert score >= 0.0


def test_inverted_u_over_complexity():
    """E_aff peaks at intermediate complexity, lower at extremes."""
    # Use c_opt=15 so that complexity 15 is optimal
    aff = AffectiveEvaluator(k=1.0, c_opt=15.0, sigma_c=5.0)

    # Create patterns with specific complexity levels
    # description_length = n for identity covariance with zero mu
    low_c = GaussianPattern(mu=np.zeros(5), sigma=np.eye(5))     # complexity = 5
    mid_c = GaussianPattern(mu=np.zeros(15), sigma=np.eye(15))   # complexity = 15 (optimal)
    high_c = GaussianPattern(mu=np.zeros(25), sigma=np.eye(25))  # complexity = 25

    # All have same improvement: delta_A = 0 (first call, no previous)
    # so they differ only by g(c)
    score_low = aff.update(low_c, current_accuracy=-1.0)
    aff2 = AffectiveEvaluator(k=1.0, c_opt=15.0, sigma_c=5.0)
    score_mid = aff2.update(mid_c, current_accuracy=-1.0)
    aff3 = AffectiveEvaluator(k=1.0, c_opt=15.0, sigma_c=5.0)
    score_high = aff3.update(high_c, current_accuracy=-1.0)

    # With same improvement, g(c) dominates: mid (c=15) peaks
    # g(5) ≈ 0.135, g(15) = 1.0, g(25) ≈ 0.135
    assert score_mid >= score_low
    assert score_mid >= score_high


def test_external_reward_added(dim):
    """External reward (alpha_r * r_t) adds to E_aff when alpha_r > 0."""
    aff_no_reward = AffectiveEvaluator(k=1.0, c_opt=5.0, sigma_c=3.0, alpha_r=0.0)
    aff_reward = AffectiveEvaluator(k=1.0, c_opt=5.0, sigma_c=3.0, alpha_r=1.0)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    s1 = aff_no_reward.update(pattern, current_accuracy=-1.0, reward=2.0)
    s2 = aff_reward.update(pattern, current_accuracy=-1.0, reward=2.0)
    assert s2 > s1


def test_last_capacity_returns_zero_for_unknown_pattern_id():
    """ID never passed to update() returns 0.0 (dict default)."""
    evaluator = AffectiveEvaluator()
    assert evaluator.last_capacity("never_seen") == 0.0


def test_last_capacity_reflects_current_step():
    """After one update(), last_capacity() returns the capacity from that step."""
    evaluator = AffectiveEvaluator(k=1.0)
    pattern = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    evaluator.update(pattern, current_accuracy=1.0)
    cap = evaluator.last_capacity(pattern.id)
    # First update: delta_A=0 -> novelty=sigmoid(0)=0.5 -> capacity=0.5
    assert abs(cap - 0.5) < 1e-9


def test_capacity_is_one_minus_novelty():
    """last_capacity() == 1 - novelty for a given step."""
    evaluator = AffectiveEvaluator(k=2.0)
    pattern = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    # First step
    evaluator.update(pattern, current_accuracy=5.0)
    # Second step with different accuracy to produce non-zero delta_A
    evaluator.update(pattern, current_accuracy=3.0)
    # delta_A = 3.0 - 5.0 = -2.0, novelty = sigmoid(k * delta_A) = sigmoid(2.0 * -2.0) = sigmoid(-4)
    expected_novelty = 1.0 / (1.0 + np.exp(4.0))
    expected_capacity = 1.0 - expected_novelty
    assert abs(evaluator.last_capacity(pattern.id) - expected_capacity) < 1e-9
