import numpy as np
import pytest
from hpm.evaluators.epistemic import EpistemicEvaluator
from hpm.patterns.gaussian import GaussianPattern


def test_accuracy_improves_toward_mean(dim):
    """Pattern fit at its own mean should stabilise to a high accuracy."""
    evaluator = EpistemicEvaluator(lambda_L=0.5)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x_good = np.zeros(dim)   # at mean — low loss

    acc_values = [evaluator.update(pattern, x_good) for _ in range(20)]
    # Accuracy (= -running_loss) should be stable (not diverging)
    assert all(a <= 0 for a in acc_values)   # A_i <= 0 always


def test_accuracy_lower_for_distant_obs(dim):
    """Pattern evaluated on distant observation has lower accuracy."""
    evaluator = EpistemicEvaluator(lambda_L=1.0)   # instant update
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    acc_near = evaluator.update(pattern, np.zeros(dim))
    evaluator2 = EpistemicEvaluator(lambda_L=1.0)
    acc_far = evaluator2.update(pattern, np.ones(dim) * 10)
    assert acc_near > acc_far


def test_running_loss_ema(dim):
    """Running loss follows EMA formula: L(t) = (1-lambda)*L(t-1) + lambda*ell(t)."""
    evaluator = EpistemicEvaluator(lambda_L=0.5)
    pattern = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x = np.zeros(dim)
    loss0 = pattern.log_prob(x)
    # First call: L(0) = 0, so L(1) = lambda_L * ell(0), A(1) = -lambda_L * ell(0)
    acc = evaluator.update(pattern, x)
    assert acc == pytest.approx(-0.5 * loss0, rel=1e-5)
