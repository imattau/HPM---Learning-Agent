import numpy as np
import pytest
from hpm.agents.l4_generative import L4GenerativeHead


def test_predict_returns_none_before_fit():
    head = L4GenerativeHead(feature_dim_in=10, feature_dim_out=12)
    result = head.predict(np.zeros(10))
    assert result is None


def test_no_fit_with_fewer_than_two_pairs():
    head = L4GenerativeHead(feature_dim_in=10, feature_dim_out=12)
    head.accumulate(np.ones(10), np.ones(12))
    head.fit()
    assert head.predict(np.zeros(10)) is None


def test_fit_and_predict_shape():
    head = L4GenerativeHead(feature_dim_in=10, feature_dim_out=12)
    rng = np.random.default_rng(0)
    for _ in range(4):
        head.accumulate(rng.standard_normal(10), rng.standard_normal(12))
    head.fit()
    pred = head.predict(rng.standard_normal(10))
    assert pred is not None
    assert pred.shape == (12,)


def test_fit_is_idempotent():
    head = L4GenerativeHead(feature_dim_in=10, feature_dim_out=12)
    rng = np.random.default_rng(1)
    for _ in range(3):
        head.accumulate(rng.standard_normal(10), rng.standard_normal(12))
    head.fit()
    pred1 = head.predict(np.ones(10))
    head.fit()
    pred2 = head.predict(np.ones(10))
    np.testing.assert_array_equal(pred1, pred2)


def test_reset_clears_state():
    head = L4GenerativeHead(feature_dim_in=10, feature_dim_out=12)
    rng = np.random.default_rng(2)
    for _ in range(3):
        head.accumulate(rng.standard_normal(10), rng.standard_normal(12))
    head.fit()
    head.reset()
    assert head.predict(np.zeros(10)) is None


def test_ridge_regression_exact_solution():
    """With identity X, solution should be close to Y (regularised)."""
    head = L4GenerativeHead(feature_dim_in=2, feature_dim_out=2, alpha=1e-6)
    # X = I_2, Y = [[1,0],[0,1]] → W ≈ I_2
    head.accumulate(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    head.accumulate(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    head.fit()
    pred = head.predict(np.array([1.0, 0.0]))
    assert pred is not None
    np.testing.assert_allclose(pred, np.array([1.0, 0.0]), atol=1e-4)
