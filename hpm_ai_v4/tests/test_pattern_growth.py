import pytest
import numpy as np
import warnings
from hpm_ai_v4.pattern import HierarchicalPattern


@pytest.fixture
def pattern():
    np.random.seed(42)
    return HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)


def test_A_shape(pattern):
    assert pattern.A.shape == (2, 2)


def test_B_shape(pattern):
    assert pattern.B.shape == (2, 5)


def test_pi_shape(pattern):
    assert pattern.pi.shape == (2,)


def test_A_rows_sum_to_one(pattern):
    np.testing.assert_allclose(pattern.A.sum(axis=1), np.ones(2), atol=1e-6)


def test_B_rows_sum_to_one(pattern):
    np.testing.assert_allclose(pattern.B.sum(axis=1), np.ones(2), atol=1e-6)


def test_pi_sums_to_one(pattern):
    assert abs(pattern.pi.sum() - 1.0) < 1e-6


def test_A_dtype(pattern):
    assert pattern.A.dtype == np.float32


def test_log_likelihood_finite_after_init(pattern):
    ll = pattern.log_likelihood([0, 1, 2, 3, 4, 0, 1])
    assert np.isfinite(ll)


def test_get_top_state_returns_valid_int():
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    result = p.get_top_state([0, 1, 0, 2, 1])
    assert isinstance(result, int)
    assert result in {0, 1}


def test_get_top_state_empty_returns_int():
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    result = p.get_top_state([])
    assert isinstance(result, int)


def test_get_top_state_k3():
    p = HierarchicalPattern(pattern_id=0, latent_dim=3, obs_dim=5)
    result = p.get_top_state([0, 1, 2, 0, 1])
    assert result in {0, 1, 2}


def test_constructor_warns_if_k_gt_4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(pattern_id=0, latent_dim=5, obs_dim=5)
        assert len(w) == 1
        assert "latent_dim" in str(w[0].message).lower() or "K" in str(w[0].message)


def test_constructor_no_warn_k_eq_4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(pattern_id=0, latent_dim=4, obs_dim=5)
        assert len(w) == 0
