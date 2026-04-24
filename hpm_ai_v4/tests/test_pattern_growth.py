import pytest
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern

@pytest.fixture
def pattern():
    np.random.seed(42)
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    return p

def test_latent_dim_incremented(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.latent_dim == old_K + 1

def test_A3_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.A3.shape == (old_K + 1, old_K + 1)

def test_A3_rows_sum_to_one(pattern):
    pattern.grow_latent()
    row_sums = pattern.A3.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(pattern.latent_dim), atol=1e-6)

def test_A32_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.A32.shape == (old_K + 1, old_K + 1)

def test_A32_rows_sum_to_one(pattern):
    pattern.grow_latent()
    row_sums = pattern.A32.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(pattern.latent_dim), atol=1e-6)

def test_A21_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.A21.shape == (old_K + 1, old_K + 1)

def test_B_shape(pattern):
    old_K = pattern.latent_dim
    obs_dim = pattern.obs_dim
    pattern.grow_latent()
    assert pattern.B.shape == (old_K + 1, obs_dim)

def test_B_rows_sum_to_one(pattern):
    pattern.grow_latent()
    row_sums = pattern.B.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(pattern.latent_dim), atol=1e-6)

def test_pi3_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.pi3.shape == (old_K + 1,)

def test_pi3_sums_to_one(pattern):
    pattern.grow_latent()
    assert abs(pattern.pi3.sum() - 1.0) < 1e-6

def test_old_A3_values_preserved(pattern):
    old_A3 = pattern.A3.copy()
    pattern.grow_latent(noise_scale=0.0)
    # Old block should be preserved (up to re-normalisation)
    K = old_A3.shape[0]
    # Check ratios are consistent (rows of old block still normalise same way)
    for i in range(K):
        old_row = old_A3[i]
        # After expansion with noise_scale=0, new col K is 0 so old rows
        # re-normalise to same relative values
        new_row = pattern.A3[i, :K]
        old_norm = old_row / (old_row.sum() + 1e-12)
        new_norm = new_row / (new_row.sum() + 1e-12)
        np.testing.assert_allclose(old_norm, new_norm, atol=1e-5)

def test_log_likelihood_finite_after_growth(pattern):
    obs_seq = [0, 1, 2, 3, 4, 0, 1]
    pattern.grow_latent()
    ll = pattern.log_likelihood(obs_seq)
    assert np.isfinite(ll)

def test_ss_shapes_updated(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    K1 = old_K + 1
    assert pattern.SS_A3.shape == (K1, K1)
    assert pattern.SS_B.shape == (K1, pattern.obs_dim)

def test_grow_twice(pattern):
    pattern.grow_latent()
    pattern.grow_latent()
    assert pattern.latent_dim == 4
    assert pattern.A3.shape == (4, 4)
