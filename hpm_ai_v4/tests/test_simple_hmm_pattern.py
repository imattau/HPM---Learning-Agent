import numpy as np
import pytest
from hpm_ai_v4.pattern import HierarchicalPattern


def test_init_shapes():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert p.A.shape == (2, 2)
    assert p.B.shape == (2, 6)
    assert p.pi.shape == (2,)
    assert p.A.dtype == np.float32


def test_rows_sum_to_one():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert np.allclose(p.A.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.B.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.pi.sum(), 1.0, atol=1e-5)


def test_log_likelihood_finite():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    ll = p.log_likelihood([0, 1, 2, 0, 1])
    assert np.isfinite(ll)
    assert ll < 0


def test_update_parameters_online():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    obs = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10
    p.update_parameters_online(obs, window_size=30)
    assert np.allclose(p.A.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.B.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p.pi.sum(), 1.0, atol=1e-5)


def test_get_top_state():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    state = p.get_top_state([0, 1, 0, 1])
    assert state in {0, 1}


def test_get_top_state_empty():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    state = p.get_top_state([])
    assert isinstance(state, int)
    assert 0 <= state < 2


def test_compression_nonneg():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert p.compression() >= 0


def test_compression_no_obs_arg():
    # compression() takes no obs_seq argument in new interface
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    c = p.compression()
    assert isinstance(c, float)


def test_predict_next():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    pred = p.predict_next([0, 1, 2])
    assert 0 <= pred < 6


def test_predict_next_distribution_shape():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    dist = p.predict_next_distribution([0, 1, 2])
    assert dist.shape == (6,)
    assert np.allclose(dist.sum(), 1.0, atol=1e-5)


def test_log_cache_refreshed():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert np.allclose(np.exp(p.logA), p.A, atol=1e-5)
    assert np.allclose(np.exp(p.logB), p.B, atol=1e-5)


def test_update_running_loss():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    p.update_running_loss([0, 1, 2, 0, 1])
    assert np.isfinite(p.running_loss)
    assert p.running_loss > 0


def test_warns_if_k_gt_4():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(0, latent_dim=5, obs_dim=6)
        assert len(w) == 1


def test_no_warn_k_eq_4():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(0, latent_dim=4, obs_dim=6)
        assert len(w) == 0


def test_log_likelihood_empty():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    assert p.log_likelihood([]) == 0.0


def test_predictive_entropy_finite():
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    h = p.predictive_entropy([0, 1, 2])
    assert np.isfinite(h)
    assert h >= 0


def test_viterbi_path_returns_latent_length_sequence():
    p = HierarchicalPattern(1, latent_dim=3, obs_dim=6)
    path = p.viterbi_path([0, 1, 2, 1, 0])
    assert isinstance(path, list)
    assert len(path) == 5
    assert all(0 <= state < 3 for state in path)
