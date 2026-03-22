"""Unit tests for BetaPattern."""
import numpy as np
import pytest
from hpm.patterns.beta import BetaPattern
from hpm.patterns.gaussian import GaussianPattern


def make_beta(D=4, alpha=1.0, beta=1.0):
    params = np.full((D, 2), [alpha, beta])
    return BetaPattern(params)


# --- Construction ---

def test_construction_valid():
    p = make_beta(D=4)
    assert p.D == 4
    assert p.params.shape == (4, 2)
    assert p._n_obs == 2


def test_construction_floors_params():
    params = np.array([[0.0, 1.0], [1.0, 0.0]])
    p = BetaPattern(params)
    assert np.all(p.params >= 1e-8)


def test_construction_invalid_shape():
    with pytest.raises(ValueError):
        BetaPattern(np.ones(4))  # 1-D, not (D, 2)


# --- log_prob ---

def test_log_prob_finite_at_construction():
    p = make_beta(D=4)
    x = np.full(4, 0.5)
    assert np.isfinite(p.log_prob(x))


def test_log_prob_lower_for_probable_values():
    # Beta(5, 1) is skewed toward 1.0 — high x is more probable
    params = np.full((3, 2), [5.0, 1.0])
    p = BetaPattern(params)
    x_high = np.full(3, 0.9)
    x_low = np.full(3, 0.1)
    assert p.log_prob(x_high) < p.log_prob(x_low)


def test_log_prob_clips_boundary_values():
    p = make_beta(D=3)
    # Should not raise or return inf even at boundary
    x = np.array([0.0, 0.5, 1.0])
    nll = p.log_prob(x)
    assert np.isfinite(nll)


def test_log_prob_nll_sign_convention():
    # NLL must be positive (consistent with GaussianPattern)
    p = make_beta(D=4)
    x = np.full(4, 0.5)
    assert p.log_prob(x) >= 0.0


# --- update ---

def test_update_increments_n_obs():
    p = make_beta(D=4)
    x = np.full(4, 0.9)
    p2 = p.update(x)
    assert p2._n_obs == p._n_obs + 1


def test_update_returns_new_instance():
    p = make_beta(D=4)
    x = np.full(4, 0.5)
    p2 = p.update(x)
    assert p2 is not p


def test_update_preserves_id():
    p = make_beta(D=4)
    p2 = p.update(np.full(4, 0.5))
    assert p2.id == p.id


def test_update_shifts_params_toward_observed():
    # After many updates at x=0.9, alpha >> beta in each dimension
    p = make_beta(D=2)
    for _ in range(100):
        p = p.update(np.array([0.9, 0.9]))
    assert np.all(p.params[:, 0] > p.params[:, 1])


def test_update_nll_decreases_on_repeated_obs():
    p = make_beta(D=3)
    x = np.full(3, 0.8)
    nll_before = p.log_prob(x)
    for _ in range(200):
        p = p.update(x)
    nll_after = p.log_prob(x)
    assert nll_after < nll_before


# --- recombine ---

def test_recombine_weighted_average():
    params_a = np.full((3, 2), [5.0, 1.0])
    params_b = np.full((3, 2), [1.0, 5.0])
    a = BetaPattern(params_a)
    b = BetaPattern(params_b)
    c = a.recombine(b)
    # With equal _n_obs the result should be midpoint
    expected = (params_a + params_b) / 2.0
    np.testing.assert_allclose(c.params, expected, atol=1e-6)


def test_recombine_type_error():
    p = make_beta(D=3)
    g = GaussianPattern(np.zeros(3), np.eye(3))
    with pytest.raises(TypeError):
        p.recombine(g)


def test_recombine_dim_mismatch():
    a = make_beta(D=3)
    b = make_beta(D=5)
    with pytest.raises(ValueError):
        a.recombine(b)


def test_recombine_zero_n_obs_fallback():
    params = np.ones((3, 2))
    a = BetaPattern(params)
    b = BetaPattern(params)
    a._n_obs = 0
    b._n_obs = 0
    c = a.recombine(b)
    np.testing.assert_allclose(c.params, params, atol=1e-6)


# --- sample ---

def test_sample_shape():
    p = make_beta(D=5)
    rng = np.random.default_rng(0)
    s = p.sample(10, rng)
    assert s.shape == (10, 5)


def test_sample_values_in_unit_interval():
    p = make_beta(D=4)
    rng = np.random.default_rng(0)
    s = p.sample(100, rng)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_no_sigma_attribute():
    p = make_beta(D=3)
    assert not hasattr(p, 'sigma')


# --- structural methods ---

def test_description_length_returns_float():
    p = make_beta(D=4)
    assert isinstance(p.description_length(), float)


def test_description_length_increases_with_concentration():
    p_uninformed = make_beta(D=4, alpha=1.0, beta=1.0)
    p_concentrated = make_beta(D=4, alpha=10.0, beta=10.0)
    assert p_concentrated.description_length() >= p_uninformed.description_length()


def test_connectivity_zero():
    assert make_beta(D=4).connectivity() == 0.0


def test_compress_returns_one_for_uniform():
    # All dimensions identical → max/mean = 1.0
    p = make_beta(D=4, alpha=2.0, beta=2.0)
    assert abs(p.compress() - 1.0) < 1e-6


def test_is_structurally_valid_fresh():
    assert make_beta(D=4).is_structurally_valid()


def test_is_structurally_valid_false_if_zero():
    p = make_beta(D=3)
    p.params[0, 0] = 0.0
    assert not p.is_structurally_valid()


# --- serialisation ---

def test_round_trip():
    p = make_beta(D=4)
    p = p.update(np.array([0.2, 0.4, 0.6, 0.8]))
    d = p.to_dict()
    p2 = BetaPattern.from_dict(d)
    assert d['type'] == 'beta'
    np.testing.assert_allclose(p2.params, p.params)
    assert p2.id == p.id
    assert p2.level == p.level
    assert p2._n_obs == p._n_obs
