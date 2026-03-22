"""Unit tests for PoissonPattern."""
import numpy as np
import pytest
from hpm.patterns.poisson import PoissonPattern
from hpm.patterns.gaussian import GaussianPattern


def make_poisson(D=4, lam=1.0):
    return PoissonPattern(np.full(D, lam))


# --- Construction ---

def test_construction_valid():
    p = make_poisson(D=4)
    assert p.D == 4
    assert p.lam.shape == (4,)
    assert p._n_obs == 1


def test_construction_floors_lam():
    p = PoissonPattern(np.array([0.0, 1.0, -1.0]))
    assert np.all(p.lam >= 1e-8)


def test_construction_invalid_shape():
    with pytest.raises((ValueError, Exception)):
        PoissonPattern(np.ones((3, 2)))  # 2-D not allowed


# --- log_prob ---

def test_log_prob_finite_at_construction():
    p = make_poisson(D=4)
    x = np.zeros(4)
    assert np.isfinite(p.log_prob(x))


def test_log_prob_lower_for_probable_values():
    # Poisson(5): k=5 is most probable
    p = PoissonPattern(np.full(3, 5.0))
    x_likely = np.full(3, 5)
    x_unlikely = np.full(3, 50)
    assert p.log_prob(x_likely) < p.log_prob(x_unlikely)


def test_log_prob_nll_positive():
    p = make_poisson(D=4, lam=2.0)
    x = np.array([2, 2, 2, 2])
    # NLL should be positive
    assert p.log_prob(x) >= 0.0


# --- update ---

def test_update_increments_n_obs():
    p = make_poisson(D=4)
    x = np.array([3, 3, 3, 3])
    p2 = p.update(x)
    assert p2._n_obs == p._n_obs + 1


def test_update_returns_new_instance():
    p = make_poisson(D=4)
    p2 = p.update(np.zeros(4))
    assert p2 is not p


def test_update_preserves_id():
    p = make_poisson(D=4)
    p2 = p.update(np.zeros(4))
    assert p2.id == p.id


def test_update_converges_to_true_rate():
    # After many updates at k=7, lambda should converge toward 7
    p = make_poisson(D=2)
    for _ in range(500):
        p = p.update(np.array([7, 7]))
    assert np.all(np.abs(p.lam - 7.0) < 0.5)


def test_update_nll_decreases_on_repeated_obs():
    p = make_poisson(D=3)
    x = np.array([4, 4, 4])
    nll_before = p.log_prob(x)
    for _ in range(300):
        p = p.update(x)
    nll_after = p.log_prob(x)
    assert nll_after < nll_before


# --- recombine ---

def test_recombine_weighted_average():
    a = PoissonPattern(np.full(3, 2.0))
    b = PoissonPattern(np.full(3, 8.0))
    c = a.recombine(b)
    # Equal _n_obs → midpoint
    np.testing.assert_allclose(c.lam, np.full(3, 5.0), atol=1e-6)


def test_recombine_type_error():
    p = make_poisson(D=3)
    g = GaussianPattern(np.zeros(3), np.eye(3))
    with pytest.raises(TypeError):
        p.recombine(g)


def test_recombine_dim_mismatch():
    a = make_poisson(D=3)
    b = make_poisson(D=5)
    with pytest.raises(ValueError):
        a.recombine(b)


def test_recombine_zero_n_obs_fallback():
    a = PoissonPattern(np.full(3, 2.0))
    b = PoissonPattern(np.full(3, 4.0))
    a._n_obs = 0
    b._n_obs = 0
    c = a.recombine(b)
    np.testing.assert_allclose(c.lam, np.full(3, 3.0), atol=1e-6)


# --- sample ---

def test_sample_shape():
    p = make_poisson(D=5)
    rng = np.random.default_rng(0)
    s = p.sample(10, rng)
    assert s.shape == (10, 5)


def test_sample_non_negative():
    p = make_poisson(D=4, lam=3.0)
    rng = np.random.default_rng(0)
    s = p.sample(100, rng)
    assert np.all(s >= 0)


def test_no_sigma_attribute():
    p = make_poisson(D=3)
    assert not hasattr(p, 'sigma')


# --- structural methods ---

def test_description_length_returns_float():
    p = make_poisson(D=4)
    assert isinstance(p.description_length(), float)


def test_description_length_zero_at_lambda_one():
    # lambda=1 is the uninformative prior — no learned deviation
    p = make_poisson(D=4, lam=1.0)
    assert p.description_length() == 0.0


def test_description_length_nonzero_at_extreme_lambda():
    p = PoissonPattern(np.array([0.1, 10.0, 1.0, 1.0]))
    assert p.description_length() >= 2.0


def test_connectivity_zero():
    assert make_poisson(D=4).connectivity() == 0.0


def test_compress_one_for_uniform():
    p = make_poisson(D=4, lam=3.0)
    assert abs(p.compress() - 1.0) < 1e-6


def test_is_structurally_valid_fresh():
    assert make_poisson(D=4).is_structurally_valid()


def test_is_structurally_valid_false_if_zero():
    p = make_poisson(D=3)
    p.lam[0] = 0.0
    assert not p.is_structurally_valid()


# --- serialisation ---

def test_round_trip():
    p = make_poisson(D=4, lam=2.0)
    p = p.update(np.array([3, 5, 2, 4]))
    d = p.to_dict()
    p2 = PoissonPattern.from_dict(d)
    assert d['type'] == 'poisson'
    np.testing.assert_allclose(p2.lam, p.lam)
    assert p2.id == p.id
    assert p2.level == p.level
    assert p2._n_obs == p._n_obs
