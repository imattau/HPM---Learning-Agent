import numpy as np
import pytest
from hpm.patterns.laplace import LaplacePattern
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def pattern():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(4)
    b = np.array([0.5, 1.0, 1.5, 2.0])
    return LaplacePattern(mu, b)


def test_log_prob_at_mu_is_minimum(pattern):
    """log_prob at mu should be lower (more probable) than at a distant point."""
    far = pattern.mu + 10.0
    assert pattern.log_prob(pattern.mu) < pattern.log_prob(far)


def test_log_prob_finite(pattern):
    x = np.ones(4)
    assert np.isfinite(pattern.log_prob(x))


def test_no_sigma_attribute(pattern):
    """Critical: absence of sigma routes MetaPatternRule to MC fallback."""
    assert not hasattr(pattern, 'sigma')


def test_update_increments_n_obs(pattern):
    x = np.ones(4)
    updated = pattern.update(x)
    assert updated._n_obs == pattern._n_obs + 1


def test_update_moves_mu_toward_observation(pattern):
    target = np.ones(4) * 5.0
    p = LaplacePattern(np.zeros(4), np.ones(4))
    for _ in range(100):
        p = p.update(target)
    assert np.allclose(p.mu, target, atol=0.1)


def test_b_updates_using_mu_old():
    """b converges toward |x - mu_old| after repeated identical observations."""
    p = LaplacePattern(np.zeros(2), np.ones(2) * 10.0)
    x = np.array([3.0, 3.0])
    for _ in range(500):
        p = p.update(x)
    assert np.all(p.b < 0.1)


def test_b_floor_prevents_zero():
    p = LaplacePattern(np.zeros(2), np.ones(2) * 10.0)
    x = np.zeros(2)
    for _ in range(1000):
        p = p.update(x)
    assert np.all(p.b >= 1e-6)


def test_b_floor_on_construction():
    p = LaplacePattern(np.zeros(2), np.array([-1.0, 0.0]))
    assert np.all(p.b >= 1e-6)


def test_sample_shape(pattern):
    rng = np.random.default_rng(42)
    samples = pattern.sample(50, rng)
    assert samples.shape == (50, 4)


def test_sample_mean_close_to_mu():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -2.0, 3.0])
    b = np.ones(3) * 0.1
    p = LaplacePattern(mu, b)
    samples = p.sample(2000, rng)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)


def test_connectivity_always_zero(pattern):
    assert pattern.connectivity() == 0.0


def test_compress_ratio(pattern):
    c = pattern.compress()
    assert 0.0 <= c <= 1.0


def test_is_structurally_valid(pattern):
    assert pattern.is_structurally_valid()


def test_is_structurally_invalid_when_b_zero():
    p = LaplacePattern.__new__(LaplacePattern)
    p.mu = np.zeros(3)
    p.b = np.array([0.0, 1.0, 1.0])
    assert not p.is_structurally_valid()


def test_recombine_averages_mu_and_b():
    p1 = LaplacePattern(np.zeros(3), np.ones(3))
    p2 = LaplacePattern(np.ones(3) * 2.0, np.ones(3) * 3.0)
    child = p1.recombine(p2)
    assert np.allclose(child.mu, np.ones(3))
    assert np.allclose(child.b, np.ones(3) * 2.0)


def test_recombine_with_gaussian_raises():
    p = LaplacePattern(np.zeros(3), np.ones(3))
    g = GaussianPattern(np.zeros(3), np.eye(3))
    with pytest.raises(TypeError):
        p.recombine(g)


def test_to_dict_type_field(pattern):
    d = pattern.to_dict()
    assert d['type'] == 'laplace'


def test_roundtrip_serialisation(pattern):
    d = pattern.to_dict()
    restored = LaplacePattern.from_dict(d)
    assert restored.id == pattern.id
    assert np.allclose(restored.mu, pattern.mu)
    assert np.allclose(restored.b, pattern.b)
    assert restored.level == pattern.level
    assert restored._n_obs == pattern._n_obs


def test_id_preserved_on_update(pattern):
    updated = pattern.update(np.ones(4))
    assert updated.id == pattern.id


def test_description_length_positive(pattern):
    assert pattern.description_length() > 0
