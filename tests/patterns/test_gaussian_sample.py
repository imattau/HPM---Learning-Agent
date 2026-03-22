import numpy as np
from hpm.patterns.gaussian import GaussianPattern


def test_sample_shape():
    rng = np.random.default_rng(0)
    p = GaussianPattern(np.zeros(3), np.eye(3))
    samples = p.sample(50, rng)
    assert samples.shape == (50, 3)


def test_sample_mean_close_to_mu():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -2.0, 3.0])
    p = GaussianPattern(mu, np.eye(3) * 0.01)
    samples = p.sample(2000, rng)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
