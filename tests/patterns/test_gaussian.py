import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern


def test_log_prob_is_positive_loss(dim):
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x = np.zeros(dim)
    assert p.log_prob(x) >= 0


def test_log_prob_lower_at_mean(dim):
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    loss_near = p.log_prob(np.zeros(dim))
    loss_far = p.log_prob(np.ones(dim) * 10)
    assert loss_near < loss_far


def test_update_returns_new_instance(dim):
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    x = np.ones(dim)
    p2 = p.update(x)
    assert p2 is not p
    assert p2.id == p.id
    assert not np.allclose(p2.mu, p.mu)


def test_recombine_produces_new_id(dim):
    p1 = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    p2 = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    child = p1.recombine(p2)
    assert child.id != p1.id
    assert child.id != p2.id


def test_structural_validity(dim):
    p_valid = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    assert p_valid.is_structurally_valid()
    bad_sigma = -np.eye(dim)
    p_invalid = GaussianPattern(mu=np.zeros(dim), sigma=bad_sigma)
    assert not p_invalid.is_structurally_valid()


def test_serialisation_roundtrip(dim):
    p = GaussianPattern(mu=np.arange(dim, dtype=float), sigma=np.eye(dim) * 2)
    d = p.to_dict()
    p2 = GaussianPattern.from_dict(d)
    assert p2.id == p.id
    assert np.allclose(p2.mu, p.mu)
    assert np.allclose(p2.sigma, p.sigma)


def test_description_length_positive(dim):
    p = GaussianPattern(mu=np.ones(dim), sigma=np.eye(dim))
    assert p.description_length() > 0


def test_compress_between_zero_and_one(dim):
    p = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
    c = p.compress()
    assert 0.0 <= c <= 1.0
