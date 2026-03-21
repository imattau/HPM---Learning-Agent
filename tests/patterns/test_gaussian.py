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


def test_source_id_defaults_to_none():
    p = GaussianPattern(np.zeros(2), np.eye(2))
    assert p.source_id is None


def test_source_id_stored_from_constructor():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    assert p.source_id == 'abc-123'


def test_to_dict_includes_source_id():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    d = p.to_dict()
    assert 'source_id' in d
    assert d['source_id'] == 'abc-123'


def test_to_dict_source_id_none_when_not_set():
    p = GaussianPattern(np.zeros(2), np.eye(2))
    assert p.to_dict()['source_id'] is None


def test_from_dict_restores_source_id():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    p2 = GaussianPattern.from_dict(p.to_dict())
    assert p2.source_id == 'abc-123'


def test_from_dict_without_source_id_key_defaults_none():
    """Existing serialised patterns (no source_id key) round-trip cleanly."""
    p = GaussianPattern(np.zeros(2), np.eye(2))
    d = p.to_dict()
    del d['source_id']
    p2 = GaussianPattern.from_dict(d)
    assert p2.source_id is None


def test_update_preserves_source_id():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    p2 = p.update(np.ones(2))
    assert p2.source_id == 'abc-123'


def test_recombine_does_not_inherit_source_id():
    pa = GaussianPattern(np.zeros(2), np.eye(2), source_id='parent-a')
    pb = GaussianPattern(np.ones(2), np.eye(2), source_id='parent-b')
    child = pa.recombine(pb)
    assert child.source_id is None
