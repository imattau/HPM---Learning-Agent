import numpy as np
import pytest
from hpm.patterns.factory import make_pattern, pattern_from_dict
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern


def test_make_pattern_gaussian():
    p = make_pattern(np.zeros(3), np.eye(3), pattern_type="gaussian")
    assert isinstance(p, GaussianPattern)


def test_make_pattern_laplace():
    p = make_pattern(np.zeros(3), np.ones(3), pattern_type="laplace")
    assert isinstance(p, LaplacePattern)


def test_make_pattern_laplace_scalar_scale_broadcast():
    p = make_pattern(np.zeros(4), 2.0, pattern_type="laplace")
    assert isinstance(p, LaplacePattern)
    assert p.b.shape == (4,)
    assert np.allclose(p.b, 2.0)


def test_make_pattern_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pattern_type"):
        make_pattern(np.zeros(3), np.eye(3), pattern_type="von_mises")


def test_pattern_from_dict_gaussian():
    p = GaussianPattern(np.zeros(3), np.eye(3))
    d = p.to_dict()
    restored = pattern_from_dict(d)
    assert isinstance(restored, GaussianPattern)
    assert restored.id == p.id


def test_pattern_from_dict_laplace():
    p = LaplacePattern(np.zeros(3), np.ones(3))
    d = p.to_dict()
    restored = pattern_from_dict(d)
    assert isinstance(restored, LaplacePattern)
    assert restored.id == p.id


def test_pattern_from_dict_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pattern type in dict"):
        pattern_from_dict({'type': 'dirac', 'mu': [0.0]})


def test_pattern_from_dict_defaults_to_gaussian():
    """Dict without 'type' key should default to Gaussian (backward compat)."""
    p = GaussianPattern(np.zeros(3), np.eye(3))
    d = p.to_dict()
    del d['type']
    restored = pattern_from_dict(d)
    assert isinstance(restored, GaussianPattern)
