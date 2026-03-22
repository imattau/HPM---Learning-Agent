"""Tests for factory.py with categorical pattern support."""
import numpy as np
import pytest
from hpm.patterns.factory import make_pattern, pattern_from_dict
from hpm.patterns.categorical import CategoricalPattern
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern


class TestMakePatternCategorical:
    def test_make_categorical_returns_categorical(self):
        mu = np.zeros(5)
        p = make_pattern(mu, scale=1.0, pattern_type="categorical")
        assert isinstance(p, CategoricalPattern)

    def test_make_categorical_default_alphabet_size(self):
        mu = np.zeros(3)
        p = make_pattern(mu, scale=1.0, pattern_type="categorical")
        assert p.K == 10  # default alphabet_size=10

    def test_make_categorical_custom_alphabet_size(self):
        mu = np.zeros(4)
        p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=5)
        assert p.K == 5

    def test_make_categorical_k_kwarg_overrides_alphabet_size(self):
        mu = np.zeros(3)
        p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=10, K=6)
        assert p.K == 6

    def test_make_categorical_probs_shape(self):
        D = 4
        K = 7
        mu = np.zeros(D)
        p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=K)
        assert p.probs.shape == (D, K)

    def test_make_categorical_uniform_prior(self):
        """Initial probs should be uniform."""
        mu = np.zeros(3)
        p = make_pattern(mu, scale=1.0, pattern_type="categorical", alphabet_size=4)
        expected = np.ones((3, 4)) / 4
        np.testing.assert_allclose(p.probs, expected, atol=1e-10)

    def test_make_categorical_uses_mu_for_D(self):
        """mu is used only for dimensionality."""
        mu = [1.0, 2.0, 3.0, 4.0, 5.0]  # D=5
        p = make_pattern(mu, scale=99.0, pattern_type="categorical", alphabet_size=3)
        assert p.probs.shape[0] == 5

    def test_make_categorical_passes_kwargs(self):
        mu = np.zeros(2)
        p = make_pattern(mu, scale=1.0, pattern_type="categorical",
                         alphabet_size=4, level=3, id="my-cat")
        assert p.level == 3
        assert p.id == "my-cat"

    def test_make_gaussian_unaffected(self):
        """alphabet_size has no effect on Gaussian patterns."""
        mu = np.zeros(3)
        p = make_pattern(mu, np.eye(3), pattern_type="gaussian", alphabet_size=99)
        assert isinstance(p, GaussianPattern)

    def test_make_laplace_unaffected(self):
        """alphabet_size has no effect on Laplace patterns."""
        mu = np.zeros(3)
        p = make_pattern(mu, 1.0, pattern_type="laplace", alphabet_size=99)
        assert isinstance(p, LaplacePattern)

    def test_unknown_pattern_type_raises(self):
        with pytest.raises(ValueError):
            make_pattern(np.zeros(3), 1.0, pattern_type="unknown")

    def test_error_message_includes_categorical(self):
        with pytest.raises(ValueError, match="categorical"):
            make_pattern(np.zeros(3), 1.0, pattern_type="bad_type")


class TestPatternFromDictCategorical:
    def test_from_dict_categorical(self):
        probs = np.ones((3, 4)) / 4
        p = CategoricalPattern(probs, K=4, id="abc", level=2)
        d = p.to_dict()
        p2 = pattern_from_dict(d)
        assert isinstance(p2, CategoricalPattern)
        np.testing.assert_allclose(p2.probs, p.probs)
        assert p2.K == p.K
        assert p2.id == p.id

    def test_from_dict_gaussian(self):
        p = GaussianPattern(np.zeros(2), np.eye(2), id="gauss-1")
        d = p.to_dict()
        p2 = pattern_from_dict(d)
        assert isinstance(p2, GaussianPattern)

    def test_from_dict_laplace(self):
        p = LaplacePattern(np.zeros(2), np.ones(2), id="lap-1")
        d = p.to_dict()
        p2 = pattern_from_dict(d)
        assert isinstance(p2, LaplacePattern)

    def test_from_dict_unknown_raises(self):
        d = {'type': 'unknown', 'id': 'x'}
        with pytest.raises(ValueError, match="categorical"):
            pattern_from_dict(d)

    def test_categorical_round_trip_via_factory(self):
        mu = np.zeros(5)
        p = make_pattern(mu, 1.0, pattern_type="categorical", alphabet_size=6)
        x = np.array([0, 1, 2, 3, 4], dtype=int)
        for _ in range(5):
            p = p.update(x)
        d = p.to_dict()
        p2 = pattern_from_dict(d)
        assert isinstance(p2, CategoricalPattern)
        np.testing.assert_allclose(p2.probs, p.probs)
        assert p2._n_obs == p._n_obs
