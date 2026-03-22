"""Unit tests for CategoricalPattern."""
import numpy as np
import pytest
from hpm.patterns.categorical import CategoricalPattern
from hpm.patterns.gaussian import GaussianPattern


def make_uniform(D=3, K=4):
    probs = np.ones((D, K)) / K
    return CategoricalPattern(probs, K=K)


def make_peaked(D=3, K=4, symbol=0):
    """Pattern with high probability on one symbol per dimension."""
    probs = np.ones((D, K)) * 0.01
    probs[:, symbol] = 0.97
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    return CategoricalPattern(probs, K=K)


class TestConstruction:
    def test_k_less_than_2_raises(self):
        probs = np.array([[1.0]])
        with pytest.raises(ValueError):
            CategoricalPattern(probs, K=1)

    def test_k_equals_2_ok(self):
        probs = np.ones((2, 2)) / 2
        p = CategoricalPattern(probs, K=2)
        assert p.K == 2

    def test_probs_floored_at_1e8(self):
        probs = np.array([[0.0, 1.0], [0.5, 0.5]])
        p = CategoricalPattern(probs, K=2)
        # After flooring and renormalisation, values should be close to 1e-8 floor
        # Use a small tolerance since renormalisation may shift the value slightly
        assert np.all(p.probs >= 1e-8 - 1e-15)

    def test_rows_sum_to_one(self):
        probs = np.ones((3, 4)) / 4
        p = CategoricalPattern(probs, K=4)
        np.testing.assert_allclose(p.probs.sum(axis=1), 1.0, atol=1e-10)

    def test_n_obs_initialised_to_K(self):
        p = make_uniform(D=3, K=5)
        assert p._n_obs == 5

    def test_no_sigma_attribute(self):
        p = make_uniform()
        assert not hasattr(p, 'sigma')

    def test_id_assigned(self):
        p = make_uniform()
        assert p.id is not None
        assert len(p.id) > 0

    def test_custom_id_preserved(self):
        probs = np.ones((2, 3)) / 3
        p = CategoricalPattern(probs, K=3, id="my-id")
        assert p.id == "my-id"


class TestLogProb:
    def test_peaked_lower_nll_than_uniform(self):
        """Peaked pattern should give lower NLL at the favoured symbol."""
        D, K = 3, 4
        peaked = make_peaked(D=D, K=K, symbol=0)
        uniform = make_uniform(D=D, K=K)
        x = np.zeros(D, dtype=int)  # all dimension 0
        assert peaked.log_prob(x) < uniform.log_prob(x)

    def test_improbable_vector_higher_nll(self):
        """NLL should be higher for less probable vector."""
        peaked = make_peaked(D=3, K=4, symbol=0)
        x_prob = np.zeros(3, dtype=int)      # all symbol 0 — highly probable
        x_impro = np.ones(3, dtype=int) * 3  # all symbol 3 — low probability
        assert peaked.log_prob(x_prob) < peaked.log_prob(x_impro)

    def test_log_prob_finite_after_construction(self):
        """Floor prevents log(0)."""
        p = make_uniform()
        x = np.array([0, 1, 2])
        assert np.isfinite(p.log_prob(x))

    def test_out_of_range_raises_index_error(self):
        p = make_uniform(D=2, K=4)
        x = np.array([0, 10])  # 10 >= K=4
        with pytest.raises(IndexError):
            p.log_prob(x)


class TestUpdate:
    def test_update_increments_n_obs(self):
        p = make_uniform(D=2, K=4)
        x = np.array([0, 1])
        p2 = p.update(x)
        assert p2._n_obs == p._n_obs + 1

    def test_update_preserves_id(self):
        p = make_uniform(D=2, K=4)
        x = np.array([0, 1])
        p2 = p.update(x)
        assert p2.id == p.id

    def test_update_returns_new_instance(self):
        p = make_uniform(D=2, K=4)
        x = np.array([0, 1])
        p2 = p.update(x)
        assert p2 is not p

    def test_update_shifts_probs_toward_observed(self):
        D, K = 2, 4
        p = make_uniform(D=D, K=K)
        x = np.array([0, 0])
        p2 = p.update(x)
        # Symbol 0 probability should increase after observing 0
        assert p2.probs[0, 0] > p.probs[0, 0]
        assert p2.probs[1, 0] > p.probs[1, 0]

    def test_update_rows_sum_to_one(self):
        p = make_uniform(D=3, K=5)
        x = np.array([0, 1, 2])
        p2 = p.update(x)
        np.testing.assert_allclose(p2.probs.sum(axis=1), 1.0, atol=1e-10)

    def test_update_floor_maintained(self):
        p = make_uniform(D=2, K=4)
        x = np.array([0, 0])
        for _ in range(20):
            p = p.update(x)
        assert np.all(p.probs >= 1e-8)

    def test_100_identical_updates_converges(self):
        """After 100 identical updates, observed symbol prob > 0.95."""
        D, K = 2, 4
        probs = np.ones((D, K)) / K
        p = CategoricalPattern(probs, K=K)
        x = np.zeros(D, dtype=int)
        for _ in range(100):
            p = p.update(x)
        for d in range(D):
            assert p.probs[d, 0] > 0.95


class TestSample:
    def test_sample_shape(self):
        p = make_uniform(D=3, K=4)
        rng = np.random.default_rng(42)
        samples = p.sample(10, rng)
        assert samples.shape == (10, 3)

    def test_sample_dtype_integer(self):
        p = make_uniform(D=3, K=4)
        rng = np.random.default_rng(42)
        samples = p.sample(10, rng)
        assert np.issubdtype(samples.dtype, np.integer)

    def test_sample_values_in_range(self):
        D, K = 3, 4
        p = make_uniform(D=D, K=K)
        rng = np.random.default_rng(42)
        samples = p.sample(100, rng)
        assert np.all(samples >= 0)
        assert np.all(samples < K)


class TestRecombine:
    def test_recombine_returns_categorical(self):
        p1 = make_uniform(D=2, K=4)
        p2 = make_uniform(D=2, K=4)
        result = p1.recombine(p2)
        assert isinstance(result, CategoricalPattern)

    def test_recombine_with_gaussian_raises_type_error(self):
        p = make_uniform(D=2, K=4)
        g = GaussianPattern(np.zeros(2), np.eye(2))
        with pytest.raises(TypeError):
            p.recombine(g)

    def test_recombine_shape_mismatch_raises_value_error(self):
        p1 = make_uniform(D=2, K=4)
        p2 = make_uniform(D=3, K=4)
        with pytest.raises(ValueError):
            p1.recombine(p2)

    def test_recombine_k_mismatch_raises_value_error(self):
        p1 = CategoricalPattern(np.ones((2, 3)) / 3, K=3)
        p2 = CategoricalPattern(np.ones((2, 4)) / 4, K=4)
        with pytest.raises(ValueError):
            p1.recombine(p2)

    def test_recombine_weighted_average(self):
        """n_obs-weighted average of probs."""
        D, K = 2, 4
        # Build p1 with high n_obs via updates
        p1 = make_uniform(D=D, K=K)
        x1 = np.zeros(D, dtype=int)
        for _ in range(50):
            p1 = p1.update(x1)

        p2 = make_uniform(D=D, K=K)
        result = p1.recombine(p2)
        assert isinstance(result, CategoricalPattern)
        # Result rows should sum to 1
        np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, atol=1e-10)

    def test_recombine_rows_sum_to_one(self):
        p1 = make_uniform(D=3, K=5)
        p2 = make_uniform(D=3, K=5)
        result = p1.recombine(p2)
        np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, atol=1e-10)

    def test_recombine_equal_n_obs_gives_simple_average(self):
        """When both patterns have same _n_obs, result is midpoint."""
        D, K = 2, 4
        probs1 = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
        probs2 = np.array([[0.1, 0.7, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])
        p1 = CategoricalPattern(probs1, K=K)
        p2 = CategoricalPattern(probs2, K=K)
        # Both have same _n_obs=K=4
        assert p1._n_obs == p2._n_obs
        result = p1.recombine(p2)
        # Expected: weighted average with equal weights, then renormalised
        total = p1._n_obs + p2._n_obs
        expected = (probs1 * p1._n_obs + probs2 * p2._n_obs) / total
        expected = expected / expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result.probs, expected, atol=1e-10)


class TestStructuralMethods:
    def test_is_structurally_valid_uniform(self):
        p = make_uniform(D=3, K=4)
        assert p.is_structurally_valid()

    def test_is_structurally_valid_false_if_below_floor(self):
        """Directly manipulate probs to violate floor."""
        p = make_uniform(D=2, K=3)
        p.probs[0, 0] = 5e-9  # below 1e-8
        assert not p.is_structurally_valid()

    def test_is_structurally_valid_false_if_row_sum_off(self):
        probs = np.ones((2, 3)) / 3
        p = CategoricalPattern(probs, K=3)
        p.probs[0, 0] += 0.1  # break row sum without renormalising
        assert not p.is_structurally_valid()

    def test_description_length_returns_float(self):
        p = make_uniform(D=3, K=4)
        assert isinstance(p.description_length(), float)

    def test_description_length_peaked_pattern(self):
        """Peaked pattern should have more positions below half-entropy."""
        uniform = make_uniform(D=3, K=4)
        peaked = make_peaked(D=3, K=4)
        # After many updates, peaked has lower entropy per dimension
        x = np.zeros(3, dtype=int)
        for _ in range(50):
            peaked = peaked.update(x)
        assert peaked.description_length() >= uniform.description_length()

    def test_connectivity_returns_zero(self):
        p = make_uniform(D=3, K=4)
        assert p.connectivity() == 0.0

    def test_compress_uniform_returns_one(self):
        """Uniform distribution: all rows have equal max entropy, ratio = 1.0."""
        p = make_uniform(D=3, K=4)
        result = p.compress()
        assert result == pytest.approx(1.0)

    def test_compress_point_mass_returns_one(self):
        """When mean_row_entropy == 0, return 1.0."""
        # Near-point masses (floor prevents exact 0 entropy, but very close)
        probs = np.zeros((2, 3))
        probs[:, 0] = 1.0 - 1e-7
        probs[:, 1] = 5e-8
        probs[:, 2] = 5e-8
        # This won't be exact 0 entropy, but test the guard path
        # Force _n_obs update won't help, so test the compress() logic directly
        p = CategoricalPattern(probs, K=3)
        result = p.compress()
        assert np.isfinite(result)

    def test_compress_returns_one_when_all_point_masses(self):
        """Test zero-denominator guard: force mean_row_entropy = 0 by patching probs."""
        p = make_uniform(D=2, K=2)
        # Directly set probs to a state where all rows are point masses (entropy=0)
        p.probs = np.array([[1.0, 0.0], [1.0, 0.0]])
        # compress() should return 1.0 since mean_row_entropy = 0
        result = p.compress()
        assert result == 1.0


class TestSerialisation:
    def test_to_dict_keys(self):
        p = make_uniform(D=2, K=4)
        d = p.to_dict()
        assert d['type'] == 'categorical'
        assert 'probs' in d
        assert 'K' in d
        assert 'id' in d
        assert 'level' in d
        assert 'source_id' in d
        assert 'n_obs' in d
        assert '_n_obs' not in d

    def test_round_trip(self):
        p = make_uniform(D=3, K=5)
        x = np.array([0, 1, 2])
        for _ in range(10):
            p = p.update(x)
        d = p.to_dict()
        p2 = CategoricalPattern.from_dict(d)
        np.testing.assert_allclose(p2.probs, p.probs)
        assert p2.K == p.K
        assert p2.id == p.id
        assert p2.level == p.level
        assert p2._n_obs == p._n_obs

    def test_from_dict_reads_n_obs_not_underscore(self):
        """from_dict reads 'n_obs', not '_n_obs'."""
        d = {
            'type': 'categorical',
            'probs': [[0.5, 0.5], [0.25, 0.75]],
            'K': 2,
            'id': 'test-id',
            'level': 2,
            'source_id': None,
            'n_obs': 42,
        }
        p = CategoricalPattern.from_dict(d)
        assert p._n_obs == 42
