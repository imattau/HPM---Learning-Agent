"""
Unit tests for hfn.recombination.Recombination.

Tests cover:
  - absorb: deregisters both nodes, registers merged, returns HFN
  - compress: registers new node, originals stay active, returns HFN
"""

import numpy as np
import pytest

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.recombination import Recombination


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_node(mu, node_id):
    D = len(mu)
    return HFN(
        mu=np.array(mu, dtype=float),
        sigma=np.eye(D),
        id=node_id,
    )


@pytest.fixture
def recombination():
    return Recombination()


@pytest.fixture
def forest_with_two_nodes():
    f = Forest(D=2, forest_id="test")
    a = make_node([0.0, 0.0], "node_a")
    b = make_node([1.0, 0.0], "node_b")
    f.register(a)
    f.register(b)
    return f, a, b


# ---------------------------------------------------------------------------
# absorb
# ---------------------------------------------------------------------------

class TestAbsorb:
    def test_returns_hfn(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        result = recombination.absorb(absorbed=a, dominant=b, forest=forest)
        assert isinstance(result, HFN)

    def test_new_node_has_str_id(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        result = recombination.absorb(absorbed=a, dominant=b, forest=forest)
        assert isinstance(result.id, str)

    def test_both_originals_deregistered(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        recombination.absorb(absorbed=a, dominant=b, forest=forest)
        assert a.id not in forest
        assert b.id not in forest

    def test_merged_node_registered(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        result = recombination.absorb(absorbed=a, dominant=b, forest=forest)
        assert result.id in forest

    def test_forest_has_exactly_one_node(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        recombination.absorb(absorbed=a, dominant=b, forest=forest)
        assert len(list(forest.active_nodes())) == 1

    def test_merged_mu_is_average_of_dominant_and_absorbed(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        # dominant=b at [1,0], absorbed=a at [0,0] → merged mu ≈ [0.5, 0]
        result = recombination.absorb(absorbed=a, dominant=b, forest=forest)
        expected_mu = np.array([0.5, 0.0])
        np.testing.assert_allclose(result.mu, expected_mu, atol=1e-9)

    def test_either_direction_works(self, recombination):
        """absorb(a, b) and absorb(b, a) both produce valid merged nodes."""
        for absorbed_id, dominant_id in [("a", "b"), ("b", "a")]:
            f = Forest(D=2, forest_id="test_dir")
            a = make_node([0.0, 0.0], "a")
            b = make_node([1.0, 0.0], "b")
            f.register(a)
            f.register(b)
            rec = Recombination()
            absorbed = a if absorbed_id == "a" else b
            dominant = b if dominant_id == "b" else a
            result = rec.absorb(absorbed=absorbed, dominant=dominant, forest=f)
            assert isinstance(result, HFN)
            assert result.id in f


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------

class TestCompress:
    def test_returns_hfn(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        compressed_id = f"compressed({a.id[:8]},{b.id[:8]})"
        result = recombination.compress(a, b, forest, compressed_id)
        assert isinstance(result, HFN)

    def test_compressed_id_assigned(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        compressed_id = f"compressed({a.id[:8]},{b.id[:8]})"
        result = recombination.compress(a, b, forest, compressed_id)
        assert result.id == compressed_id

    def test_new_node_registered(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        compressed_id = f"compressed({a.id[:8]},{b.id[:8]})"
        result = recombination.compress(a, b, forest, compressed_id)
        assert result.id in forest

    def test_originals_remain_active(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        compressed_id = f"compressed({a.id[:8]},{b.id[:8]})"
        recombination.compress(a, b, forest, compressed_id)
        assert a.id in forest
        assert b.id in forest

    def test_forest_has_three_nodes_after_compress(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        compressed_id = f"compressed({a.id[:8]},{b.id[:8]})"
        recombination.compress(a, b, forest, compressed_id)
        assert len(list(forest.active_nodes())) == 3

    def test_compressed_mu_is_average(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        compressed_id = "compressed_test"
        result = recombination.compress(a, b, forest, compressed_id)
        expected_mu = np.array([0.5, 0.0])
        np.testing.assert_allclose(result.mu, expected_mu, atol=1e-9)

    def test_custom_compressed_id(self, recombination, forest_with_two_nodes):
        forest, a, b = forest_with_two_nodes
        custom_id = "my_custom_compressed_id"
        result = recombination.compress(a, b, forest, custom_id)
        assert result.id == custom_id
        assert custom_id in forest
