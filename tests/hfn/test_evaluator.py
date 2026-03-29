"""
Unit tests for hfn.evaluator.Evaluator.

Tests cover all three responsibility classes:
  1. Fractal geometry: crowding, density_ratio, nearest_prior_dist,
                       hausdorff_candidates, persistence_scores
  2. Gap detection: coverage_gap, underrepresented_regions
  3. HPM framework: accuracy, description_length, score, coherence,
                    curiosity, boredom, reinforcement_signal
"""

import numpy as np
import pytest

from hfn.hfn import HFN
from hfn.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_node(mu, sigma_scale=1.0, node_id=None):
    D = len(mu)
    return HFN(
        mu=np.array(mu, dtype=float),
        sigma=np.eye(D) * sigma_scale,
        id=node_id or f"node_{mu}",
    )


@pytest.fixture
def evaluator():
    return Evaluator()


@pytest.fixture
def three_nodes():
    """Three nodes in 2D, spread out."""
    return [
        make_node([0.0, 0.0], node_id="a"),
        make_node([1.0, 0.0], node_id="b"),
        make_node([0.5, 1.0], node_id="c"),
    ]


# ---------------------------------------------------------------------------
# 1. Fractal geometry
# ---------------------------------------------------------------------------

class TestCrowding:
    def test_counts_nodes_within_radius(self, evaluator, three_nodes):
        # Node "a" at [0,0]; "b" at [1,0]; "c" at [0.5,1.0]
        # Radius 0.6 from [0,0]: only "a" itself is within 0.6
        centre = np.array([0.0, 0.0])
        count = evaluator.crowding(centre, three_nodes, radius=0.6)
        assert count == 1  # only "a"

    def test_larger_radius_includes_more(self, evaluator, three_nodes):
        centre = np.array([0.0, 0.0])
        count = evaluator.crowding(centre, three_nodes, radius=1.5)
        # "a" at [0,0] dist=0, "b" at [1,0] dist=1, "c" at [0.5,1.0] dist≈1.118 — all within 1.5
        assert count == 3

    def test_zero_when_all_outside(self, evaluator, three_nodes):
        centre = np.array([10.0, 10.0])
        assert evaluator.crowding(centre, three_nodes, radius=0.1) == 0

    def test_empty_nodes(self, evaluator):
        assert evaluator.crowding(np.array([0.0]), [], radius=1.0) == 0


class TestDensityRatio:
    def test_returns_zero_for_fewer_than_3_nodes(self, evaluator):
        nodes = [make_node([0.0, 0.0], node_id="x"), make_node([1.0, 0.0], node_id="y")]
        assert evaluator.density_ratio(np.array([0.0, 0.0]), nodes, radius=1.0) == 0.0

    def test_dense_region_ratio_greater_than_sparse(self, evaluator):
        # Create 5 nodes clustered near origin
        nodes = [make_node([i * 0.1, 0.0], node_id=f"n{i}") for i in range(5)]
        # Query at origin (dense) vs query far away (sparse)
        dense_r = evaluator.density_ratio(np.array([0.0, 0.0]), nodes, radius=0.3)
        sparse_r = evaluator.density_ratio(np.array([10.0, 0.0]), nodes, radius=0.3)
        assert dense_r >= sparse_r

    def test_returns_float(self, evaluator, three_nodes):
        result = evaluator.density_ratio(np.array([0.5, 0.5]), three_nodes, radius=0.5)
        assert isinstance(result, float)


class TestNearestPriorDist:
    def test_correct_distance(self, evaluator):
        prior_mus = np.array([[0.0, 0.0], [1.0, 0.0]])
        mu = np.array([0.3, 0.0])
        dist = evaluator.nearest_prior_dist(mu, prior_mus)
        assert abs(dist - 0.3) < 1e-9

    def test_returns_inf_for_empty(self, evaluator):
        dist = evaluator.nearest_prior_dist(np.array([0.0, 0.0]), np.array([]).reshape(0, 2))
        assert dist == float("inf")

    def test_single_prior(self, evaluator):
        prior_mus = np.array([[2.0, 0.0]])
        mu = np.array([0.0, 0.0])
        assert abs(evaluator.nearest_prior_dist(mu, prior_mus) - 2.0) < 1e-9


class TestHausdorffCandidates:
    def test_finds_weak_node_near_strong(self, evaluator):
        weak = make_node([0.1, 0.0], node_id="weak")
        strong = make_node([0.0, 0.0], node_id="strong")
        weights = {"weak": 0.05, "strong": 0.8}
        pairs = evaluator.hausdorff_candidates(
            [weak, strong], weights, threshold=0.5, weight_floor=0.2,
            protected_ids=set()
        )
        assert len(pairs) == 1
        assert pairs[0][0].id == "weak"
        assert pairs[0][1].id == "strong"

    def test_excludes_protected(self, evaluator):
        weak = make_node([0.1, 0.0], node_id="weak")
        strong = make_node([0.0, 0.0], node_id="strong")
        weights = {"weak": 0.05, "strong": 0.8}
        pairs = evaluator.hausdorff_candidates(
            [weak, strong], weights, threshold=0.5, weight_floor=0.2,
            protected_ids={"strong"}
        )
        assert len(pairs) == 0

    def test_excludes_above_weight_floor(self, evaluator):
        node_a = make_node([0.1, 0.0], node_id="a")
        node_b = make_node([0.0, 0.0], node_id="b")
        weights = {"a": 0.5, "b": 0.8}  # a is above floor
        pairs = evaluator.hausdorff_candidates(
            [node_a, node_b], weights, threshold=0.5, weight_floor=0.3,
            protected_ids=set()
        )
        assert len(pairs) == 0

    def test_excludes_nodes_too_far(self, evaluator):
        weak = make_node([5.0, 0.0], node_id="weak")
        strong = make_node([0.0, 0.0], node_id="strong")
        weights = {"weak": 0.05, "strong": 0.8}
        pairs = evaluator.hausdorff_candidates(
            [weak, strong], weights, threshold=0.3, weight_floor=0.2,
            protected_ids=set()
        )
        assert len(pairs) == 0


class TestPersistenceScores:
    def test_returns_dict(self, evaluator, three_nodes):
        weights = {n.id: 0.5 for n in three_nodes}
        result = evaluator.persistence_scores(three_nodes, weights)
        assert isinstance(result, dict)
        assert set(result.keys()) == {n.id for n in three_nodes}

    def test_values_are_floats(self, evaluator, three_nodes):
        weights = {n.id: 0.5 for n in three_nodes}
        result = evaluator.persistence_scores(three_nodes, weights)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# 2. Gap detection
# ---------------------------------------------------------------------------

class TestCoverageGap:
    def test_max_gap_when_empty(self, evaluator):
        result = evaluator.coverage_gap(np.array([0.0, 0.0]), [], radius=1.0)
        assert result == 1.0

    def test_gap_decreases_with_nearby_nodes(self, evaluator):
        nodes = [make_node([0.0, 0.0], node_id="a")]
        gap = evaluator.coverage_gap(np.array([0.1, 0.0]), nodes, radius=1.0)
        assert gap < 1.0

    def test_full_gap_when_node_outside_radius(self, evaluator):
        nodes = [make_node([10.0, 0.0], node_id="far")]
        gap = evaluator.coverage_gap(np.array([0.0, 0.0]), nodes, radius=0.5)
        assert gap == 1.0

    def test_returns_value_in_0_1(self, evaluator, three_nodes):
        result = evaluator.coverage_gap(np.array([0.5, 0.5]), three_nodes, radius=0.5)
        assert 0.0 <= result <= 1.0


class TestUnderrepresentedRegions:
    def test_returns_empty_for_few_nodes(self, evaluator):
        nodes = [make_node([0.0], node_id="x"), make_node([1.0], node_id="y")]
        result = evaluator.underrepresented_regions(nodes)
        assert result == []

    def test_returns_list_of_arrays(self, evaluator):
        nodes = [make_node([float(i), 0.0], node_id=f"n{i}") for i in range(5)]
        result = evaluator.underrepresented_regions(nodes)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, np.ndarray)

    def test_no_crash_with_clustered_nodes(self, evaluator):
        nodes = [make_node([0.0, float(i) * 0.1], node_id=f"n{i}") for i in range(4)]
        result = evaluator.underrepresented_regions(nodes)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. HPM framework evaluations
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_returns_float_in_0_1(self, evaluator):
        node = make_node([0.0, 0.0])
        result = evaluator.accuracy(np.array([0.0, 0.0]), node)
        assert 0.0 < result <= 1.0

    def test_higher_at_mean(self, evaluator):
        node = make_node([0.0, 0.0])
        at_mean = evaluator.accuracy(np.array([0.0, 0.0]), node)
        far_away = evaluator.accuracy(np.array([100.0, 0.0]), node)
        assert at_mean > far_away


class TestDescriptionLength:
    def test_returns_float(self, evaluator):
        node = make_node([1.0, 2.0])
        result = evaluator.description_length(node)
        assert isinstance(result, float)

    def test_matches_node_method(self, evaluator):
        node = make_node([1.0, 2.0])
        assert evaluator.description_length(node) == node.description_length()


class TestScore:
    def test_combines_accuracy_and_complexity(self, evaluator):
        node = make_node([0.0, 0.0])
        x = np.array([0.0, 0.0])
        s = evaluator.score(x, node, lambda_complexity=0.1)
        expected = evaluator.accuracy(x, node) - 0.1 * evaluator.description_length(node)
        assert abs(s - expected) < 1e-12

    def test_higher_lambda_penalises_more(self, evaluator):
        node = make_node([0.0, 0.0])
        x = np.array([0.0, 0.0])
        s_low = evaluator.score(x, node, lambda_complexity=0.0)
        s_high = evaluator.score(x, node, lambda_complexity=1.0)
        assert s_low >= s_high


class TestCoherence:
    def test_identity_sigma_high_coherence(self, evaluator):
        node = make_node([0.0, 0.0], sigma_scale=1.0)
        result = evaluator.coherence(node)
        assert result > 0.0

    def test_returns_float_in_0_1(self, evaluator):
        node = make_node([1.0, 2.0, 3.0])
        result = evaluator.coherence(node)
        assert 0.0 <= result <= 1.0

    def test_ill_conditioned_returns_low(self, evaluator):
        # Very large condition number — near-singular sigma
        sigma = np.diag([1e6, 1e-6])
        node = HFN(mu=np.array([0.0, 0.0]), sigma=sigma, id="ill")
        result = evaluator.coherence(node)
        assert result < 0.01


class TestCuriosity:
    def test_returns_one_for_empty_nodes(self, evaluator):
        result = evaluator.curiosity(np.array([0.0, 0.0]), [], {})
        assert result == 1.0

    def test_returns_float_in_0_1(self, evaluator, three_nodes):
        weights = {n.id: 0.5 for n in three_nodes}
        result = evaluator.curiosity(np.array([0.5, 0.5]), three_nodes, weights)
        assert 0.0 <= result <= 1.0

    def test_higher_for_distant_observation(self, evaluator, three_nodes):
        weights = {n.id: 0.5 for n in three_nodes}
        near = evaluator.curiosity(np.array([0.0, 0.0]), three_nodes, weights)
        far = evaluator.curiosity(np.array([100.0, 100.0]), three_nodes, weights)
        assert far > near


class TestBoredom:
    def test_high_weight_zero_score_is_bored(self, evaluator):
        node = make_node([0.0, 0.0], node_id="n")
        weights = {"n": 0.9}
        scores = {"n": 0.0}
        result = evaluator.boredom(node, weights, scores)
        assert result > 0.8

    def test_high_weight_high_score_not_bored(self, evaluator):
        node = make_node([0.0, 0.0], node_id="n")
        weights = {"n": 0.9}
        scores = {"n": 0.85}
        result = evaluator.boredom(node, weights, scores)
        assert result < 0.1

    def test_returns_float_in_0_1(self, evaluator):
        node = make_node([0.0, 0.0], node_id="n")
        result = evaluator.boredom(node, {"n": 0.5}, {"n": 0.3})
        assert 0.0 <= result <= 1.0

    def test_zero_weight_zero_boredom(self, evaluator):
        node = make_node([0.0, 0.0], node_id="n")
        result = evaluator.boredom(node, {"n": 0.0}, {"n": 0.0})
        assert result == pytest.approx(0.0, abs=0.01)


class TestReinforcementSignal:
    def test_default_returns_zero(self, evaluator):
        assert evaluator.reinforcement_signal("any_id") == 0.0

    def test_subclass_can_override(self):
        class CustomEvaluator(Evaluator):
            def reinforcement_signal(self, node_id: str) -> float:
                return 1.0 if node_id == "special" else 0.0

        ev = CustomEvaluator()
        assert ev.reinforcement_signal("special") == 1.0
        assert ev.reinforcement_signal("other") == 0.0
