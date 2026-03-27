"""
Tests for the HPM Forest.

Tests cover:
- Registry: registration, deregistration, initial weights
- Retrieval: nearest-neighbour proximity search
- Competitive dynamics: weight gain for explaining nodes, weight loss for overlapping non-explaining nodes
- Structural absorption: persistent overlap resolves into hierarchy
- Node creation: residual surprise spawns leaf, co-occurrence triggers compression node
"""

import numpy as np
import pytest
from hpm_fractal_node.hfn import HFN, make_leaf
from hpm_fractal_node.forest import Forest, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_query_result(nodes, accuracies, residual=0.0):
    return QueryResult(
        explanation_tree=nodes,
        accuracy_scores={n.id: acc for n, acc in zip(nodes, accuracies)},
        residual_surprise=residual,
    )


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------

def test_register_adds_node():
    forest = Forest()
    node = make_leaf("a")
    forest.register(node)
    assert node in forest.active_nodes()


def test_new_node_starts_with_low_weight():
    forest = Forest(w_init=0.1)
    node = make_leaf("a")
    forest.register(node)
    assert forest.get_weight(node.id) == pytest.approx(0.1)


def test_register_is_idempotent():
    forest = Forest()
    node = make_leaf("a")
    forest.register(node)
    forest.register(node)
    assert len(forest) == 1


def test_deregister_removes_node():
    forest = Forest()
    node = make_leaf("a")
    forest.register(node)
    forest.deregister(node.id)
    assert node not in forest.active_nodes()


# ---------------------------------------------------------------------------
# 2. Retrieval
# ---------------------------------------------------------------------------

def test_retrieve_returns_nearest_nodes():
    forest = Forest()
    near = HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="near")
    far  = HFN(mu=np.array([9.0, 0.0]), sigma=np.eye(2), id="far")
    forest.register(near)
    forest.register(far)

    x = np.array([1.1, 0.0])
    results = forest.retrieve(x, k=1)
    assert results[0] is near


def test_retrieve_respects_k():
    forest = Forest()
    for i in range(5):
        forest.register(HFN(mu=np.array([float(i), 0.0]), sigma=np.eye(2), id=f"n{i}"))

    results = forest.retrieve(np.array([0.0, 0.0]), k=3)
    assert len(results) == 3


def test_retrieve_empty_forest_returns_empty():
    forest = Forest()
    assert forest.retrieve(np.array([0.0, 0.0])) == []


# ---------------------------------------------------------------------------
# 3. Competitive weight dynamics
# ---------------------------------------------------------------------------

def test_explaining_node_gains_weight():
    forest = Forest(alpha_gain=0.2, w_init=0.1)
    node = make_leaf("explainer")
    forest.register(node)
    x = node.mu.copy()

    result = make_query_result([node], [1.0])
    forest.update(x, result)

    assert forest.get_weight(node.id) > 0.1


def test_non_explaining_overlapping_node_loses_weight():
    forest = Forest(beta_loss=0.1, w_init=0.5)

    # Two nodes very close in latent space — high overlap
    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 2.0, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 2.0, id="loser")
    forest.register(winner)
    forest.register(loser)

    x = winner.mu.copy()
    result = make_query_result([winner], [1.0])
    forest.update(x, result)

    assert forest.get_weight(loser.id) < 0.5


def test_non_overlapping_node_weight_unchanged():
    forest = Forest(beta_loss=0.1, w_init=0.5)

    explainer = HFN(mu=np.array([0.0, 0.0]),  sigma=np.eye(2) * 0.01, id="explainer")
    distant   = HFN(mu=np.array([100.0, 0.0]), sigma=np.eye(2) * 0.01, id="distant")
    forest.register(explainer)
    forest.register(distant)

    x = explainer.mu.copy()
    result = make_query_result([explainer], [1.0])
    initial_weight = forest.get_weight(distant.id)
    forest.update(x, result)

    # overlap is essentially zero — weight should be nearly unchanged
    assert abs(forest.get_weight(distant.id) - initial_weight) < 1e-6


def test_score_updated_after_query():
    forest = Forest(lambda_complexity=0.0)  # no complexity penalty
    node = make_leaf("a")
    forest.register(node)
    x = node.mu.copy()

    result = make_query_result([node], [0.8])
    forest.update(x, result)

    assert forest.get_score(node.id) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 4. Structural absorption
# ---------------------------------------------------------------------------

def test_absorption_triggers_after_persistent_misses():
    forest = Forest(
        absorption_overlap_threshold=0.3,
        absorption_query_threshold=3,
        beta_loss=0.0,  # disable weight loss to isolate absorption logic
        w_init=0.5,
    )

    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 3.0, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 3.0, id="loser")
    forest.register(winner)
    forest.register(loser)

    x = winner.mu.copy()
    result = make_query_result([winner], [1.0])

    for _ in range(3):
        forest.update(x, result)

    assert loser.id in forest.absorbed_ids


def test_absorbed_node_is_deregistered():
    forest = Forest(
        absorption_overlap_threshold=0.3,
        absorption_query_threshold=3,
        beta_loss=0.0,
        w_init=0.5,
    )

    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 3.0, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 3.0, id="loser")
    forest.register(winner)
    forest.register(loser)

    x = winner.mu.copy()
    result = make_query_result([winner], [1.0])
    for _ in range(3):
        forest.update(x, result)

    active_ids = {n.id for n in forest.active_nodes()}
    assert loser.id not in active_ids


def test_absorbed_node_accessible_via_new_parent():
    forest = Forest(
        absorption_overlap_threshold=0.3,
        absorption_query_threshold=3,
        beta_loss=0.0,
        w_init=0.5,
    )

    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 3.0, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 3.0, id="loser")
    forest.register(winner)
    forest.register(loser)

    x = winner.mu.copy()
    result = make_query_result([winner], [1.0])
    for _ in range(3):
        forest.update(x, result)

    # A new parent node should exist in the forest containing loser as a child
    def find_node_in_tree(root, target_id):
        if root.id == target_id:
            return True
        return any(find_node_in_tree(c, target_id) for c in root.children())

    found = any(
        find_node_in_tree(n, loser.id)
        for n in forest.active_nodes()
    )
    assert found, "absorbed node should be reachable via expansion of new parent"


# ---------------------------------------------------------------------------
# 5. Node creation — residual surprise
# ---------------------------------------------------------------------------

def test_high_residual_surprise_creates_new_leaf():
    forest = Forest(residual_surprise_threshold=1.0, w_init=0.1)
    node = make_leaf("existing")
    forest.register(node)
    initial_count = len(forest)

    x = np.array([5.0, 5.0, 5.0, 5.0])  # far from existing node
    result = make_query_result([node], [0.1], residual=2.0)
    forest.update(x, result)

    assert len(forest) > initial_count


def test_low_residual_surprise_does_not_create_node():
    forest = Forest(residual_surprise_threshold=1.0, w_init=0.1)
    node = make_leaf("existing")
    forest.register(node)
    initial_count = len(forest)

    x = node.mu.copy()
    result = make_query_result([node], [1.0], residual=0.1)
    forest.update(x, result)

    assert len(forest) == initial_count


def test_new_leaf_starts_with_low_weight():
    forest = Forest(residual_surprise_threshold=1.0, w_init=0.1)
    node = make_leaf("existing")
    forest.register(node)

    x = np.array([5.0, 5.0, 5.0, 5.0])
    result = make_query_result([node], [0.1], residual=2.0)
    forest.update(x, result)

    new_nodes = [n for n in forest.active_nodes() if n.id != node.id]
    assert len(new_nodes) == 1
    assert forest.get_weight(new_nodes[0].id) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 6. Node creation — query-induced compression
# ---------------------------------------------------------------------------

def test_cooccurrence_creates_compression_node():
    forest = Forest(compression_cooccurrence_threshold=3, w_init=0.1)

    node_a = make_leaf("a")
    node_b = make_leaf("b")
    forest.register(node_a)
    forest.register(node_b)
    initial_count = len(forest)

    x = node_a.mu.copy()
    result = make_query_result([node_a, node_b], [1.0, 1.0])

    for _ in range(3):
        forest.update(x, result)

    assert len(forest) > initial_count


def test_compression_node_contains_both_originals():
    forest = Forest(compression_cooccurrence_threshold=3, w_init=0.1)

    node_a = make_leaf("alpha")
    node_b = make_leaf("beta")
    forest.register(node_a)
    forest.register(node_b)

    x = node_a.mu.copy()
    result = make_query_result([node_a, node_b], [1.0, 1.0])
    for _ in range(3):
        forest.update(x, result)

    new_nodes = [
        n for n in forest.active_nodes()
        if n.id not in {"alpha", "beta"}
    ]
    assert len(new_nodes) >= 1
    compressed = new_nodes[0]
    child_ids = {c.id for c in compressed.children()}
    assert "alpha" in child_ids
    assert "beta" in child_ids


def test_compression_node_not_created_twice():
    forest = Forest(compression_cooccurrence_threshold=3, w_init=0.1)

    node_a = make_leaf("a")
    node_b = make_leaf("b")
    forest.register(node_a)
    forest.register(node_b)

    x = node_a.mu.copy()
    result = make_query_result([node_a, node_b], [1.0, 1.0])

    for _ in range(6):
        forest.update(x, result)

    compression_nodes = [
        n for n in forest.active_nodes()
        if n.id not in {"a", "b"}
    ]
    assert len(compression_nodes) == 1
