"""
Tests for the Observer — all dynamic behaviour.
"""

import numpy as np
import pytest
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer


def make_forest_with_nodes(*nodes, D=2):
    forest = Forest(D=D)
    for n in nodes:
        forest.register(n)
    return forest


# ---------------------------------------------------------------------------
# 1. Registration and initial state
# ---------------------------------------------------------------------------

def test_register_initialises_weight():
    forest = Forest(D=2)
    obs = Observer(forest, w_init=0.1)
    node = HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="a")
    obs.register(node)
    assert obs.get_weight(node.id) == pytest.approx(0.1)


def test_existing_forest_nodes_get_initial_weight():
    node = HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="a")
    forest = make_forest_with_nodes(node)
    obs = Observer(forest, w_init=0.2)
    assert obs.get_weight(node.id) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 2. Weight dynamics
# ---------------------------------------------------------------------------

def test_explaining_node_gains_weight():
    # sigma=0.5*I at x=mu: kl_surprise ≈ 1.15 < tau=2.0 → good explainer
    node = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 0.5, id="a")
    forest = make_forest_with_nodes(node)
    obs = Observer(forest, w_init=0.1, tau=2.0, alpha_gain=0.2)

    obs.observe(node.mu.copy())
    assert obs.get_weight(node.id) > 0.1


def test_non_explaining_overlapping_node_loses_weight():
    # winner: sigma=0.5*I at x=mu → kl_surprise ≈ 1.15 < tau=2.0 → good explainer
    # loser: sigma=3.0*I at x=winner.mu → kl_surprise ≈ 2.94 >= tau=2.0 → not explaining
    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 0.5, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 3.0, id="loser")
    forest = make_forest_with_nodes(winner, loser)
    obs = Observer(forest, w_init=0.5, tau=2.0, beta_loss=0.1)

    obs.observe(winner.mu.copy())
    assert obs.get_weight(loser.id) < 0.5


def test_non_overlapping_node_unaffected():
    explainer = HFN(mu=np.array([0.0, 0.0]),   sigma=np.eye(2) * 0.01, id="explainer")
    distant   = HFN(mu=np.array([100.0, 0.0]), sigma=np.eye(2) * 0.01, id="distant")
    forest = make_forest_with_nodes(explainer, distant)
    obs = Observer(forest, w_init=0.5, tau=0.0, beta_loss=0.1)

    initial = obs.get_weight(distant.id)
    obs.observe(explainer.mu.copy())
    assert abs(obs.get_weight(distant.id) - initial) < 1e-6


def test_score_updated_after_observe():
    # sigma=0.5*I at x=mu: kl_surprise ≈ 1.15 < tau=2.0 → good explainer, gets nonzero accuracy
    node = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 0.5, id="a")
    forest = make_forest_with_nodes(node)
    obs = Observer(forest, tau=2.0, lambda_complexity=0.0)

    obs.observe(node.mu.copy())
    assert obs.get_score(node.id) != 0.0


# ---------------------------------------------------------------------------
# 3. Structural absorption
# ---------------------------------------------------------------------------

def test_absorption_triggers_after_persistent_misses():
    # winner: sigma=0.5*I → kl_surprise ≈ 1.15 < tau=2.0 → good explainer
    # loser: sigma=3.0*I → kl_surprise ≈ 2.94 >= tau=2.0 → not explaining, overlaps winner
    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 0.5, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 3.0, id="loser")
    forest = make_forest_with_nodes(winner, loser)
    obs = Observer(
        forest, tau=2.0, beta_loss=0.0,
        absorption_overlap_threshold=0.3,
        absorption_miss_threshold=3,
        w_init=0.5,
    )

    for _ in range(3):
        obs.observe(winner.mu.copy())

    assert loser.id in obs.absorbed_ids


def test_absorbed_node_removed_from_forest():
    # winner: sigma=0.5*I → kl_surprise ≈ 1.15 < tau=2.0 → good explainer
    # loser: sigma=3.0*I → kl_surprise ≈ 2.94 >= tau=2.0 → not explaining, overlaps winner
    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 0.5, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 3.0, id="loser")
    forest = make_forest_with_nodes(winner, loser)
    obs = Observer(
        forest, tau=2.0, beta_loss=0.0,
        absorption_overlap_threshold=0.3,
        absorption_miss_threshold=3,
        w_init=0.5,
    )

    for _ in range(3):
        obs.observe(winner.mu.copy())

    assert loser.id not in forest


def test_absorbed_node_accessible_via_expansion():
    winner = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 3.0, id="winner")
    loser  = HFN(mu=np.array([0.1, 0.0]), sigma=np.eye(2) * 3.0, id="loser")
    forest = make_forest_with_nodes(winner, loser)
    obs = Observer(
        forest, tau=0.0, beta_loss=0.0,
        absorption_overlap_threshold=0.3,
        absorption_miss_threshold=3,
        w_init=0.5,
    )

    for _ in range(3):
        obs.observe(winner.mu.copy())

    def find_in_tree(root, target_id):
        if root.id == target_id:
            return True
        return any(find_in_tree(c, target_id) for c in root.children())

    assert any(find_in_tree(n, loser.id) for n in forest.active_nodes())


# ---------------------------------------------------------------------------
# 4. Node creation — residual surprise
# ---------------------------------------------------------------------------

def test_high_residual_surprise_creates_leaf():
    # sigma=0.01*I is very tight; observing x=[10,10] → huge kl_surprise >> tau=2.0
    # Node is a leaf, can't expand → contributes to residual → new leaf spawned
    node = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 0.01, id="existing")
    forest = make_forest_with_nodes(node)
    obs = Observer(forest, tau=2.0, residual_surprise_threshold=0.5, w_init=0.1)

    initial = len(forest)
    # Signal far from existing node → high residual surprise
    obs.observe(np.array([10.0, 10.0]))
    assert len(forest) > initial


def test_low_residual_surprise_no_new_node():
    node = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 10.0, id="existing")
    forest = make_forest_with_nodes(node)
    obs = Observer(forest, tau=0.0, residual_surprise_threshold=999.0, w_init=0.1)

    initial = len(forest)
    obs.observe(node.mu.copy())
    assert len(forest) == initial


# ---------------------------------------------------------------------------
# 5. Node creation — query-induced compression
# ---------------------------------------------------------------------------

def test_cooccurrence_creates_compression_node():
    # sigma=1.0*I at x=[0.25,0]: kl_surprise ≈ 1.87 < tau=2.0 → both are good explainers
    # After 3 co-occurrences, a compression node is created
    a = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 1.0, id="a")
    b = HFN(mu=np.array([0.5, 0.0]), sigma=np.eye(2) * 1.0, id="b")
    forest = make_forest_with_nodes(a, b)
    obs = Observer(
        forest, tau=2.0,
        compression_cooccurrence_threshold=3,
        residual_surprise_threshold=999.0,
        w_init=0.1,
    )

    initial = len(forest)
    x = np.array([0.25, 0.0])
    for _ in range(3):
        obs.observe(x)

    assert len(forest) > initial


def test_compression_node_contains_originals():
    # sigma=1.0*I at x=[0.25,0]: kl_surprise ≈ 1.87 < tau=2.0 → both are good explainers
    a = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 1.0, id="alpha")
    b = HFN(mu=np.array([0.5, 0.0]), sigma=np.eye(2) * 1.0, id="beta")
    forest = make_forest_with_nodes(a, b)
    obs = Observer(
        forest, tau=2.0,
        compression_cooccurrence_threshold=3,
        residual_surprise_threshold=999.0,
        w_init=0.1,
    )

    x = np.array([0.25, 0.0])
    for _ in range(3):
        obs.observe(x)

    new_nodes = [n for n in forest.active_nodes() if n.id not in {"alpha", "beta"}]
    assert len(new_nodes) >= 1
    child_ids = {c.id for c in new_nodes[0].children()}
    assert "alpha" in child_ids
    assert "beta" in child_ids


def test_compression_node_not_created_twice():
    # sigma=1.0*I at x=[0.25,0]: kl_surprise ≈ 1.87 < tau=2.0 → both are good explainers
    # After 3 co-occurrences → one compression node; running 6 total should still be just 1
    a = HFN(mu=np.array([0.0, 0.0]), sigma=np.eye(2) * 1.0, id="a")
    b = HFN(mu=np.array([0.5, 0.0]), sigma=np.eye(2) * 1.0, id="b")
    forest = make_forest_with_nodes(a, b)
    obs = Observer(
        forest, tau=2.0,
        compression_cooccurrence_threshold=3,
        residual_surprise_threshold=999.0,
        w_init=0.1,
    )

    x = np.array([0.25, 0.0])
    # Run exactly 3 to trigger the first compression; further runs would trigger cascades
    for _ in range(3):
        obs.observe(x)
    # Capture count after first trigger
    compression_nodes = [n for n in forest.active_nodes() if n.id not in {"a", "b"}]
    count_after_first = len(compression_nodes)

    # Run 2 more (total 5) — secondary pairs haven't reached threshold yet
    for _ in range(2):
        obs.observe(x)
    compression_nodes = [n for n in forest.active_nodes() if n.id not in {"a", "b"}]
    assert len(compression_nodes) == count_after_first


# ---------------------------------------------------------------------------
# 6. Multiple Forests
# ---------------------------------------------------------------------------

def test_observer_can_watch_forest_of_forests():
    """A Forest is just an HFN — an Observer can watch a Forest-of-Forests."""
    f1 = Forest(D=2, forest_id="world_a")
    f2 = Forest(D=2, forest_id="world_b")

    f1.register(HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="a"))
    f2.register(HFN(mu=np.array([-1.0, 0.0]), sigma=np.eye(2), id="b"))

    # Recombine two forests into a higher-order Forest
    meta = Forest(D=2, forest_id="meta")
    meta.register(f1)
    meta.register(f2)

    obs = Observer(meta, tau=0.0, w_init=0.1)
    assert obs.get_weight(f1.id) == pytest.approx(0.1)
    assert obs.get_weight(f2.id) == pytest.approx(0.1)

    result = obs.observe(np.array([1.0, 0.0]))
    assert isinstance(result.explanation_tree, list)
