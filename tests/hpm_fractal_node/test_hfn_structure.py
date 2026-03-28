"""
Structural tests for the HFN data structure.

Tests the base concept only: can a complex hierarchical idea be represented
correctly using HFN nodes? No learning, compression, or evaluator logic tested.

Scenario: "catching a ball" — a 3-level HPM hierarchy from the paper.

    Level 1 (leaves):   motion_pattern, gravity_pattern, hand_position
    Level 2 (internal): trajectory_model [motion_pattern, gravity_pattern]
                        grip_action      [hand_position,  motion_pattern]
    Level 3 (root):     catch_a_ball     [trajectory_model, grip_action]

Key structural invariant under test: motion_pattern is the SAME object
in both trajectory_model and grip_action — not a copy.
"""

import numpy as np
import pytest
from hfn.hfn import HFN, Edge, make_leaf, make_parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ball_trees():
    """
    Build both catching-a-ball and throwing-a-ball hierarchies.

    motion_pattern and gravity_pattern are SHARED across both trees —
    the same objects, referenced from four different parent polygraphs.

    catch_a_ball
    ├── trajectory_model  [motion_pattern*, gravity_pattern*]
    └── grip_action       [hand_position,   motion_pattern*]

    throw_a_ball
    ├── throw_trajectory  [motion_pattern*, gravity_pattern*, target_direction]
    └── release_action    [release_timing,  motion_pattern*]

    * = same object across both trees
    """
    # Level 1 — shared leaves
    motion   = make_leaf("motion_pattern")
    gravity  = make_leaf("gravity_pattern")

    # Level 1 — catch-only leaves
    hand_pos = make_leaf("hand_position")

    # Level 1 — throw-only leaves
    target_dir     = make_leaf("target_direction")
    release_timing = make_leaf("release_timing")

    # --- catch tree ---
    trajectory = make_parent(
        "trajectory_model",
        children=[motion, gravity],
        edges=[("motion_pattern", "gravity_pattern", "constrains")],
    )
    grip = make_parent(
        "grip_action",
        children=[hand_pos, motion],
        edges=[("hand_position", "motion_pattern", "guides")],
    )
    catch = make_parent(
        "catch_a_ball",
        children=[trajectory, grip],
        edges=[("trajectory_model", "grip_action", "coordinates")],
    )

    # --- throw tree (reuses motion and gravity) ---
    throw_trajectory = make_parent(
        "throw_trajectory",
        children=[motion, gravity, target_dir],
        edges=[
            ("motion_pattern",   "gravity_pattern",  "constrains"),
            ("gravity_pattern",  "target_direction",  "shapes"),
        ],
    )
    release_action = make_parent(
        "release_action",
        children=[release_timing, motion],
        edges=[("release_timing", "motion_pattern", "triggers")],
    )
    throw = make_parent(
        "throw_a_ball",
        children=[throw_trajectory, release_action],
        edges=[("throw_trajectory", "release_action", "coordinates")],
    )

    return {
        # shared
        "motion": motion,
        "gravity": gravity,
        # catch-specific
        "catch": catch,
        "trajectory": trajectory,
        "grip": grip,
        "hand_pos": hand_pos,
        # throw-specific
        "throw": throw,
        "throw_trajectory": throw_trajectory,
        "release_action": release_action,
        "target_dir": target_dir,
        "release_timing": release_timing,
    }


@pytest.fixture
def ball_catch_tree(ball_trees):
    """Backwards-compatible fixture: catch subtree only."""
    return ball_trees


# ---------------------------------------------------------------------------
# 1. Basic structure
# ---------------------------------------------------------------------------

def test_leaves_have_no_children(ball_catch_tree):
    t = ball_catch_tree
    assert t["motion"].is_leaf()
    assert t["gravity"].is_leaf()
    assert t["hand_pos"].is_leaf()


def test_internal_nodes_have_correct_children(ball_catch_tree):
    t = ball_catch_tree
    assert len(t["trajectory"].children()) == 2
    assert len(t["grip"].children()) == 2
    assert len(t["catch"].children()) == 2


def test_root_children_are_correct(ball_catch_tree):
    t = ball_catch_tree
    root_children = t["catch"].children()
    assert t["trajectory"] in root_children
    assert t["grip"] in root_children


# ---------------------------------------------------------------------------
# 2. Shared reference — the key invariant
# ---------------------------------------------------------------------------

def test_motion_pattern_is_same_object_in_both_parents(ball_catch_tree):
    """
    motion_pattern must be the identical object in trajectory_model AND grip_action.
    Not a copy — the same node referenced from two different polygraphs.
    """
    t = ball_catch_tree
    motion_in_trajectory = t["trajectory"].children()[0]  # first child
    motion_in_grip = t["grip"].children()[1]               # second child

    assert motion_in_trajectory is t["motion"]
    assert motion_in_grip is t["motion"]
    assert motion_in_trajectory is motion_in_grip  # same object


def test_shared_node_has_no_parent_references(ball_catch_tree):
    """motion_pattern knows nothing about trajectory_model or grip_action."""
    t = ball_catch_tree
    assert not hasattr(t["motion"], "_parents")
    assert not hasattr(t["motion"], "parents")
    # The node's only knowledge is its own children (none — it's a leaf)
    assert t["motion"].children() == []


# ---------------------------------------------------------------------------
# 3. Fractal uniformity — same interface at every depth
# ---------------------------------------------------------------------------

def test_expand_depth_zero_returns_self(ball_catch_tree):
    t = ball_catch_tree
    for node in t.values():
        assert node.expand(0) is node


def test_expand_on_leaf_always_returns_self(ball_catch_tree):
    t = ball_catch_tree
    for depth in [0, 1, 2, 5]:
        assert t["motion"].expand(depth) is t["motion"]
        assert t["gravity"].expand(depth) is t["gravity"]


def test_children_of_expansion_are_hfn_nodes(ball_catch_tree):
    """At every depth, children() returns HFN instances — fractal uniformity."""
    t = ball_catch_tree
    root = t["catch"]

    # depth 1: children of root
    for child in root.children():
        assert isinstance(child, HFN)

    # depth 2: grandchildren
    for child in root.children():
        for grandchild in child.children():
            assert isinstance(grandchild, HFN)


def test_recursive_traversal_reaches_all_leaves(ball_catch_tree):
    """Walking expand(1) recursively from root reaches all leaf nodes."""
    t = ball_catch_tree

    def collect_leaves(node):
        if node.is_leaf():
            return {node.id}
        result = set()
        for child in node.children():
            result |= collect_leaves(child)
        return result

    leaves = collect_leaves(t["catch"])
    assert "motion_pattern" in leaves
    assert "gravity_pattern" in leaves
    assert "hand_position" in leaves


# ---------------------------------------------------------------------------
# 4. Edges
# ---------------------------------------------------------------------------

def test_edges_reference_actual_children(ball_catch_tree):
    """Every edge source and target must be a child of the containing node."""
    t = ball_catch_tree
    for node in [t["trajectory"], t["grip"], t["catch"]]:
        children_set = set(node.children())
        for edge in node.edges():
            assert isinstance(edge, Edge)
            assert edge.source in children_set
            assert edge.target in children_set


def test_edge_relations_are_strings(ball_catch_tree):
    t = ball_catch_tree
    for node in [t["trajectory"], t["grip"], t["catch"]]:
        for edge in node.edges():
            assert isinstance(edge.relation, str)
            assert len(edge.relation) > 0


# ---------------------------------------------------------------------------
# 5. Immutability under query
# ---------------------------------------------------------------------------

def test_querying_does_not_mutate_nodes(ball_catch_tree):
    t = ball_catch_tree
    motion = t["motion"]

    original_mu = motion.mu.copy()
    original_id = motion.id

    # Run every read operation
    _ = motion.log_prob(np.zeros_like(motion.mu))
    _ = motion.overlap(t["gravity"])
    _ = motion.description_length()
    _ = motion.children()
    _ = motion.edges()
    _ = motion.expand(0)
    _ = motion.expand(1)

    assert np.array_equal(motion.mu, original_mu)
    assert motion.id == original_id


# ---------------------------------------------------------------------------
# 6. Recombination produces a new node, no mutation
# ---------------------------------------------------------------------------

def test_recombine_produces_new_node(ball_catch_tree):
    t = ball_catch_tree
    new_node = t["trajectory"].recombine(t["grip"])

    assert new_node is not t["trajectory"]
    assert new_node is not t["grip"]
    assert isinstance(new_node, HFN)


def test_recombine_children_are_original_nodes(ball_catch_tree):
    t = ball_catch_tree
    new_node = t["trajectory"].recombine(t["grip"])

    assert t["trajectory"] in new_node.children()
    assert t["grip"] in new_node.children()


def test_recombine_does_not_mutate_inputs(ball_catch_tree):
    t = ball_catch_tree
    traj_children_before = t["trajectory"].children()
    grip_children_before = t["grip"].children()

    _ = t["trajectory"].recombine(t["grip"])

    assert t["trajectory"].children() == traj_children_before
    assert t["grip"].children() == grip_children_before


# ---------------------------------------------------------------------------
# 7. Gaussian identity is present on every node
# ---------------------------------------------------------------------------

def test_every_node_has_gaussian(ball_catch_tree):
    t = ball_catch_tree
    for node in t.values():
        assert isinstance(node.mu, np.ndarray)
        assert isinstance(node.sigma, np.ndarray)
        assert node.mu.ndim == 1
        assert node.sigma.ndim == 2
        assert node.sigma.shape == (len(node.mu), len(node.mu))


def test_log_prob_returns_scalar(ball_catch_tree):
    t = ball_catch_tree
    for node in t.values():
        x = np.zeros_like(node.mu)
        result = node.log_prob(x)
        assert isinstance(result, float)


def test_overlap_returns_scalar_between_zero_and_one(ball_catch_tree):
    t = ball_catch_tree
    val = t["trajectory"].overlap(t["grip"])
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# 8. Cross-tree reuse — throw_a_ball shares nodes with catch_a_ball
# ---------------------------------------------------------------------------

def test_motion_shared_across_both_trees(ball_trees):
    """motion_pattern is the same object in both catch and throw hierarchies."""
    t = ball_trees
    motion_in_catch_trajectory = t["trajectory"].children()[0]
    motion_in_catch_grip       = t["grip"].children()[1]
    motion_in_throw_trajectory = t["throw_trajectory"].children()[0]
    motion_in_throw_release    = t["release_action"].children()[1]

    assert motion_in_catch_trajectory is t["motion"]
    assert motion_in_catch_grip       is t["motion"]
    assert motion_in_throw_trajectory is t["motion"]
    assert motion_in_throw_release    is t["motion"]


def test_gravity_shared_across_both_trees(ball_trees):
    """gravity_pattern is the same object in both catch and throw hierarchies."""
    t = ball_trees
    gravity_in_catch = t["trajectory"].children()[1]
    gravity_in_throw = t["throw_trajectory"].children()[1]

    assert gravity_in_catch is t["gravity"]
    assert gravity_in_throw is t["gravity"]
    assert gravity_in_catch is gravity_in_throw


def test_shared_nodes_have_no_parent_references(ball_trees):
    """Despite being in four polygraphs, motion_pattern knows none of its parents."""
    t = ball_trees
    assert not hasattr(t["motion"], "_parents")
    assert not hasattr(t["motion"], "parents")
    assert t["motion"].children() == []


def test_catch_tree_traversal_unaffected_by_throw_tree(ball_trees):
    """Leaves reachable from catch_a_ball are exactly the catch leaves."""
    t = ball_trees

    def collect_ids(node):
        if node.is_leaf():
            return {node.id}
        result = set()
        for child in node.children():
            result |= collect_ids(child)
        return result

    catch_leaves = collect_ids(t["catch"])
    throw_leaves = collect_ids(t["throw"])

    # catch has its own leaves
    assert "motion_pattern"   in catch_leaves
    assert "gravity_pattern"  in catch_leaves
    assert "hand_position"    in catch_leaves

    # throw has its own leaves
    assert "motion_pattern"    in throw_leaves
    assert "gravity_pattern"   in throw_leaves
    assert "target_direction"  in throw_leaves
    assert "release_timing"    in throw_leaves

    # catch does NOT reach throw-specific leaves
    assert "target_direction" not in catch_leaves
    assert "release_timing"   not in catch_leaves

    # throw does NOT reach catch-specific leaves
    assert "hand_position" not in throw_leaves


def test_roots_are_independent(ball_trees):
    """catch_a_ball and throw_a_ball are separate roots — neither contains the other."""
    t = ball_trees
    assert t["throw"] not in t["catch"].children()
    assert t["catch"] not in t["throw"].children()

    def all_nodes(node, visited=None):
        if visited is None:
            visited = set()
        if node.id in visited:
            return visited
        visited.add(node.id)
        for child in node.children():
            all_nodes(child, visited)
        return visited

    catch_ids = all_nodes(t["catch"])
    throw_ids = all_nodes(t["throw"])

    # roots are not in each other's trees
    assert t["throw"].id not in catch_ids
    assert t["catch"].id not in throw_ids

    # shared nodes appear in both
    assert t["motion"].id in catch_ids
    assert t["motion"].id in throw_ids
