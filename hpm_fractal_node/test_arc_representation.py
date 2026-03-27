"""
ARC-AGI-2 structural representation test.

Goal: show that complex ARC puzzle patterns can be decomposed into HFN
structures, and that shared sub-patterns across different puzzles reuse
the same node — not a copy.

Puzzles used:
  007bbfb7 — self-tiling: the 3x3 input is placed at positions where
              the input itself is non-zero. Input is BOTH tile and mask.
  0692e18c — shape-guided tiling: a shape (diagonal/cross) determines
              where copies of the pattern are placed in a 9x9 output.

Shared sub-pattern: spatial_tiling — the operation of placing a tile
pattern at positions defined by a placement mask.

This mirrors the catch/throw ball test: different top-level concepts
sharing a common structural node.
"""

import numpy as np
import pytest
from hpm_fractal_node.hfn import HFN
from hpm_fractal_node.forest import Forest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grid_to_vec(grid: list[list[int]]) -> np.ndarray:
    """Flatten a 2D grid to a 1D float vector."""
    return np.array(grid, dtype=float).flatten()


def make_node(name: str, vec: np.ndarray, *children: HFN) -> HFN:
    D = len(vec)
    node = HFN(mu=vec, sigma=np.eye(D), id=name)
    for c in children:
        node._children.append(c)
    return node


# ---------------------------------------------------------------------------
# Shared sub-pattern nodes (built once, reused across both puzzles)
# ---------------------------------------------------------------------------

# A single coloured cell: encoded as [colour, row, col]
colour_cell = HFN(mu=np.array([6.0, 0.0, 2.0]), sigma=np.eye(3), id="colour_cell")

# A nonzero position marker: encodes the idea of "a cell that triggers placement"
nonzero_position = HFN(mu=np.array([1.0, 0.0, 0.0]), sigma=np.eye(3), id="nonzero_position")

# Spatial tiling: the operation of placing a tile at positions defined by a mask
# Encoded as a feature vector: [has_tile=1, has_mask=1, scale_factor=3, self_referential=0]
spatial_tiling = HFN(
    mu=np.array([1.0, 1.0, 3.0, 0.0]),
    sigma=np.eye(4),
    id="spatial_tiling",
)
spatial_tiling._children.append(nonzero_position)
spatial_tiling._children.append(colour_cell)


# ---------------------------------------------------------------------------
# Puzzle 007bbfb7 — self-tiling rule
#
# The 3x3 input is BOTH the tile content and the placement mask.
# Output is 9x9: a copy of the input appears at each non-zero cell's position.
# ---------------------------------------------------------------------------

# Concrete input grid for one training example
grid_007 = [[6, 6, 0], [6, 0, 0], [0, 6, 6]]
input_pattern_007 = HFN(
    mu=grid_to_vec(grid_007),
    sigma=np.eye(9),
    id="input_pattern_007",
)

# Self-referential tiling: the tile IS the mask
# Feature: [has_tile=1, has_mask=1, scale_factor=3, self_referential=1]
self_ref_tiling = HFN(
    mu=np.array([1.0, 1.0, 3.0, 1.0]),
    sigma=np.eye(4),
    id="self_referential_tiling",
)
self_ref_tiling._children.append(spatial_tiling)   # reuses shared node
self_ref_tiling._children.append(input_pattern_007)

puzzle_007 = make_node(
    "puzzle_007bbfb7",
    np.array([0.0, 7.0, 0.0, 7.0]),  # summary feature vector
    self_ref_tiling,
    input_pattern_007,
)


# ---------------------------------------------------------------------------
# Puzzle 0692e18c — shape-guided tiling rule
#
# A shape (e.g. diagonal, cross) determines WHERE copies of the pattern
# are placed. Tile ≠ mask — the shape is the mask, not the content.
# ---------------------------------------------------------------------------

grid_0692 = [[0, 0, 6], [0, 6, 0], [6, 0, 0]]  # diagonal shape as mask
input_pattern_0692 = HFN(
    mu=grid_to_vec(grid_0692),
    sigma=np.eye(9),
    id="input_pattern_0692",
)

# Shape-guided tiling: uses shape to control placement
# Feature: [has_tile=1, has_mask=1, scale_factor=3, self_referential=0]
shape_guided_tiling = HFN(
    mu=np.array([1.0, 1.0, 3.0, 0.0]),
    sigma=np.eye(4),
    id="shape_guided_tiling",
)
shape_guided_tiling._children.append(spatial_tiling)   # reuses the same shared node
shape_guided_tiling._children.append(input_pattern_0692)

puzzle_0692 = make_node(
    "puzzle_0692e18c",
    np.array([0.0, 6.9, 0.0, 0.0]),
    shape_guided_tiling,
    input_pattern_0692,
)


# ---------------------------------------------------------------------------
# Register everything in a Forest
# ---------------------------------------------------------------------------

forest = Forest(D=4, forest_id="arc_world_model")
forest.register(puzzle_007)
forest.register(puzzle_0692)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def find_in_tree(root: HFN, target_id: str) -> bool:
    if root.id == target_id:
        return True
    return any(find_in_tree(c, target_id) for c in root.children())


def test_both_puzzles_registered():
    assert "puzzle_007bbfb7" in forest
    assert "puzzle_0692e18c" in forest


def test_spatial_tiling_reachable_from_007():
    """The spatial_tiling node is reachable from puzzle 007bbfb7."""
    assert find_in_tree(puzzle_007, "spatial_tiling")


def test_spatial_tiling_reachable_from_0692():
    """The spatial_tiling node is reachable from puzzle 0692e18c."""
    assert find_in_tree(puzzle_0692, "spatial_tiling")


def test_spatial_tiling_is_same_node_not_copy():
    """
    The spatial_tiling node in both puzzle trees is the SAME object.
    Different puzzles don't each carry their own copy of the sub-pattern —
    they share a single node. This is structural compression.
    """
    def find_node(root: HFN, target_id: str) -> HFN | None:
        if root.id == target_id:
            return root
        for c in root.children():
            found = find_node(c, target_id)
            if found:
                return found
        return None

    tiling_in_007 = find_node(puzzle_007, "spatial_tiling")
    tiling_in_0692 = find_node(puzzle_0692, "spatial_tiling")

    assert tiling_in_007 is not None
    assert tiling_in_0692 is not None
    assert tiling_in_007 is tiling_in_0692


def test_self_ref_tiling_differs_from_shape_guided():
    """
    The two puzzle-specific tiling nodes are distinct: self-referential
    tiling (tile == mask) is a different concept from shape-guided tiling
    (tile != mask), even though both use spatial_tiling underneath.
    """
    assert self_ref_tiling is not shape_guided_tiling
    assert self_ref_tiling.id != shape_guided_tiling.id


def test_colour_cell_reachable_through_both_paths():
    """
    colour_cell is accessible from both puzzles through the shared
    spatial_tiling node. One leaf, multiple paths.
    """
    assert find_in_tree(puzzle_007, "colour_cell")
    assert find_in_tree(puzzle_0692, "colour_cell")


def test_puzzle_hierarchy_depth():
    """
    Puzzle → puzzle-specific rule → spatial_tiling → nonzero_position.
    Four levels of hierarchy, all structurally navigable.
    """
    # Level 1: forest contains puzzles
    assert puzzle_007 in forest.children()
    # Level 2: puzzle contains puzzle-specific tiling rule
    assert self_ref_tiling in puzzle_007.children()
    # Level 3: puzzle-specific rule contains shared spatial_tiling
    assert spatial_tiling in self_ref_tiling.children()
    # Level 4: spatial_tiling contains nonzero_position
    assert nonzero_position in spatial_tiling.children()


def test_forest_is_hfn_can_be_registered_elsewhere():
    """
    The arc_world_model Forest is just an HFN — it can become a child
    of a higher-order node if more is learned. No fixed role.
    """
    meta = Forest(D=4, forest_id="meta_world")
    meta.register(forest)
    assert forest in meta.children()
