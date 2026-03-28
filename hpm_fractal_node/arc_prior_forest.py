"""
ARC prior Forest — pre-populated world model for 3x3 ARC grids.

All priors are HFN nodes. The hierarchy:

  prior_spatial_pattern
    ├── full_grid
    ├── diagonal_tlbr / diagonal_trbl
    ├── cross
    ├── border
    └── single_cell
          └── cell_00 … cell_22  (L1 leaves)

  prior_transformation
    ├── prior_tiling           (place copies at positions defined by a mask)
    ├── prior_rotation         (90° rotation of shape)
    ├── prior_reflection       (horizontal / vertical flip)
    └── prior_colour_subst     (same shape, different colour → same binary)

  prior_relationship
    ├── prior_shape_guides_placement  (mask ≠ tile)
    └── prior_self_ref_tiling         (mask == tile)

No distinction between "prior" and "pattern" in the HFN sense — these are
just nodes registered before any observations. Learning refines them.
"""

from __future__ import annotations

import numpy as np
from hpm_fractal_node.hfn import HFN
from hpm_fractal_node.forest import Forest


D = 9
CELL_NAMES = [f"cell_{r}{c}" for r in range(3) for c in range(3)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(*cells: int) -> np.ndarray:
    """Build a 9-dim binary vector from cell indices (0-8)."""
    v = np.zeros(D)
    for i in cells:
        v[i] = 1.0
    return v


def _node(name: str, mu: np.ndarray, *children: HFN, sigma_scale: float = 1.0) -> HFN:
    n = HFN(mu=mu, sigma=np.eye(D) * sigma_scale, id=name)
    for c in children:
        n._children.append(c)
    return n


def _centroid(*nodes: HFN) -> np.ndarray:
    return np.mean([n.mu for n in nodes], axis=0)


# ---------------------------------------------------------------------------
# L1 cell leaves
# ---------------------------------------------------------------------------

def make_cell_leaves() -> dict[str, HFN]:
    cells = {}
    for i, name in enumerate(CELL_NAMES):
        cells[name] = _node(name, _vec(i))
    return cells


# ---------------------------------------------------------------------------
# Build prior Forest
# ---------------------------------------------------------------------------

def build_prior_forest() -> tuple[Forest, dict[str, HFN]]:
    """
    Returns (forest, node_registry) where node_registry maps name → HFN
    for all prior nodes, including those that are children only.
    """
    registry: dict[str, HFN] = {}

    # --- L1 cell leaves ---
    cells = make_cell_leaves()
    registry.update(cells)
    c = cells  # shorthand

    # --- Spatial shape priors ---

    full_grid = _node(
        "full_grid",
        _vec(*range(9)),
        *cells.values(),
    )

    diagonal_tlbr = _node(
        "diagonal_tlbr",
        _vec(0, 4, 8),  # (0,0) (1,1) (2,2)
        c["cell_00"], c["cell_11"], c["cell_22"],
    )

    diagonal_trbl = _node(
        "diagonal_trbl",
        _vec(2, 4, 6),  # (0,2) (1,1) (2,0)
        c["cell_02"], c["cell_11"], c["cell_20"],
    )

    cross = _node(
        "cross",
        _vec(1, 3, 4, 5, 7),  # centre row + centre col
        c["cell_01"], c["cell_10"], c["cell_11"], c["cell_12"], c["cell_21"],
    )

    border = _node(
        "border",
        _vec(0, 1, 2, 3, 5, 6, 7, 8),  # all except centre
        c["cell_00"], c["cell_01"], c["cell_02"],
        c["cell_10"], c["cell_12"],
        c["cell_20"], c["cell_21"], c["cell_22"],
    )

    # single_cell: abstract concept — mu is centroid of all unit vectors = uniform
    single_cell = _node(
        "single_cell",
        np.full(D, 1.0 / D),
        *cells.values(),
        sigma_scale=2.0,  # broad: covers any single-cell pattern
    )

    prior_spatial = _node(
        "prior_spatial_pattern",
        _centroid(full_grid, diagonal_tlbr, diagonal_trbl, cross, border, single_cell),
        full_grid, diagonal_tlbr, diagonal_trbl, cross, border, single_cell,
        sigma_scale=2.0,
    )

    for n in [full_grid, diagonal_tlbr, diagonal_trbl, cross, border, single_cell, prior_spatial]:
        registry[n.id] = n

    # --- Transformation priors ---
    # Encoded in binary 9-dim space as the "input side" of the transformation.
    # A tiling transformation takes a sparse input and produces a dense output.
    # The prior captures what inputs look like, not what outputs look like.

    prior_tiling = _node(
        "prior_tiling",
        _centroid(diagonal_tlbr, cross),   # sparse inputs are typical tiling inputs
        diagonal_tlbr, diagonal_trbl, cross,
        sigma_scale=1.5,
    )

    # Rotation: a rotated pattern has the same density as the original.
    # Prior mu = average of diagonal variants (rotation maps one to the other).
    prior_rotation = _node(
        "prior_rotation",
        _centroid(diagonal_tlbr, diagonal_trbl),
        diagonal_tlbr, diagonal_trbl,
        sigma_scale=1.5,
    )

    # Reflection: similar to rotation — same shapes, flipped.
    prior_reflection = _node(
        "prior_reflection",
        _centroid(diagonal_tlbr, diagonal_trbl, cross),
        diagonal_tlbr, diagonal_trbl, cross,
        sigma_scale=1.5,
    )

    # Colour substitution: in binary space, colour-substituted grids look identical.
    # This prior represents "any shape, colour doesn't matter" — broad sigma.
    prior_colour_subst = _node(
        "prior_colour_subst",
        _centroid(diagonal_tlbr, cross, full_grid),
        diagonal_tlbr, cross, full_grid,
        sigma_scale=2.5,
    )

    prior_transformation = _node(
        "prior_transformation",
        _centroid(prior_tiling, prior_rotation, prior_reflection, prior_colour_subst),
        prior_tiling, prior_rotation, prior_reflection, prior_colour_subst,
        sigma_scale=2.5,
    )

    for n in [prior_tiling, prior_rotation, prior_reflection,
              prior_colour_subst, prior_transformation]:
        registry[n.id] = n

    # --- Relationship priors ---
    # These capture structural relationships between concepts.
    # In 9-dim space they're represented as the "signature" of the relationship —
    # the typical input pattern for which this relationship holds.

    # Shape guides placement: mask ≠ tile. Sparse shapes (diagonals, cross) guide where
    # copies of a pattern are placed. Typical input: sparse shape.
    prior_shape_guides = _node(
        "prior_shape_guides_placement",
        _centroid(diagonal_tlbr, diagonal_trbl, cross),
        diagonal_tlbr, diagonal_trbl, cross,
        sigma_scale=1.5,
    )

    # Self-referential tiling: mask == tile. The input IS the placement map.
    # Same shapes as shape_guides but with a different conceptual role.
    prior_self_ref = _node(
        "prior_self_referential_tiling",
        _centroid(diagonal_tlbr, cross),
        diagonal_tlbr, cross,
        sigma_scale=1.5,
    )

    prior_relationship = _node(
        "prior_relationship",
        _centroid(prior_shape_guides, prior_self_ref),
        prior_shape_guides, prior_self_ref,
        sigma_scale=2.5,
    )

    for n in [prior_shape_guides, prior_self_ref, prior_relationship]:
        registry[n.id] = n

    # --- Build Forest ---
    # Register specific shape and transformation priors directly — they need
    # to compete in retrieval and can explain observations (sigma=I, baseline<tau).
    # Group nodes (prior_spatial, prior_transformation, prior_relationship) are
    # structural containers: registered so the Observer can expand into their
    # children, but they have broad sigma so they're always surprising and get
    # expanded rather than used as direct explainers.

    forest = Forest(D=D, forest_id="arc_world_model")

    # Specific shape priors — directly retrievable, can explain observations
    for node in [full_grid, diagonal_tlbr, diagonal_trbl, cross, border, single_cell]:
        forest.register(node)

    # Transformation and relationship priors — directly retrievable
    for node in [prior_tiling, prior_rotation, prior_reflection,
                 prior_colour_subst, prior_shape_guides, prior_self_ref]:
        forest.register(node)

    return forest, registry
