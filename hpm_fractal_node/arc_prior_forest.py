"""
ARC prior Forest — conceptual-level world model for ARC grids.

Priors are concepts, not specific patterns. They represent categories of
things that exist in the domain. Specific patterns (diagonal, cross, etc.)
are what the Observer learns within these categories — not what it starts with.

Prior hierarchy:

  prior_grid                    (root frame: this observation is a grid)
    ├── prior_extent             (grid has dimensions; all density is relative to these)
    ├── prior_density            (fill level relative to extent)
    │     ├── prior_sparse       (~0-30% cells filled)
    │     ├── prior_medium       (~30-70% cells filled)
    │     └── prior_dense        (~70-100% cells filled)
    └── prior_spatial_organisation  (cells have positional meaning within extent)
          ├── prior_row_band     (horizontal organisation: top / mid / bottom row)
          ├── prior_col_band     (vertical organisation: left / mid / right col)
          ├── prior_diagonal     (diagonal orientation: either diagonal)
          └── prior_corners      (corner positions as salient locations)

  prior_structure               (how filled cells relate to each other)
    ├── prior_connectivity       (cells form connected regions)
    ├── prior_symmetry           (pattern invariant under some transformation)
    └── prior_boundary           (cells on perimeter vs interior)

  prior_colour                  (cells have value-identity beyond presence/absence)

  prior_grid_transform          (input and output grids are related by a rule)
    ├── prior_size_preserving    (output dimensions match input)
    ├── prior_size_changing      (output dimensions differ: scale, crop, tile)
    └── prior_content_transform  (same shape, different values)

  prior_transformation          (categories of transformation rule)
    ├── prior_placement          (copy pattern to positions)
    ├── prior_substitution       (replace values while preserving shape)
    └── prior_geometric          (rotate, reflect, scale)

None of these are specific instances. They define what kinds of properties
and relationships can exist in the ARC world.

All priors are parameterised by grid size (rows, cols) so the same
conceptual ontology applies to any grid dimension.
"""

from __future__ import annotations

import numpy as np
from hpm_fractal_node.hfn import HFN
from hpm_fractal_node.forest import Forest


# Default grid size (kept for backwards compatibility)
D = 9
CELL_NAMES = [f"cell_{r}{c}" for r in range(3) for c in range(3)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(name: str, mu: np.ndarray, *children: HFN, sigma_scale: float = 1.0) -> HFN:
    D = mu.shape[0]
    n = HFN(mu=mu, sigma=np.eye(D) * sigma_scale, id=name)
    for c in children:
        n._children.append(c)
    return n


def _centroid(*nodes: HFN) -> np.ndarray:
    return np.mean([n.mu for n in nodes], axis=0)


def _uniform(value: float, D: int) -> np.ndarray:
    return np.full(D, value)


def _grid_mu(rows: int, cols: int, fn) -> np.ndarray:
    """Build a mu vector by applying fn(r, c) for each cell position."""
    return np.array([fn(r, c) for r in range(rows) for c in range(cols)], dtype=float)


# ---------------------------------------------------------------------------
# Grid priors
# ---------------------------------------------------------------------------

def make_grid_priors(rows: int, cols: int) -> tuple[HFN, HFN]:
    D = rows * cols
    prior_extent = _node("prior_extent", _uniform(0.5, D), sigma_scale=3.0)
    prior_grid = _node("prior_grid", _uniform(0.5, D), prior_extent, sigma_scale=3.0)
    return prior_grid, prior_extent


# ---------------------------------------------------------------------------
# Density priors
#
# Sparse/medium/dense defined as proportions of total cells, not absolute
# counts, so the concept scales across grid sizes.
#   sparse:  ~20% fill  → centroid_i = 0.20
#   medium:  ~50% fill  → centroid_i = 0.50
#   dense:   ~80% fill  → centroid_i = 0.80
# ---------------------------------------------------------------------------

def make_density_priors(rows: int, cols: int) -> tuple[HFN, HFN, HFN, HFN]:
    D = rows * cols
    prior_sparse = _node("prior_sparse", _uniform(0.20, D))
    prior_medium = _node("prior_medium", _uniform(0.50, D))
    prior_dense  = _node("prior_dense",  _uniform(0.80, D))
    prior_density = _node(
        "prior_density",
        _centroid(prior_sparse, prior_medium, prior_dense),
        prior_sparse, prior_medium, prior_dense,
        sigma_scale=2.0,
    )
    return prior_density, prior_sparse, prior_medium, prior_dense


# ---------------------------------------------------------------------------
# Structure priors
# ---------------------------------------------------------------------------

def make_structure_priors(rows: int, cols: int) -> tuple[HFN, ...]:
    cr, cc = rows / 2.0, cols / 2.0  # grid centre (fractional)

    # Connectivity: weight cells by proximity to centre (Gaussian falloff).
    def connectivity_weight(r, c):
        dist = ((r - cr) / cr) ** 2 + ((c - cc) / cc) ** 2
        return float(np.exp(-dist))

    connectivity_mu = _grid_mu(rows, cols, connectivity_weight)
    # Normalise to [0.3, 0.8] range
    mn, mx = connectivity_mu.min(), connectivity_mu.max()
    connectivity_mu = 0.3 + 0.5 * (connectivity_mu - mn) / (mx - mn)
    prior_connectivity = _node("prior_connectivity", connectivity_mu)

    # Symmetry: uniform centroid (symmetric patterns span all positions equally).
    D = rows * cols
    prior_symmetry = _node("prior_symmetry", _uniform(0.5, D))

    # Boundary: perimeter cells (0.7) vs interior cells (0.3).
    def boundary_weight(r, c):
        return 0.7 if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) else 0.3

    prior_boundary = _node("prior_boundary", _grid_mu(rows, cols, boundary_weight))

    prior_structure = _node(
        "prior_structure",
        _centroid(prior_connectivity, prior_symmetry, prior_boundary),
        prior_connectivity, prior_symmetry, prior_boundary,
        sigma_scale=2.0,
    )
    return prior_structure, prior_connectivity, prior_symmetry, prior_boundary


# ---------------------------------------------------------------------------
# Colour prior
# ---------------------------------------------------------------------------

def make_colour_prior(rows: int, cols: int) -> HFN:
    D = rows * cols
    return _node("prior_colour", _uniform(0.5, D), sigma_scale=3.0)


# ---------------------------------------------------------------------------
# Spatial organisation priors
#
# Row/col bands split the grid into thirds. Weights: dominant=0.7,
# adjacent=0.3, opposite=0.1.
# Diagonal: main + anti diagonal cells weighted, centre highest.
# Corners: corner region cells weighted by distance from nearest corner.
# ---------------------------------------------------------------------------

def make_spatial_organisation_priors(rows: int, cols: int) -> tuple[HFN, ...]:
    r_third = rows / 3.0
    c_third = cols / 3.0

    def row_weight(r, band):
        # band: 0=top, 1=mid, 2=bot
        if band == 0:
            return 0.7 if r < r_third else (0.3 if r < 2 * r_third else 0.1)
        elif band == 1:
            return 0.1 if r < r_third else (0.7 if r < 2 * r_third else 0.1)
        else:
            return 0.1 if r < r_third else (0.3 if r < 2 * r_third else 0.7)

    def col_weight(c, band):
        if band == 0:
            return 0.7 if c < c_third else (0.3 if c < 2 * c_third else 0.1)
        elif band == 1:
            return 0.1 if c < c_third else (0.7 if c < 2 * c_third else 0.1)
        else:
            return 0.1 if c < c_third else (0.3 if c < 2 * c_third else 0.7)

    prior_row_top = _node("prior_row_top", _grid_mu(rows, cols, lambda r, c: row_weight(r, 0)))
    prior_row_mid = _node("prior_row_mid", _grid_mu(rows, cols, lambda r, c: row_weight(r, 1)))
    prior_row_bot = _node("prior_row_bot", _grid_mu(rows, cols, lambda r, c: row_weight(r, 2)))
    prior_row_band = _node(
        "prior_row_band",
        _centroid(prior_row_top, prior_row_mid, prior_row_bot),
        prior_row_top, prior_row_mid, prior_row_bot,
        sigma_scale=2.0,
    )

    prior_col_left  = _node("prior_col_left",  _grid_mu(rows, cols, lambda r, c: col_weight(c, 0)))
    prior_col_mid   = _node("prior_col_mid",   _grid_mu(rows, cols, lambda r, c: col_weight(c, 1)))
    prior_col_right = _node("prior_col_right", _grid_mu(rows, cols, lambda r, c: col_weight(c, 2)))
    prior_col_band = _node(
        "prior_col_band",
        _centroid(prior_col_left, prior_col_mid, prior_col_right),
        prior_col_left, prior_col_mid, prior_col_right,
        sigma_scale=2.0,
    )

    # Diagonal: both main (r==c scaled) and anti diagonal weighted.
    # Use normalised distance to nearest diagonal as the weight.
    def diagonal_weight(r, c):
        nr = r / (rows - 1) if rows > 1 else 0.5
        nc = c / (cols - 1) if cols > 1 else 0.5
        d_main = abs(nr - nc)           # 0 on main diagonal
        d_anti = abs(nr + nc - 1.0)     # 0 on anti diagonal
        d_nearest = min(d_main, d_anti)
        return float(np.exp(-4 * d_nearest))  # falloff

    diag_mu = _grid_mu(rows, cols, diagonal_weight)
    mn, mx = diag_mu.min(), diag_mu.max()
    diag_mu = 0.0 + 1.0 * (diag_mu - mn) / (mx - mn)
    prior_diagonal = _node("prior_diagonal", diag_mu)

    # Corners: weight cells by proximity to nearest corner.
    def corner_weight(r, c):
        nr = r / (rows - 1) if rows > 1 else 0.5
        nc = c / (cols - 1) if cols > 1 else 0.5
        d = min(
            nr**2 + nc**2,
            nr**2 + (1 - nc)**2,
            (1 - nr)**2 + nc**2,
            (1 - nr)**2 + (1 - nc)**2,
        )
        return float(np.exp(-4 * d))

    corner_mu = _grid_mu(rows, cols, corner_weight)
    mn, mx = corner_mu.min(), corner_mu.max()
    corner_mu = 0.1 + 0.6 * (corner_mu - mn) / (mx - mn)
    prior_corners = _node("prior_corners", corner_mu)

    prior_spatial_organisation = _node(
        "prior_spatial_organisation",
        _centroid(prior_row_band, prior_col_band, prior_diagonal, prior_corners),
        prior_row_band, prior_col_band, prior_diagonal, prior_corners,
        sigma_scale=2.0,
    )
    return (
        prior_spatial_organisation,
        prior_row_band, prior_row_top, prior_row_mid, prior_row_bot,
        prior_col_band, prior_col_left, prior_col_mid, prior_col_right,
        prior_diagonal, prior_corners,
    )


# ---------------------------------------------------------------------------
# Grid transform priors
# ---------------------------------------------------------------------------

def make_grid_transform_priors(rows: int, cols: int) -> tuple[HFN, ...]:
    D = rows * cols
    prior_size_preserving  = _node("prior_size_preserving",  _uniform(0.5, D))
    prior_size_changing    = _node("prior_size_changing",    _uniform(0.20, D))
    prior_content_transform = _node("prior_content_transform", _uniform(0.5, D), sigma_scale=3.0)
    prior_grid_transform = _node(
        "prior_grid_transform",
        _centroid(prior_size_preserving, prior_size_changing, prior_content_transform),
        prior_size_preserving, prior_size_changing, prior_content_transform,
        sigma_scale=2.0,
    )
    return prior_grid_transform, prior_size_preserving, prior_size_changing, prior_content_transform


# ---------------------------------------------------------------------------
# Transformation priors
# ---------------------------------------------------------------------------

def make_transformation_priors(rows: int, cols: int) -> tuple[HFN, ...]:
    D = rows * cols
    prior_placement    = _node("prior_placement",    _uniform(0.20, D))
    prior_substitution = _node("prior_substitution", _uniform(0.5, D))
    prior_geometric    = _node("prior_geometric",    _uniform(0.5, D))
    prior_transformation = _node(
        "prior_transformation",
        _centroid(prior_placement, prior_substitution, prior_geometric),
        prior_placement, prior_substitution, prior_geometric,
        sigma_scale=2.0,
    )
    return prior_transformation, prior_placement, prior_substitution, prior_geometric


# ---------------------------------------------------------------------------
# Build prior Forest
# ---------------------------------------------------------------------------

def build_prior_forest(rows: int = 3, cols: int = 3) -> tuple[Forest, dict[str, HFN]]:
    """
    Returns (forest, node_registry) for a grid of shape (rows, cols).

    The same conceptual ontology applies to any grid size — the mu vectors
    are computed relative to the grid dimensions.
    """
    registry: dict[str, HFN] = {}

    prior_grid, prior_extent = make_grid_priors(rows, cols)
    prior_density, prior_sparse, prior_medium, prior_dense = make_density_priors(rows, cols)
    prior_structure, prior_connectivity, prior_symmetry, prior_boundary = make_structure_priors(rows, cols)
    prior_colour = make_colour_prior(rows, cols)
    (
        prior_spatial_organisation,
        prior_row_band, prior_row_top, prior_row_mid, prior_row_bot,
        prior_col_band, prior_col_left, prior_col_mid, prior_col_right,
        prior_diagonal, prior_corners,
    ) = make_spatial_organisation_priors(rows, cols)
    prior_grid_transform, prior_size_preserving, prior_size_changing, prior_content_transform = make_grid_transform_priors(rows, cols)
    prior_transformation, prior_placement, prior_substitution, prior_geometric = make_transformation_priors(rows, cols)

    # Wire prior_grid as conceptual parent of prior_density and prior_spatial_organisation
    prior_grid._children.append(prior_density)
    prior_grid._children.append(prior_spatial_organisation)

    all_nodes = [
        prior_grid, prior_extent,
        prior_density, prior_sparse, prior_medium, prior_dense,
        prior_structure, prior_connectivity, prior_symmetry, prior_boundary,
        prior_colour,
        prior_spatial_organisation,
        prior_row_band, prior_row_top, prior_row_mid, prior_row_bot,
        prior_col_band, prior_col_left, prior_col_mid, prior_col_right,
        prior_diagonal, prior_corners,
        prior_grid_transform, prior_size_preserving, prior_size_changing, prior_content_transform,
        prior_transformation, prior_placement, prior_substitution, prior_geometric,
    ]
    for n in all_nodes:
        registry[n.id] = n

    forest = Forest(D=rows * cols, forest_id=f"arc_world_model_{rows}x{cols}")

    # Register specific-level priors (can explain observations directly)
    for node in [
        prior_sparse, prior_medium, prior_dense,
        prior_connectivity, prior_symmetry, prior_boundary,
        prior_colour,
        prior_row_top, prior_row_mid, prior_row_bot,
        prior_col_left, prior_col_mid, prior_col_right,
        prior_diagonal, prior_corners,
        prior_size_preserving, prior_size_changing,
        prior_placement, prior_substitution, prior_geometric,
    ]:
        forest.register(node)

    # Register group/frame priors (expanded into by Observer when surprising)
    for node in [
        prior_grid, prior_extent,
        prior_density, prior_structure,
        prior_spatial_organisation,
        prior_grid_transform, prior_content_transform,
        prior_transformation,
    ]:
        forest.register(node)

    return forest, registry
