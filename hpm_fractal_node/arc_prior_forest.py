"""
ARC prior Forest — conceptual-level world model for 3x3 ARC grids.

Priors are concepts, not specific patterns. They represent categories of
things that exist in the domain. Specific patterns (diagonal, cross, etc.)
are what the Observer learns within these categories — not what it starts with.

Prior hierarchy:

  prior_grid                    (root frame: this observation is a grid)
    ├── prior_extent             (grid has dimensions; all density is relative to these)
    ├── prior_density            (fill level relative to extent)
    │     ├── prior_sparse       (1-3 cells filled)
    │     ├── prior_medium       (4-6 cells filled)
    │     └── prior_dense        (7-9 cells filled)
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

def _node(name: str, mu: np.ndarray, *children: HFN, sigma_scale: float = 1.0) -> HFN:
    n = HFN(mu=mu, sigma=np.eye(D) * sigma_scale, id=name)
    for c in children:
        n._children.append(c)
    return n


def _centroid(*nodes: HFN) -> np.ndarray:
    return np.mean([n.mu for n in nodes], axis=0)


def _uniform(value: float) -> np.ndarray:
    return np.full(D, value)


# ---------------------------------------------------------------------------
# Grid priors
#
# prior_grid is the root frame: "this observation is a grid-structured space."
# prior_extent encodes that the grid has dimensions — fill density is only
# meaningful relative to the total cell count. Both are very broad (sigma=3)
# because they are conceptual anchors, not pattern detectors.
# ---------------------------------------------------------------------------

def make_grid_priors() -> tuple[HFN, HFN]:
    # Extent: grid has a size. mu=0.5 everywhere — maximally uninformative
    # about which cells are filled, but anchors the concept of spatial extent.
    prior_extent = _node("prior_extent", _uniform(0.5), sigma_scale=3.0)

    # Grid: root frame prior. Parent of extent, density, spatial organisation.
    # Very broad — just says "this is a grid-structured observation."
    prior_grid = _node("prior_grid", _uniform(0.5), prior_extent, sigma_scale=3.0)

    return prior_grid, prior_extent


# ---------------------------------------------------------------------------
# Density priors
#
# Encode the concept of fill level. In binary 9-dim space, a pattern's
# density is how many cells are occupied. The prior mu is the theoretical
# centroid of all patterns in that density class.
#
# For k cells filled uniformly at random over 9 positions:
#   centroid_i = k/9  (each cell equally likely to be the filled one)
# ---------------------------------------------------------------------------

def make_density_priors() -> tuple[HFN, HFN, HFN, HFN]:
    # Sparse: 1-3 cells filled. Centroid ≈ 2/9 per cell.
    prior_sparse = _node("prior_sparse", _uniform(2.0 / 9))

    # Medium: 4-6 cells filled. Centroid ≈ 5/9 per cell.
    prior_medium = _node("prior_medium", _uniform(5.0 / 9))

    # Dense: 7-9 cells filled. Centroid ≈ 8/9 per cell.
    prior_dense = _node("prior_dense", _uniform(8.0 / 9))

    prior_density = _node(
        "prior_density",
        _centroid(prior_sparse, prior_medium, prior_dense),
        prior_sparse, prior_medium, prior_dense,
        sigma_scale=2.0,
    )
    return prior_density, prior_sparse, prior_medium, prior_dense


# ---------------------------------------------------------------------------
# Structure priors
#
# Encode the concept that spatial arrangement matters — not just how many
# cells are filled, but how they relate spatially.
# ---------------------------------------------------------------------------

def make_structure_priors() -> tuple[HFN, ...]:
    # Connectivity: connected patterns cluster around centre cells.
    # A connected blob typically involves the centre (cell_11 = index 4).
    connectivity_mu = np.array([0.3, 0.5, 0.3,
                                 0.5, 0.8, 0.5,
                                 0.3, 0.5, 0.3])
    prior_connectivity = _node("prior_connectivity", connectivity_mu)

    # Symmetry: patterns invariant under reflection/rotation.
    # Centroid of all symmetric patterns = uniform 0.5.
    prior_symmetry = _node("prior_symmetry", _uniform(0.5))

    # Boundary: the perimeter vs interior distinction.
    # Perimeter cells: 0,1,2,3,5,6,7,8 (all except centre index 4).
    boundary_mu = np.array([0.7, 0.7, 0.7,
                             0.7, 0.3, 0.7,
                             0.7, 0.7, 0.7])
    prior_boundary = _node("prior_boundary", boundary_mu)

    prior_structure = _node(
        "prior_structure",
        _centroid(prior_connectivity, prior_symmetry, prior_boundary),
        prior_connectivity, prior_symmetry, prior_boundary,
        sigma_scale=2.0,
    )
    return prior_structure, prior_connectivity, prior_symmetry, prior_boundary


# ---------------------------------------------------------------------------
# Colour prior
#
# In binary encoding, colour (value identity) is collapsed to presence/absence.
# This prior encodes the concept that cells HAVE colour — it's a placeholder
# that tells the system "colour is a meaningful attribute in this domain".
# Its explanatory power activates when the encoding is extended to include colour.
# ---------------------------------------------------------------------------

def make_colour_prior() -> HFN:
    # Very broad — conceptual placeholder.
    # mu = 0.5 everywhere: "any pattern could have any colour"
    return _node("prior_colour", _uniform(0.5), sigma_scale=3.0)


# ---------------------------------------------------------------------------
# Spatial organisation priors
#
# Encode the concept that cells are arranged directionally — rows, columns,
# diagonals, and corners are all salient spatial organisations in ARC.
#
# Row/col priors: centroid weights the dominant band heavily (0.7),
# adjacent band lightly (0.3), and opposite band minimally (0.1).
# Diagonal prior: centroid weights diagonal cells (0.5), centre (1.0).
# Corner prior: centroid weights the 4 corners (0.7), edges low (0.1).
# ---------------------------------------------------------------------------

def make_spatial_organisation_priors() -> tuple[HFN, ...]:
    # Row bands: horizontal organisation
    prior_row_top = _node("prior_row_top", np.array([0.7, 0.7, 0.7,
                                                      0.3, 0.3, 0.3,
                                                      0.1, 0.1, 0.1]))
    prior_row_mid = _node("prior_row_mid", np.array([0.1, 0.1, 0.1,
                                                      0.7, 0.7, 0.7,
                                                      0.1, 0.1, 0.1]))
    prior_row_bot = _node("prior_row_bot", np.array([0.1, 0.1, 0.1,
                                                      0.3, 0.3, 0.3,
                                                      0.7, 0.7, 0.7]))
    prior_row_band = _node(
        "prior_row_band",
        _centroid(prior_row_top, prior_row_mid, prior_row_bot),
        prior_row_top, prior_row_mid, prior_row_bot,
        sigma_scale=2.0,
    )

    # Column bands: vertical organisation
    prior_col_left  = _node("prior_col_left",  np.array([0.7, 0.3, 0.1,
                                                          0.7, 0.3, 0.1,
                                                          0.7, 0.3, 0.1]))
    prior_col_mid   = _node("prior_col_mid",   np.array([0.1, 0.7, 0.1,
                                                          0.1, 0.7, 0.1,
                                                          0.1, 0.7, 0.1]))
    prior_col_right = _node("prior_col_right", np.array([0.1, 0.3, 0.7,
                                                          0.1, 0.3, 0.7,
                                                          0.1, 0.3, 0.7]))
    prior_col_band = _node(
        "prior_col_band",
        _centroid(prior_col_left, prior_col_mid, prior_col_right),
        prior_col_left, prior_col_mid, prior_col_right,
        sigma_scale=2.0,
    )

    # Diagonal: either main diagonal (TL→BR) or anti-diagonal (TR→BL).
    # Centroid weights both diagonals equally + the centre intersection.
    prior_diagonal = _node("prior_diagonal", np.array([0.5, 0.0, 0.5,
                                                        0.0, 1.0, 0.0,
                                                        0.5, 0.0, 0.5]))

    # Corners: the four corner positions as salient locations.
    prior_corners = _node("prior_corners", np.array([0.7, 0.1, 0.7,
                                                      0.1, 0.1, 0.1,
                                                      0.7, 0.1, 0.7]))

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
#
# Encode the structural relationship between input and output grids.
# Orthogonal to transformation priors (which describe rule *type*) —
# these describe the *size relationship* between the two grid objects.
# ---------------------------------------------------------------------------

def make_grid_transform_priors() -> tuple[HFN, ...]:
    # Size-preserving: output has same dimensions as input.
    # In binary space this looks like any density — uniform centroid.
    prior_size_preserving = _node("prior_size_preserving", _uniform(0.5))

    # Size-changing: output dimensions differ (tiling makes it larger;
    # cropping makes it smaller). Typically involves sparse inputs (the
    # object being tiled/cropped is a sub-pattern).
    prior_size_changing = _node("prior_size_changing", _uniform(2.0 / 9))

    # Content-transform: same spatial shape, different cell values.
    # Binary encoding collapses this to identity — placeholder for
    # when colour encoding is added.
    prior_content_transform = _node("prior_content_transform", _uniform(0.5), sigma_scale=3.0)

    prior_grid_transform = _node(
        "prior_grid_transform",
        _centroid(prior_size_preserving, prior_size_changing, prior_content_transform),
        prior_size_preserving, prior_size_changing, prior_content_transform,
        sigma_scale=2.0,
    )
    return prior_grid_transform, prior_size_preserving, prior_size_changing, prior_content_transform


# ---------------------------------------------------------------------------
# Transformation priors
#
# Encode the concept that input/output pairs in ARC are related by rules.
# Three categories of rule:
#   - Placement: copy a pattern to positions defined by another pattern
#   - Substitution: replace values while preserving spatial structure
#   - Geometric: rotate, reflect, scale
# ---------------------------------------------------------------------------

def make_transformation_priors() -> tuple[HFN, ...]:
    # Placement rules operate on sparse inputs (the mask is sparse).
    prior_placement = _node("prior_placement", _uniform(2.0 / 9))

    # Substitution rules preserve shape: binary pattern looks the same
    # before and after (colour changes, structure doesn't).
    prior_substitution = _node("prior_substitution", _uniform(0.5))

    # Geometric rules (rotation, reflection) preserve density.
    # Centroid uniform — rotational symmetry distributes weight equally.
    prior_geometric = _node("prior_geometric", _uniform(0.5))

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

def build_prior_forest() -> tuple[Forest, dict[str, HFN]]:
    """
    Returns (forest, node_registry).

    Top-level registered nodes are the specific-level priors that can
    directly explain observations (sigma=I). Conceptual group nodes
    (sigma=2.0 or 3.0) are registered too — the Observer expands into
    their children when they're surprising.
    """
    registry: dict[str, HFN] = {}

    prior_grid, prior_extent = make_grid_priors()
    prior_density, prior_sparse, prior_medium, prior_dense = make_density_priors()
    prior_structure, prior_connectivity, prior_symmetry, prior_boundary = make_structure_priors()
    prior_colour = make_colour_prior()
    (
        prior_spatial_organisation,
        prior_row_band, prior_row_top, prior_row_mid, prior_row_bot,
        prior_col_band, prior_col_left, prior_col_mid, prior_col_right,
        prior_diagonal, prior_corners,
    ) = make_spatial_organisation_priors()
    prior_grid_transform, prior_size_preserving, prior_size_changing, prior_content_transform = make_grid_transform_priors()
    prior_transformation, prior_placement, prior_substitution, prior_geometric = make_transformation_priors()

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

    forest = Forest(D=D, forest_id="arc_world_model")

    # Register specific-level priors directly (can explain observations)
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
