"""
ARC Perception priors — the sensory foundation of the world model.

These priors encode how the world is perceived before any structure
is recognised. They sit below primitives in the hierarchy:

  prior_signal              (the world sends discrete measurements)
    └── prior_pixel         (each measurement has a position and value)
          └── prior_cell    (positioned values form a spatial unit)
                └── prior_grid  (spatial units form a structured field)

This is the perceptual chain: raw signal → pixel → cell → grid.

Without these, the system is handed cells and told they form a grid.
With these, it knows WHY cells form a grid — because adjacent positioned
values with consistent structure are perceived as a coherent spatial field.

Key concepts encoded:

  prior_signal:
    The world arrives as a sequence of discrete measurements. No spatial
    structure assumed yet — just "values exist and can be observed."
    mu = uniform 0.5 (any value equally likely), very broad sigma.

  prior_pixel:
    A measurement has both a VALUE and a POSITION. Position is as
    meaningful as value. Nearby positions are more related than distant ones.
    mu = centre-weighted (positions near centre are most salient in perception).

  prior_spatial_adjacency:
    Adjacent pixels influence each other — they are more likely to share
    structure than non-adjacent pixels. The foundation of region perception.
    mu = locally correlated (adjacent cells have similar weights).

  prior_cell:
    A positioned value within a known spatial frame (the grid).
    Distinguished from prior_pixel by the presence of the frame context.
    mu = uniform (any cell position equally a "cell").

  prior_field:
    A collection of cells forms a field — a structured spatial whole.
    The field has extent, density, and internal organisation.
    This is what makes a grid a grid rather than a set of unrelated cells.
    mu = uniform 0.5, broad — the concept of a bounded spatial field.
"""

from __future__ import annotations

import numpy as np
from hpm_fractal_node.hfn import HFN


def _node(name: str, mu: np.ndarray, *children: HFN, sigma_scale: float = 1.0) -> HFN:
    D = mu.shape[0]
    n = HFN(mu=mu, sigma=np.eye(D) * sigma_scale, id=name)
    for c in children:
        n._children.append(c)
    return n


def _centroid(*nodes: HFN) -> np.ndarray:
    return np.mean([n.mu for n in nodes], axis=0)


def _grid_mu(rows: int, cols: int, fn) -> np.ndarray:
    return np.array([fn(r, c) for r in range(rows) for c in range(cols)], dtype=float)


def build_perception_priors(rows: int = 3, cols: int = 3) -> tuple[HFN, dict[str, HFN]]:
    """
    Build the perceptual foundation layer for a grid of (rows, cols).

    Returns (prior_signal, registry).
    prior_signal is the root — all other perception priors are its descendants.
    """
    D = rows * cols
    registry: dict[str, HFN] = {}
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0

    # prior_signal: the world sends discrete measurements.
    # Maximally broad — no spatial assumption, no value assumption.
    prior_signal = _node("prior_signal", np.full(D, 0.5), sigma_scale=4.0)
    registry[prior_signal.id] = prior_signal

    # prior_pixel: a measurement has both value and position.
    # Centre-weighted: central positions are more perceptually salient
    # (fixation bias — the eye/attention tends to centre).
    def pixel_weight(r, c):
        # Gaussian falloff from centre — mild, not extreme
        dist = ((r - cr) / rows) ** 2 + ((c - cc) / cols) ** 2
        return float(0.3 + 0.4 * np.exp(-4 * dist))

    prior_pixel = _node(
        "prior_pixel",
        _grid_mu(rows, cols, pixel_weight),
        prior_signal,
        sigma_scale=3.0,
    )
    registry[prior_pixel.id] = prior_pixel

    # prior_spatial_adjacency: adjacent pixels are more related than distant ones.
    # Encodes local correlation — the foundation of region/object perception.
    # mu: each cell weighted by how many neighbours it has (corners=2, edges=3, interior=4)
    # normalised to [0.3, 0.7].
    def adjacency_weight(r, c):
        neighbours = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if 0 <= r+dr < rows and 0 <= c+dc < cols:
                neighbours += 1
        return neighbours  # raw count

    adj_mu = _grid_mu(rows, cols, adjacency_weight)
    mn, mx = adj_mu.min(), adj_mu.max()
    adj_mu = 0.3 + 0.4 * (adj_mu - mn) / (mx - mn + 1e-8)
    prior_spatial_adjacency = _node(
        "prior_spatial_adjacency",
        adj_mu,
        prior_pixel,
        sigma_scale=2.0,
    )
    registry[prior_spatial_adjacency.id] = prior_spatial_adjacency

    # prior_cell: a positioned value within a known spatial frame.
    # Uniform — any position is equally a valid cell within the frame.
    prior_cell_concept = _node(
        "prior_cell_concept",
        np.full(D, 1.0 / D),
        prior_pixel,
        sigma_scale=2.0,
    )
    registry[prior_cell_concept.id] = prior_cell_concept

    # prior_field: a collection of cells forming a bounded spatial whole.
    # Uniform 0.5 — any density can constitute a field.
    # Broad sigma — the concept of "field" doesn't constrain which cells are filled.
    prior_field = _node(
        "prior_field",
        np.full(D, 0.5),
        prior_cell_concept, prior_spatial_adjacency,
        sigma_scale=3.0,
    )
    registry[prior_field.id] = prior_field

    # Wire the perceptual chain: signal → pixel → cell/adjacency → field
    # prior_signal is the root; prior_field connects back up to prior_grid
    # (that wiring happens in arc_world_model.py)

    return prior_signal, registry
