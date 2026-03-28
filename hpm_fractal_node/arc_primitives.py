"""
ARC Primitives — atomic HFN vocabulary for grid-structured observations.

Primitives are the smallest meaningful units in the ARC domain:
  - cell: a single position in the grid
  - row: a horizontal slice
  - col: a vertical slice
  - region: a connected spatial blob
  - relationship: the concept that two things can be related

All primitives live in pixel space (D = rows * cols). Their mu vectors
encode the concept of each primitive type — not specific instances.
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


def build_primitives(rows: int = 3, cols: int = 3) -> tuple[HFN, dict[str, HFN]]:
    """
    Build the primitive HFN vocabulary for a grid of (rows, cols).

    Returns (primitives_hfn, registry).
    """
    D = rows * cols
    registry: dict[str, HFN] = {}

    # --- Cells: one-hot mu for each grid position ---
    cell_nodes = []
    for r in range(rows):
        for c in range(cols):
            mu = np.zeros(D)
            mu[r * cols + c] = 1.0
            node = _node(f"primitive_cell_{r}{c}", mu)
            cell_nodes.append(node)
            registry[node.id] = node

    # primitive_cell: the concept that a single position exists
    # mu = uniform 1/D (each cell equally likely to be the one)
    primitive_cell = _node(
        "primitive_cell",
        np.full(D, 1.0 / D),
        *cell_nodes,
        sigma_scale=2.0,
    )
    registry[primitive_cell.id] = primitive_cell

    # --- Rows: uniform weight across all cells in the row ---
    row_nodes = []
    for r in range(rows):
        mu = np.zeros(D)
        mu[r * cols:(r + 1) * cols] = 1.0
        node = _node(f"primitive_row_{r}", mu)
        row_nodes.append(node)
        registry[node.id] = node

    primitive_row = _node(
        "primitive_row",
        _centroid(*row_nodes),
        *row_nodes,
        sigma_scale=2.0,
    )
    registry[primitive_row.id] = primitive_row

    # --- Cols: uniform weight across all cells in the column ---
    col_nodes = []
    for c in range(cols):
        mu = np.zeros(D)
        for r in range(rows):
            mu[r * cols + c] = 1.0
        node = _node(f"primitive_col_{c}", mu)
        col_nodes.append(node)
        registry[node.id] = node

    primitive_col = _node(
        "primitive_col",
        _centroid(*col_nodes),
        *col_nodes,
        sigma_scale=2.0,
    )
    registry[primitive_col.id] = primitive_col

    # --- Region: connected blob concept — centre-weighted Gaussian falloff ---
    cr, cc = rows / 2.0, cols / 2.0
    region_mu = np.array([
        np.exp(-((r - cr) ** 2 + (c - cc) ** 2) / (rows * cols / 4.0))
        for r in range(rows) for c in range(cols)
    ])
    mn, mx = region_mu.min(), region_mu.max()
    region_mu = 0.3 + 0.5 * (region_mu - mn) / (mx - mn)
    primitive_region = _node("primitive_region", region_mu, sigma_scale=2.0)
    registry[primitive_region.id] = primitive_region

    # --- Relationship: the concept that two things can be related ---
    # mu = 0.5 everywhere (maximally uninformative — relationship type determines specifics)
    primitive_relationship = _node(
        "primitive_relationship",
        np.full(D, 0.5),
        sigma_scale=3.0,
    )
    registry[primitive_relationship.id] = primitive_relationship

    # --- primitives_hfn: parent of all primitives ---
    primitives_hfn = _node(
        "primitives_hfn",
        _centroid(primitive_cell, primitive_row, primitive_col,
                  primitive_region, primitive_relationship),
        primitive_cell, primitive_row, primitive_col,
        primitive_region, primitive_relationship,
        sigma_scale=3.0,
    )
    registry[primitives_hfn.id] = primitives_hfn

    return primitives_hfn, registry
