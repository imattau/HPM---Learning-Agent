"""
ARC Relationships — relational HFN vocabulary for grid-structured observations.

Relationships describe how two primitives relate — not what they are.
They live in pixel space (D = rows * cols) as the centroid of the
primitive pairs that exhibit each relationship type.

  relationship_adjacency    — two things that touch (unit displacement)
  relationship_mirror       — two things that are reflections of each other
  relationship_repeat       — two things that recur with consistent displacement
  relationship_cell_colour  — a cell within the grid is bound to a colour value

prim_* nodes are compressions of a primitive concept with a relationship
type — typed relationship slots the Observer fills with specific instances.

  prim_cell_colour = recombine(primitive_cell, relationship_cell_colour)
    The Observer fills this with specific (position, colour) bindings —
    e.g. compressed(primitive_cell_23, relationship_cell_colour) means
    "cell at row 2, col 3 is bound to a specific colour."
"""

from __future__ import annotations

import numpy as np
from hpm_fractal_node.hfn import HFN
from hpm_fractal_node.arc_primitives import build_primitives


def _node(name: str, mu: np.ndarray, *children: HFN, sigma_scale: float = 1.0) -> HFN:
    D = mu.shape[0]
    n = HFN(mu=mu, sigma=np.eye(D) * sigma_scale, id=name)
    for c in children:
        n._children.append(c)
    return n


def _centroid(*nodes: HFN) -> np.ndarray:
    return np.mean([n.mu for n in nodes], axis=0)


def build_relationships(
    prim_registry: dict[str, HFN],
    rows: int = 3,
    cols: int = 3,
) -> tuple[HFN, dict[str, HFN]]:
    """
    Build the relationship HFN vocabulary for a grid of (rows, cols).

    Returns (relationships_hfn, registry).
    """
    D = rows * cols
    registry: dict[str, HFN] = {}

    def cell_mu(r: int, c: int) -> np.ndarray:
        mu = np.zeros(D)
        mu[r * cols + c] = 1.0
        return mu

    # --- Adjacency: centroid of all horizontally/vertically adjacent cell pairs ---
    adj_pairs = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:  # horizontal neighbour
                adj_pairs.append((cell_mu(r, c) + cell_mu(r, c + 1)) / 2)
            if r + 1 < rows:  # vertical neighbour
                adj_pairs.append((cell_mu(r, c) + cell_mu(r + 1, c)) / 2)

    relationship_adjacency = _node(
        "relationship_adjacency",
        np.mean(adj_pairs, axis=0),
        sigma_scale=2.0,
    )
    registry[relationship_adjacency.id] = relationship_adjacency

    # --- Mirror: centroid of horizontally and vertically reflected cell pairs ---
    mirror_pairs = []
    for r in range(rows):
        for c in range(cols):
            # Horizontal mirror: (r,c) ↔ (r, cols-1-c)
            if c < cols - 1 - c:
                mirror_pairs.append((cell_mu(r, c) + cell_mu(r, cols - 1 - c)) / 2)
            # Vertical mirror: (r,c) ↔ (rows-1-r, c)
            if r < rows - 1 - r:
                mirror_pairs.append((cell_mu(r, c) + cell_mu(rows - 1 - r, c)) / 2)

    relationship_mirror = _node(
        "relationship_mirror",
        np.mean(mirror_pairs, axis=0) if mirror_pairs else np.full(D, 0.5),
        sigma_scale=2.0,
    )
    registry[relationship_mirror.id] = relationship_mirror

    # --- Repeat: centroid of cell pairs with consistent displacement ---
    # Use pairs separated by (0,1), (1,0), (1,1) as representative repeat patterns
    repeat_pairs = []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for r in range(rows):
            for c in range(cols):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    repeat_pairs.append((cell_mu(r, c) + cell_mu(r2, c2)) / 2)

    relationship_repeat = _node(
        "relationship_repeat",
        np.mean(repeat_pairs, axis=0),
        sigma_scale=2.0,
    )
    registry[relationship_repeat.id] = relationship_repeat

    # --- Cell-colour binding: a cell within the grid is bound to a colour value ---
    # mu = uniform 0.5 — the abstract concept of colour-at-position, not a specific binding.
    # The Observer creates specific instances via recombine(primitive_cell_rc, this).
    relationship_cell_colour = _node(
        "relationship_cell_colour",
        np.full(D, 0.5),
        sigma_scale=2.0,
    )
    registry[relationship_cell_colour.id] = relationship_cell_colour

    # --- relationships_hfn: parent of all relationship types ---
    relationships_hfn = _node(
        "relationships_hfn",
        _centroid(relationship_adjacency, relationship_mirror, relationship_repeat,
                  relationship_cell_colour),
        relationship_adjacency, relationship_mirror, relationship_repeat,
        relationship_cell_colour,
        sigma_scale=3.0,
    )
    registry[relationships_hfn.id] = relationships_hfn

    # --- prim_* compressions: typed relationship slots ---
    # Each is recombine(primitive, relationship) — prior compression
    primitive_cell = prim_registry["primitive_cell"]
    primitive_region = prim_registry["primitive_region"]

    prim_adjacency = primitive_cell.recombine(relationship_adjacency)
    prim_adjacency.id = "prim_adjacency"  # type: ignore[misc]
    registry[prim_adjacency.id] = prim_adjacency

    prim_mirror = primitive_cell.recombine(relationship_mirror)
    prim_mirror.id = "prim_mirror"  # type: ignore[misc]
    registry[prim_mirror.id] = prim_mirror

    prim_repeat = primitive_region.recombine(relationship_repeat)
    prim_repeat.id = "prim_repeat"  # type: ignore[misc]
    registry[prim_repeat.id] = prim_repeat

    prim_cell_colour = primitive_cell.recombine(relationship_cell_colour)
    prim_cell_colour.id = "prim_cell_colour"  # type: ignore[misc]
    registry[prim_cell_colour.id] = prim_cell_colour

    return relationships_hfn, registry
