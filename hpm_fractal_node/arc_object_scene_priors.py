"""
ARC Object/Scene/Rule priors — semantic layer above structural priors.

This layer encodes the prior knowledge that ARC tasks involve:
  - Objects: bounded, coherent foreground regions with shape and colour
  - Scenes: grids parsed as background + foreground objects
  - Rules: relationships between input and output scenes

These are the concepts a human brings to ARC that make the tasks solvable.
Without them, the Observer can discover statistical regularities but cannot
reason about objects, scenes, or transformation rules.

Prior hierarchy:

  object_hfn                  (a bounded, coherent foreground region)
    ├── object_shape           (the spatial structure of the object)
    ├── object_colour          (the value identity of the object)
    └── object_count           (cardinality: how many of this object exist)

  scene_hfn                   (a complete grid = background + objects)
    ├── scene_background       (the dominant/frame value)
    └── scene_objects          (the foreground objects)

  rule_hfn                    (what transforms input scene → output scene)
    ├── rule_object_transform  (apply a transformation to each object)
    ├── rule_count_based       (output depends on count of something)
    └── rule_colour_map        (replace one colour/value with another)

Shared references: object_shape references primitive_region; object_colour
references prior_colour; rule_colour_map references prior_substitution.
These are the same HFN objects — no copies.
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


def _uniform(value: float, D: int) -> np.ndarray:
    return np.full(D, value)


def _grid_mu(rows: int, cols: int, fn) -> np.ndarray:
    return np.array([fn(r, c) for r in range(rows) for c in range(cols)], dtype=float)


def build_object_scene_priors(
    rows: int,
    cols: int,
    shared: dict[str, HFN],
) -> tuple[HFN, HFN, HFN, dict[str, HFN]]:
    """
    Build object, scene, and rule priors for a grid of (rows, cols).

    shared: registry of existing nodes to reference (not copy).
    Returns (object_hfn, scene_hfn, rule_hfn, registry).
    """
    D = rows * cols
    registry: dict[str, HFN] = {}
    cr, cc = rows / 2.0, cols / 2.0

    # -----------------------------------------------------------------------
    # Object priors
    # -----------------------------------------------------------------------

    # Object shape: a coherent connected region — centre-weighted.
    # Objects in ARC tend to be compact, connected blobs.
    # References primitive_region if available, otherwise builds standalone.
    if "primitive_region" in shared:
        object_shape = shared["primitive_region"]
    else:
        shape_mu = _grid_mu(rows, cols, lambda r, c:
            np.exp(-((r - cr)**2 + (c - cc)**2) / (rows * cols / 4.0)))
        mn, mx = shape_mu.min(), shape_mu.max()
        shape_mu = 0.3 + 0.5 * (shape_mu - mn) / (mx - mn)
        object_shape = _node("object_shape", shape_mu)
        registry[object_shape.id] = object_shape

    # Object colour: value identity beyond presence/absence.
    # References prior_colour if available (shared node).
    if "prior_colour" in shared:
        object_colour = shared["prior_colour"]
    else:
        object_colour = _node("object_colour", _uniform(0.5, D), sigma_scale=3.0)
        registry[object_colour.id] = object_colour

    # Object count: cardinality — how many objects exist.
    # In binary space, multiple objects = multiple sparse disconnected regions.
    # Centroid: sparse (each object occupies a small fraction of cells).
    object_count = _node("object_count", _uniform(0.15, D), sigma_scale=2.0)
    registry[object_count.id] = object_count

    object_hfn = _node(
        "object_hfn",
        _centroid(object_shape, object_colour, object_count),
        object_shape, object_colour, object_count,
        sigma_scale=2.0,
    )
    registry[object_hfn.id] = object_hfn

    # -----------------------------------------------------------------------
    # Scene priors
    # -----------------------------------------------------------------------

    # Scene background: the dominant/frame value — typically empty (0).
    # mu = very sparse: background cells are mostly 0.
    scene_background = _node("scene_background", _uniform(0.1, D), sigma_scale=2.0)
    registry[scene_background.id] = scene_background

    # Scene objects: the foreground — sparse objects on the background.
    # References object_hfn (shared).
    scene_objects = _node(
        "scene_objects",
        _uniform(0.25, D),
        object_hfn,
        sigma_scale=2.0,
    )
    registry[scene_objects.id] = scene_objects

    scene_hfn = _node(
        "scene_hfn",
        _centroid(scene_background, scene_objects),
        scene_background, scene_objects,
        sigma_scale=2.0,
    )
    registry[scene_hfn.id] = scene_hfn

    # -----------------------------------------------------------------------
    # Rule priors
    # -----------------------------------------------------------------------

    # Rule: object transform — apply a spatial transformation to each object.
    # Input looks like a scene with objects (sparse-medium).
    rule_object_transform = _node(
        "rule_object_transform",
        _uniform(0.3, D),
        object_hfn,
        sigma_scale=2.0,
    )
    registry[rule_object_transform.id] = rule_object_transform

    # Rule: count-based — output depends on counting objects/cells.
    # Input is typically sparse (small objects to count).
    rule_count_based = _node(
        "rule_count_based",
        _uniform(0.15, D),
        object_count,
        sigma_scale=2.0,
    )
    registry[rule_count_based.id] = rule_count_based

    # Rule: colour map — replace one value with another, shape preserved.
    # References prior_substitution if available (same concept, shared node).
    if "prior_substitution" in shared:
        rule_colour_map = shared["prior_substitution"]
    else:
        rule_colour_map = _node("rule_colour_map", _uniform(0.5, D))
        registry[rule_colour_map.id] = rule_colour_map

    rule_hfn = _node(
        "rule_hfn",
        _centroid(rule_object_transform, rule_count_based, rule_colour_map),
        rule_object_transform, rule_count_based, rule_colour_map,
        sigma_scale=2.0,
    )
    registry[rule_hfn.id] = rule_hfn

    return object_hfn, scene_hfn, rule_hfn, registry
