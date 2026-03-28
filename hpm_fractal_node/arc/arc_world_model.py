"""
ARC World Model — single entry point assembling all layers into one Forest.

Layers (bottom to top):
  0. Perception    — signal, pixel, spatial adjacency, cell, field
  1. Primitives    — atomic vocabulary (cell, row, col, region, relationship)
  2. Relationships — relational vocabulary (adjacency, mirror, repeat)
  3. Structural    — world model priors (density, structure, spatial, transform)
  4. Colour        — value-identity priors (background, low, mid, high)
  5. Semantic      — object, scene, rule priors (the concepts ARC tasks require)
  6. Encoder       — prior that knows how to perceive grids

The perceptual chain: prior_signal → prior_pixel → prior_cell_concept →
prior_field → prior_grid (structural layer). This encodes HOW the AI
perceives the world before any structure is recognised.

Cross-layer bridge:
  prior_pixel_colour — a pixel has both a position AND a colour identity.
  Children: prior_pixel (Layer 0) + prior_colour_group (Layer 4).
  This is the concept that makes ARC colour-encoded observations coherent:
  each cell is not just "present/absent" but "present with a specific colour
  at a specific position".

All nodes are protected priors. The Observer learns above this layer —
new nodes it discovers reference these as children, but the layer itself
is invariant structural knowledge.

Run experiment: python3 -m hpm_fractal_node.experiment_arc_world_model
"""

from __future__ import annotations

import numpy as np
from hfn.hfn import HFN
from hfn.forest import Forest
from hpm_fractal_node.arc.arc_perception_priors import build_perception_priors
from hpm_fractal_node.arc.arc_primitives import build_primitives
from hpm_fractal_node.arc.arc_relationships import build_relationships
from hpm_fractal_node.arc.arc_prior_forest import build_prior_forest
from hpm_fractal_node.arc.arc_colour_priors import build_colour_priors
from hpm_fractal_node.arc.arc_object_scene_priors import build_object_scene_priors
from hpm_fractal_node.arc.arc_encoder_hfn import build_encoder_hfn


def build_world_model(rows: int = 3, cols: int = 3) -> tuple[Forest, dict[str, HFN]]:
    """
    Assemble the full ARC world model for a grid of (rows, cols).

    Returns (forest, full_registry).
    All nodes in the registry are protected priors.
    """
    D = rows * cols
    full_registry: dict[str, HFN] = {}

    # Layer 0: Perception — how the world is perceived before structure is known
    prior_signal, perc_registry = build_perception_priors(rows, cols)
    full_registry.update(perc_registry)

    # Layer 1: Primitives — atomic spatial vocabulary
    primitives_hfn, prim_registry = build_primitives(rows, cols)
    full_registry.update(prim_registry)

    # Layer 2: Relationships — relational vocabulary (reference primitives)
    relationships_hfn, rel_registry = build_relationships(prim_registry, rows, cols)
    full_registry.update(rel_registry)

    # Layer 3: Structural priors — density, spatial organisation, structure, transformation
    prior_forest, prior_registry = build_prior_forest(rows, cols)
    full_registry.update(prior_registry)

    # Layer 4: Colour priors — value-identity beyond binary presence/absence
    prior_colour_group, colour_registry = build_colour_priors(rows, cols)
    full_registry.update(colour_registry)

    # Bridge: prior_pixel_colour — a pixel has both position AND colour identity.
    # Children: prior_pixel (perception) + prior_colour_group (colour layer).
    # This is the concept that each ARC cell carries a colour at a specific position,
    # not merely a binary presence/absence signal.
    prior_pixel = perc_registry["prior_pixel"]
    pixel_colour_mu = np.mean([prior_pixel.mu, prior_colour_group.mu], axis=0)
    prior_pixel_colour = HFN(
        mu=pixel_colour_mu,
        sigma=np.eye(D) * 2.0,
        id="prior_pixel_colour",
    )
    prior_pixel_colour._children.append(prior_pixel)
    prior_pixel_colour._children.append(prior_colour_group)
    full_registry["prior_pixel_colour"] = prior_pixel_colour

    # Layer 5: Semantic priors — object, scene, rule (share nodes from layers 1-3)
    shared = {**prim_registry, **prior_registry, "prior_colour": prior_pixel_colour}
    object_hfn, scene_hfn, rule_hfn, sem_registry = build_object_scene_priors(
        rows, cols, shared
    )
    full_registry.update(sem_registry)

    # Layer 6: Encoder — perceives grids through structural concepts
    encoder_hfn, enc_registry = build_encoder_hfn(prior_registry)
    full_registry.update(enc_registry)

    # Wire perceptual chain: prior_field → prior_grid
    # prior_field (perception layer) is the foundation that prior_grid builds on
    if "prior_field" in perc_registry and "prior_grid" in prior_registry:
        prior_grid = prior_registry["prior_grid"]
        prior_field = perc_registry["prior_field"]
        if prior_field not in prior_grid.children():
            prior_grid._children.append(prior_field)

    # Assemble into one Forest
    forest = Forest(D=D, forest_id=f"arc_world_model_{rows}x{cols}")

    # Register perception layer (foundation)
    for node in perc_registry.values():
        if node.id not in forest:
            forest.register(node)

    # Register structural priors (already internally structured)
    for node in prior_forest.active_nodes():
        if node.id not in forest:
            forest.register(node)

    # Register primitives
    for node in prim_registry.values():
        if node.id not in forest:
            forest.register(node)

    # Register relationships
    for node in rel_registry.values():
        if node.id not in forest:
            forest.register(node)

    # Register colour priors
    for node in colour_registry.values():
        if node.id not in forest:
            forest.register(node)

    # Register pixel-colour bridge
    forest.register(prior_pixel_colour)

    # Register semantic priors
    for node in sem_registry.values():
        if node.id not in forest:
            forest.register(node)

    # Register encoder
    forest.register(encoder_hfn)

    return forest, full_registry
