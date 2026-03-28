"""
ARC World Model — single entry point assembling all layers into one Forest.

Layers (bottom to top):
  1. Primitives    — atomic vocabulary (cell, row, col, region, relationship)
  2. Relationships — relational vocabulary (adjacency, mirror, repeat)
  3. Priors        — world model priors (density, structure, spatial, transform)
  4. Semantic      — object, scene, rule priors (the concepts ARC tasks require)
  5. Encoder       — prior that knows how to perceive grids

All nodes are protected priors. The Observer learns above this layer —
new nodes it discovers reference these as children, but the layer itself
is invariant structural knowledge.

Run experiment: python3 -m hpm_fractal_node.experiment_arc_world_model
"""

from __future__ import annotations

from hpm_fractal_node.hfn import HFN
from hpm_fractal_node.forest import Forest
from hpm_fractal_node.arc_primitives import build_primitives
from hpm_fractal_node.arc_relationships import build_relationships
from hpm_fractal_node.arc_prior_forest import build_prior_forest
from hpm_fractal_node.arc_object_scene_priors import build_object_scene_priors
from hpm_fractal_node.arc_colour_priors import build_colour_priors
from hpm_fractal_node.arc_encoder_hfn import build_encoder_hfn


def build_world_model(rows: int = 3, cols: int = 3) -> tuple[Forest, dict[str, HFN]]:
    """
    Assemble the full ARC world model for a grid of (rows, cols).

    Returns (forest, full_registry).
    All nodes in the registry are protected priors.
    """
    D = rows * cols
    full_registry: dict[str, HFN] = {}

    # Layer 1: Primitives
    primitives_hfn, prim_registry = build_primitives(rows, cols)
    full_registry.update(prim_registry)

    # Layer 2: Relationships (reference primitive nodes)
    relationships_hfn, rel_registry = build_relationships(prim_registry, rows, cols)
    full_registry.update(rel_registry)

    # Layer 3: World model priors
    prior_forest, prior_registry = build_prior_forest(rows, cols)
    full_registry.update(prior_registry)

    # Layer 4: Colour priors (value-identity beyond binary presence/absence)
    prior_colour_group, colour_registry = build_colour_priors(rows, cols)
    full_registry.update(colour_registry)

    # Layer 5: Semantic priors — object, scene, rule (share nodes from layers 1-3)
    shared = {**prim_registry, **prior_registry}
    object_hfn, scene_hfn, rule_hfn, sem_registry = build_object_scene_priors(
        rows, cols, shared
    )
    full_registry.update(sem_registry)

    # Layer 6: Encoder (references world model priors as children)
    encoder_hfn, enc_registry = build_encoder_hfn(prior_registry)
    full_registry.update(enc_registry)

    # Assemble into one Forest
    forest = Forest(D=D, forest_id=f"arc_world_model_{rows}x{cols}")

    # Register all prior-forest nodes first (already structured internally)
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

    # Register semantic priors
    for node in sem_registry.values():
        if node.id not in forest:
            forest.register(node)

    # Register encoder
    forest.register(encoder_hfn)

    return forest, full_registry
