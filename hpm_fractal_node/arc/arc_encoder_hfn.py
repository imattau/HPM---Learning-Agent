"""
ARC Encoder HFN — the prior that knows how to perceive grids.

encoder_hfn is a protected prior in the world model. Its children are
the world model priors — it encodes the belief that grids can be
perceived through these structural concepts.

It competes alongside all other nodes in the Forest to explain raw
pixel-space observations. When the Observer expands into it, it finds
the world model priors as decomposition paths.
"""

from __future__ import annotations

import numpy as np
from hfn.hfn import HFN


def _node(name: str, mu: np.ndarray, *children: HFN, sigma_scale: float = 1.0) -> HFN:
    D = mu.shape[0]
    n = HFN(mu=mu, sigma=np.eye(D) * sigma_scale, id=name)
    for c in children:
        n._children.append(c)
    return n


def _centroid(*nodes: HFN) -> np.ndarray:
    return np.mean([n.mu for n in nodes], axis=0)


def build_encoder_hfn(prior_registry: dict[str, HFN]) -> tuple[HFN, dict[str, HFN]]:
    """
    Build the encoder_hfn from the world model priors.

    The encoder's children are shared prior nodes — no copies.
    Returns (encoder_hfn, registry).
    """
    registry: dict[str, HFN] = {}

    # Structural perception priors — what the encoder knows how to measure
    perception_priors = [
        prior_registry[pid] for pid in [
            "prior_grid",
            "prior_density",
            "prior_spatial_organisation",
            "prior_structure",
            "prior_colour",
            "prior_transformation",
        ]
        if pid in prior_registry
    ]

    encoder_hfn = _node(
        "encoder_hfn",
        _centroid(*perception_priors),
        *perception_priors,
        sigma_scale=2.0,
    )
    registry[encoder_hfn.id] = encoder_hfn

    return encoder_hfn, registry
