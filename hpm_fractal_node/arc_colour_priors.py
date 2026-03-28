"""
ARC Colour priors — value-identity priors for colour-encoded observations.

ARC uses 10 colour values (0-9). Value 0 is typically background/empty.
Encoding: cell_value / 9.0 → each cell in [0.0, 1.0].

This gives colour-aware observations while keeping D = rows * cols.
Distances now reflect both spatial position AND colour similarity —
a cell with colour 1 is "close to" a cell with colour 2.

Colour prior hierarchy:

  prior_colour_background   (value 0 — empty/background)
  prior_colour_low          (values 1-3 — darker foreground colours)
  prior_colour_mid          (values 4-6 — medium foreground colours)
  prior_colour_high         (values 7-9 — bright foreground colours)
  prior_colour_uniform      (any colour uniformly — shape-agnostic)

  prior_colour_group        (parent: colour identity is meaningful)
    ├── prior_colour_background
    ├── prior_colour_low
    ├── prior_colour_mid
    └── prior_colour_high

ARC colour constants (0-9):
  0=black(bg), 1=blue, 2=red, 3=green, 4=yellow,
  5=grey, 6=magenta, 7=orange, 8=light_blue, 9=maroon
"""

from __future__ import annotations

import numpy as np
from hpm_fractal_node.hfn import HFN


N_COLOURS = 10  # ARC uses values 0-9


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


def build_colour_priors(rows: int = 3, cols: int = 3) -> tuple[HFN, dict[str, HFN]]:
    """
    Build colour priors for value-encoded observations (cell_value / 9.0).

    Returns (prior_colour_group, registry).
    """
    D = rows * cols
    registry: dict[str, HFN] = {}

    # Background: value 0 → 0.0. All cells at background value.
    prior_colour_background = _node(
        "prior_colour_background",
        _uniform(0.0 / 9, D),
        sigma_scale=1.0,
    )
    registry[prior_colour_background.id] = prior_colour_background

    # Low colours: values 1-3 → centroid ≈ 2/9 ≈ 0.22
    prior_colour_low = _node(
        "prior_colour_low",
        _uniform(2.0 / 9, D),
        sigma_scale=1.0,
    )
    registry[prior_colour_low.id] = prior_colour_low

    # Mid colours: values 4-6 → centroid ≈ 5/9 ≈ 0.56
    prior_colour_mid = _node(
        "prior_colour_mid",
        _uniform(5.0 / 9, D),
        sigma_scale=1.0,
    )
    registry[prior_colour_mid.id] = prior_colour_mid

    # High colours: values 7-9 → centroid ≈ 8/9 ≈ 0.89
    prior_colour_high = _node(
        "prior_colour_high",
        _uniform(8.0 / 9, D),
        sigma_scale=1.0,
    )
    registry[prior_colour_high.id] = prior_colour_high

    # Uniform: any colour — shape-agnostic, broad
    prior_colour_uniform = _node(
        "prior_colour_uniform",
        _uniform(0.5, D),
        sigma_scale=3.0,
    )
    registry[prior_colour_uniform.id] = prior_colour_uniform

    # Group: colour identity is meaningful
    prior_colour_group = _node(
        "prior_colour_group",
        _centroid(
            prior_colour_background,
            prior_colour_low,
            prior_colour_mid,
            prior_colour_high,
        ),
        prior_colour_background, prior_colour_low,
        prior_colour_mid, prior_colour_high,
        sigma_scale=2.0,
    )
    registry[prior_colour_group.id] = prior_colour_group

    return prior_colour_group, registry
