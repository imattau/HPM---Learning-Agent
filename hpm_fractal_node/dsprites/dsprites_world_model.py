"""
dSprites world model — 5 layers of human-like priors for 16x16 (D=256) images.

Layer 0  Perception    background, signal presence
Layer 1  Spatial cells 16 priors — one per 4x4 quadrant
Layer 2  Shape         square, ellipse, heart templates at center
Layer 3  Scale         small / medium / large object
Layer 4  Position      9 priors on a 3x3 spatial grid
Layer 5  Orientation   4 priors — vertical, horizontal, diagonal x2

All priors are protected — the Observer cannot absorb or remove them.
"""

from __future__ import annotations

import numpy as np

from hfn import HFN, Forest

GRID = 16
D = GRID * GRID  # 256


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _blank() -> np.ndarray:
    return np.zeros(D, dtype=np.float32)


def _render(img: np.ndarray) -> np.ndarray:
    """Flatten a (16,16) image to a (256,) vector."""
    return img.flatten().astype(np.float32)


def _gaussian_blob(cy: float, cx: float, sigma: float = 2.5) -> np.ndarray:
    """Soft Gaussian blob centred at (cy, cx) in a 16x16 grid."""
    ys, xs = np.mgrid[:GRID, :GRID]
    img = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
    return _render(img)


def _square(cy: float, cx: float, half: int) -> np.ndarray:
    img = np.zeros((GRID, GRID), dtype=np.float32)
    r0, r1 = max(0, int(cy - half)), min(GRID, int(cy + half))
    c0, c1 = max(0, int(cx - half)), min(GRID, int(cx + half))
    img[r0:r1, c0:c1] = 1.0
    return _render(img)


def _ellipse(cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    ys, xs = np.mgrid[:GRID, :GRID]
    img = ((ys - cy) ** 2 / ry ** 2 + (xs - cx) ** 2 / rx ** 2 <= 1.0).astype(np.float32)
    return _render(img)


def _heart(cy: float, cx: float, scale: float = 4.5) -> np.ndarray:
    """Parametric heart: (x²+y²-1)³ - x²y³ ≤ 0, y axis flipped."""
    ys, xs = np.mgrid[:GRID, :GRID]
    xn = (xs - cx) / scale
    yn = (cy - ys) / scale   # flip so heart points up
    img = ((xn ** 2 + yn ** 2 - 1) ** 3 - xn ** 2 * yn ** 3 <= 0).astype(np.float32)
    return _render(img)


def _bar(angle_deg: float, width: int = 1) -> np.ndarray:
    """A line bar through the centre of a 16x16 image at the given angle."""
    img = np.zeros((GRID, GRID), dtype=np.float32)
    cx, cy = GRID / 2 - 0.5, GRID / 2 - 0.5
    angle = np.deg2rad(angle_deg)
    for t in np.linspace(-GRID / 2, GRID / 2, 200):
        x = cx + t * np.cos(angle)
        y = cy + t * np.sin(angle)
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                r, c = int(round(y + dy)), int(round(x + dx))
                if 0 <= r < GRID and 0 <= c < GRID:
                    img[r, c] = 1.0
    return _render(img)


# ---------------------------------------------------------------------------
# World model builder
# ---------------------------------------------------------------------------

def build_dsprites_world_model() -> tuple[Forest, set[str]]:
    """
    Build the 5-layer dSprites world model.

    Returns
    -------
    forest : Forest
        Contains all prior nodes.
    prior_ids : set[str]
        All node IDs to pass as protected_ids to Observer.
    """
    forest = Forest(D=D, forest_id="dsprites_16x16")
    prior_ids: set[str] = set()

    def add(node: HFN) -> None:
        forest.register(node)
        prior_ids.add(node.id)

    center = GRID / 2 - 0.5  # 7.5

    # ------------------------------------------------------------------
    # Layer 0 — Perception
    # ------------------------------------------------------------------
    add(HFN(
        mu=_blank(),
        sigma=np.eye(D) * 1.0,
        id="prior_background",
    ))
    add(HFN(
        mu=np.full(D, 0.15, dtype=np.float32),
        sigma=np.eye(D) * 1.0,
        id="prior_signal",
    ))

    # ------------------------------------------------------------------
    # Layer 1 — Spatial cells (16 non-overlapping 4x4 regions)
    # ------------------------------------------------------------------
    for row in range(4):
        for col in range(4):
            img = np.zeros((GRID, GRID), dtype=np.float32)
            img[row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = 1.0
            add(HFN(
                mu=_render(img),
                sigma=np.eye(D) * 1.0,
                id=f"prior_cell_r{row}_c{col}",
            ))

    # ------------------------------------------------------------------
    # Layer 2 — Shape templates (centered, medium scale)
    # ------------------------------------------------------------------
    add(HFN(
        mu=_square(center, center, half=4),
        sigma=np.eye(D) * 1.0,
        id="prior_shape_square",
    ))
    add(HFN(
        mu=_ellipse(center, center, ry=4.0, rx=4.0),
        sigma=np.eye(D) * 1.0,
        id="prior_shape_ellipse",
    ))
    add(HFN(
        mu=_heart(center, center, scale=4.5),
        sigma=np.eye(D) * 1.0,
        id="prior_shape_heart",
    ))

    # ------------------------------------------------------------------
    # Layer 3 — Scale
    # ------------------------------------------------------------------
    add(HFN(
        mu=_square(center, center, half=2),   # 4x4 blob
        sigma=np.eye(D) * 1.0,
        id="prior_scale_small",
    ))
    add(HFN(
        mu=_square(center, center, half=4),   # 8x8 blob
        sigma=np.eye(D) * 1.0,
        id="prior_scale_medium",
    ))
    add(HFN(
        mu=_square(center, center, half=6),   # 12x12 blob
        sigma=np.eye(D) * 1.0,
        id="prior_scale_large",
    ))

    # ------------------------------------------------------------------
    # Layer 4 — Position (3x3 grid of Gaussian blobs)
    # ------------------------------------------------------------------
    positions = {
        "topleft":     (3.0,  3.0),
        "topcenter":   (3.0,  7.5),
        "topright":    (3.0, 12.0),
        "midleft":     (7.5,  3.0),
        "midcenter":   (7.5,  7.5),
        "midright":    (7.5, 12.0),
        "bottomleft":  (12.0,  3.0),
        "bottomcenter":(12.0,  7.5),
        "bottomright": (12.0, 12.0),
    }
    for name, (cy, cx) in positions.items():
        add(HFN(
            mu=_gaussian_blob(cy, cx, sigma=2.0),
            sigma=np.eye(D) * 1.0,
            id=f"prior_position_{name}",
        ))

    # ------------------------------------------------------------------
    # Layer 5 — Orientation (4 canonical directions)
    # ------------------------------------------------------------------
    for angle, name in [(90, "vertical"), (0, "horizontal"), (45, "diagonal_nw"), (135, "diagonal_ne")]:
        add(HFN(
            mu=_bar(angle),
            sigma=np.eye(D) * 1.0,
            id=f"prior_orient_{name}",
        ))

    return forest, prior_ids


def n_priors() -> int:
    """Total number of priors in the world model."""
    # 2 + 16 + 3 + 3 + 9 + 4
    return 37
