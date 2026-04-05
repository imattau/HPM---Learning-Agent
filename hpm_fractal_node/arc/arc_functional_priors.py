"""
ARC Functional Priors — SP42 Manifold-Aware.

Provides concrete functional HFN primitives for ARC laws.
Supports variable manifold dimensions (1800D for Spatial, 1850D for Explorer).
"""
from __future__ import annotations
import numpy as np
from hfn.loader import HFNLoader
from hfn.hfn import HFN
from hpm_fractal_node.arc.arc_sovereign_loader import G_DIM, COMMON_D


class FunctionalSpatialPriorLoader(HFNLoader):
    """
    Loader for geometric transformation priors in D-dimensional spatial space.

    D=1800: [Input(900), Delta(900)]
    D=1850: [Input(900), Delta(900), Attr(50)]
    """

    def __init__(self, D: int = 1800):
        self._dim = D

    @property
    def dim(self) -> int:
        return self._dim

    def build(self) -> list[HFN]:
        local_i = slice(0, G_DIM)
        local_d = slice(G_DIM, G_DIM + G_DIM)

        nodes = []

        # 1. Identity
        mu, sigma = self._base_mu_sigma(default_sigma=0.1)
        sigma[local_i] = 10.0
        sigma[local_d] = 0.02
        nodes.append(self._make_node("prior_identity", mu, sigma))

        # 2–6. Geometric transforms
        def get_flip_delta(flip_fn):
            canvas = np.zeros((30, 30))
            canvas[0:3, 0:3] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            flipped = flip_fn(canvas)
            return (flipped / 9.0 - canvas / 9.0).flatten()

        def make_geo(name, delta_vec):
            mu, sigma = self._base_mu_sigma(default_sigma=0.5)
            mu[local_d] = delta_vec
            sigma[local_i] = 10.0
            sigma[local_d] = 0.05
            return self._make_node(name, mu, sigma)

        nodes.append(make_geo("prior_flip_v",   get_flip_delta(lambda g: np.flip(g, axis=0))))
        nodes.append(make_geo("prior_flip_h",   get_flip_delta(lambda g: np.flip(g, axis=1))))
        nodes.append(make_geo("prior_rot_90",   get_flip_delta(lambda g: np.rot90(g, k=-1))))
        nodes.append(make_geo("prior_rot_180",  get_flip_delta(lambda g: np.rot90(g, k=-2))))
        nodes.append(make_geo("prior_rot_270",  get_flip_delta(lambda g: np.rot90(g, k=-3))))

        return nodes


class FunctionalSymbolicPriorLoader(HFNLoader):
    """Loader for 30D symbolic/attribute priors."""

    @property
    def dim(self) -> int:
        return 30

    def build(self) -> list[HFN]:
        nodes = []

        mu, _ = self._base_mu_sigma(default_sigma=0.5)
        mu[19] = 1.0
        nodes.append(self._make_node("prior_identity_rule", mu, np.ones(self.dim) * 0.5))

        mu, _ = self._base_mu_sigma(default_sigma=0.5)
        mu[16] = 1.0
        mu[17] = 1.0
        nodes.append(self._make_node("prior_parity_even", mu, np.ones(self.dim) * 0.5))

        return nodes


# ── Convenience wrappers (preserve existing call sites) ─────────────────────

def build_functional_spatial_priors(D: int = 1800) -> list[HFN]:
    """Returns geometric transformation prior HFNs for a D-dimensional manifold."""
    return FunctionalSpatialPriorLoader(D).build()


def build_functional_symbolic_priors() -> list[HFN]:
    """Returns 30D symbolic/attribute prior HFNs."""
    return FunctionalSymbolicPriorLoader().build()
