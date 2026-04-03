"""
ARC Functional Priors — SP42 Manifold-Aware.

Provides concrete functional HFN primitives for ARC laws.
Supports variable manifold dimensions (1800D for Spatial, 1850D for Explorer).
"""
from __future__ import annotations
import numpy as np
from hfn.hfn import HFN
from hpm_fractal_node.arc.arc_sovereign_loader import G_DIM, COMMON_D

def build_functional_spatial_priors(D: int = 1800) -> list[HFN]:
    """
    Builds a set of concrete geometric transformation HFNs.
    D=1800: [Input(900), Delta(900)]
    D=1850: [Input(900), Delta(900), Attr(50)]
    """
    priors = []
    
    # Slices relative to this manifold
    # Input is always first 900, Delta is always next 900
    local_i = slice(0, G_DIM)
    local_d = slice(G_DIM, G_DIM + G_DIM)
    
    # 1. Identity
    mu_id = np.zeros(D)
    sig_id = np.ones(D) * 0.1
    sig_id[local_i] = 10.0 # Any input
    sig_id[local_d] = 0.02  # Near-zero delta (identity), loose enough for minor rounding
    priors.append(HFN(mu=mu_id, sigma=sig_id, id="prior_identity", use_diag=True))
    
    # 2. Geometric Laws
    def get_flip_delta(flip_fn):
        canvas = np.zeros((30, 30))
        canvas[0:3, 0:3] = np.array([[1,2,3],[4,5,6],[7,8,9]])
        flipped = flip_fn(canvas)
        return (flipped / 9.0 - canvas / 9.0).flatten()

    def make_geo_prior(name, delta_vec):
        mu = np.zeros(D)
        mu[local_d] = delta_vec
        sig = np.ones(D) * 0.5
        sig[local_i] = 10.0 # Vague input context
        sig[local_d] = 0.05  # Loose enough to match varied content sizes
        return HFN(mu=mu, sigma=sig, id=name, use_diag=True)


    priors.append(make_geo_prior("prior_flip_v", get_flip_delta(lambda g: np.flip(g, axis=0))))
    priors.append(make_geo_prior("prior_flip_h", get_flip_delta(lambda g: np.flip(g, axis=1))))
    priors.append(make_geo_prior("prior_rot_90", get_flip_delta(lambda g: np.rot90(g, k=-1))))
    priors.append(make_geo_prior("prior_rot_180", get_flip_delta(lambda g: np.rot90(g, k=-2))))
    priors.append(make_geo_prior("prior_rot_270", get_flip_delta(lambda g: np.rot90(g, k=-3))))

    return priors

def build_functional_symbolic_priors() -> list[HFN]:
    """Builds 30D attribute HFNs."""
    D = 30
    priors = []
    mu_id = np.zeros(D); mu_id[19] = 1.0
    priors.append(HFN(mu=mu_id, sigma=np.ones(D)*0.5, id="prior_identity_rule", use_diag=True))
    mu_even = np.zeros(D); mu_even[16] = 1.0; mu_even[17] = 1.0
    priors.append(HFN(mu=mu_even, sigma=np.ones(D)*0.5, id="prior_parity_even", use_diag=True))
    return priors
