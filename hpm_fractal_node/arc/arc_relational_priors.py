"""
ARC Relational Priors — 80D rule-family HFN nodes.
"""
from __future__ import annotations
import numpy as np
from hfn.hfn import HFN
from hpm_fractal_node.arc.arc_relational_encoder import RD_DIM, K, D_SLOT


def build_relational_priors() -> list[HFN]:
    """Build 80D HFN priors encoding common rule families."""
    priors = []

    def make_prior(pid: str, mu: np.ndarray, sigma_base: float, slot_overrides: dict = None) -> HFN:
        sig = np.ones(RD_DIM) * sigma_base
        if slot_overrides:
            for idx, val in slot_overrides.items():
                sig[idx] = val
        return HFN(mu=mu, sigma=sig, id=pid, use_diag=True)

    # Identity: all deltas zero
    # sigma_base=0.05 keeps mean(sigma) <= 0.1 so CognitiveSolver's target_var filter passes
    mu = np.zeros(RD_DIM)
    mu[73] = 1.0   # all shapes preserved
    priors.append(make_prior("prior_rd_identity", mu, 0.05))

    # Pure translation: consistent Δrow/Δcol, no color/shape change
    mu = np.zeros(RD_DIM)
    mu[71] = 1.0   # all objects moved
    mu[73] = 1.0   # shapes preserved
    # Loosen position slots so varied translations match; tighten color slots
    overrides = {}
    for i in range(K):
        overrides[i*D_SLOT + 0] = 0.15  # Δrow: moderately loose (varies per task)
        overrides[i*D_SLOT + 1] = 0.15  # Δcol: moderately loose
        overrides[i*D_SLOT + 2] = 0.05  # Δcolor: tight (no recolor)
    priors.append(make_prior("prior_rd_translate", mu, 0.05, overrides))

    # Recolor: no position change, consistent color change
    mu = np.zeros(RD_DIM)
    mu[72] = 1.0   # all objects recolored
    mu[73] = 1.0   # shapes preserved
    overrides = {}
    for i in range(K):
        overrides[i*D_SLOT + 0] = 0.05  # Δrow: tight (no translation)
        overrides[i*D_SLOT + 1] = 0.05  # Δcol: tight
        overrides[i*D_SLOT + 2] = 0.3   # Δcolor: loose (it changes)
    priors.append(make_prior("prior_rd_recolor", mu, 0.05, overrides))

    # Count increase: new objects appear
    mu = np.zeros(RD_DIM)
    mu[70] = 0.3   # positive count delta
    mu[75] = 0.5   # new objects fraction
    priors.append(make_prior("prior_rd_count_up", mu, 0.05))

    # Count decrease: objects deleted
    mu = np.zeros(RD_DIM)
    mu[70] = -0.3
    mu[74] = 0.5   # deleted fraction
    priors.append(make_prior("prior_rd_count_down", mu, 0.05))

    return priors
