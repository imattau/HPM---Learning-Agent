"""
ARC Relational Priors — 80D rule-family HFN nodes.
"""
from __future__ import annotations
import numpy as np
from hfn.loader import HFNLoader, LoadItem
from hfn.hfn import HFN
from hpm_fractal_node.arc.arc_relational_encoder import RD_DIM, K, D_SLOT


class RelationalPriorLoader(HFNLoader):
    """Loader for the 5 relational-delta priors in 80D space."""

    namespace = "arc_rel"

    @property
    def dim(self) -> int:
        return RD_DIM

    def build(self) -> list[HFN] | list[LoadItem]:
        return [
            self._item(self._identity(), role="prior", protected=True),
            self._item(self._translate(), role="prior", protected=True),
            self._item(self._recolor(), role="prior", protected=True),
            self._item(self._count_up(), role="prior", protected=True),
            self._item(self._count_down(), role="prior", protected=True),
        ]

    def _make_prior(self, pid: str, mu: np.ndarray, sigma_base: float,
                    slot_overrides: dict | None = None) -> HFN:
        sig = np.ones(RD_DIM) * sigma_base
        if slot_overrides:
            for idx, val in slot_overrides.items():
                sig[idx] = val
        return self._make_node(pid, mu, sig)

    def _identity(self) -> HFN:
        """All deltas zero — objects unchanged."""
        mu = np.zeros(RD_DIM)
        mu[73] = 1.0   # all shapes preserved
        return self._make_prior("prior_rd_identity", mu, 0.05)

    def _translate(self) -> HFN:
        """Consistent Δrow/Δcol, no color/shape change."""
        mu = np.zeros(RD_DIM)
        mu[71] = 1.0   # all objects moved
        mu[73] = 1.0   # shapes preserved
        overrides = {}
        for i in range(K):
            overrides[i * D_SLOT + 0] = 0.15  # Δrow: moderately loose
            overrides[i * D_SLOT + 1] = 0.15  # Δcol: moderately loose
            overrides[i * D_SLOT + 2] = 0.05  # Δcolor: tight
        return self._make_prior("prior_rd_translate", mu, 0.05, overrides)

    def _recolor(self) -> HFN:
        """No position change, consistent color change."""
        mu = np.zeros(RD_DIM)
        mu[72] = 1.0   # all objects recolored
        mu[73] = 1.0   # shapes preserved
        overrides = {}
        for i in range(K):
            overrides[i * D_SLOT + 0] = 0.05  # Δrow: tight
            overrides[i * D_SLOT + 1] = 0.05  # Δcol: tight
            overrides[i * D_SLOT + 2] = 0.3   # Δcolor: loose
        return self._make_prior("prior_rd_recolor", mu, 0.05, overrides)

    def _count_up(self) -> HFN:
        """New objects appear."""
        mu = np.zeros(RD_DIM)
        mu[70] = 0.3   # positive count delta
        mu[75] = 0.5   # new objects fraction
        return self._make_prior("prior_rd_count_up", mu, 0.05)

    def _count_down(self) -> HFN:
        """Objects are deleted."""
        mu = np.zeros(RD_DIM)
        mu[70] = -0.3
        mu[74] = 0.5   # deleted fraction
        return self._make_prior("prior_rd_count_down", mu, 0.05)


def build_relational_priors() -> list[HFN]:
    """Convenience wrapper — returns the 5 relational prior HFNs."""
    return RelationalPriorLoader().build_nodes()
