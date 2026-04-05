"""
ARC Object-Level HFN Priors — 420D Object Space.

Builds a set of object-level prior HFNs for the 420D manifold:
  [0-199]   Input objects  (K=10, 10×20D each)
  [200-399] Output objects (K=10, 10×20D each)
  [400-419] Rule summary   (20D)

Each prior encodes a hypothesis about how input objects relate to output
objects under a named transformation rule.
"""
from __future__ import annotations

import numpy as np
from hfn.loader import HFNLoader, LoadItem
from hfn.hfn import HFN
from hpm_fractal_node.arc.arc_object_encoder import TOTAL_DIM, IN_SLICE, OUT_SLICE, RULE_SLICE, D_OBJ, K

# Indices within a single 20D object descriptor
_IDX_COLOR   = 0
_IDX_ROW     = 1
_IDX_COL     = 2
_IDX_HEIGHT  = 3
_IDX_WIDTH   = 4
_IDX_AREA    = 5
# [6-15] shape fingerprint, [16-19] reserved


class ObjectPriorLoader(HFNLoader):
    """Loader for the 10 object-level priors in 420D space."""

    namespace = "arc_obj"

    @property
    def dim(self) -> int:
        return TOTAL_DIM

    def build(self) -> list[HFN] | list[LoadItem]:
        return [
            self._item(self._identity(), role="prior", protected=True),
            self._item(self._recolor(), role="prior", protected=True),
            self._item(self._translate(), role="prior", protected=True),
            self._item(self._reflect_h(), role="prior", protected=True),
            self._item(self._reflect_v(), role="prior", protected=True),
            self._item(self._sort_by_size(), role="prior", protected=True),
            self._item(self._stamp(), role="prior", protected=True),
            self._item(self._fill_column(), role="prior", protected=True),
            self._item(self._fill_row(), role="prior", protected=True),
            self._item(self._count_rule(), role="prior", protected=True),
        ]

    def _base_mu_sigma(self, default_sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        return np.zeros(self.dim), np.ones(self.dim) * default_sigma

    # ── Priors ───────────────────────────────────────────────────────────────

    def _identity(self) -> HFN:
        """Output objects are the same as input objects (no change)."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][4]  = 1.0   # color delta = 0 bucket
        mu[RULE_SLICE][9]  = 0.0   # no position change
        mu[RULE_SLICE][10] = 0.0   # no size change
        mu[RULE_SLICE][11] = 1.0   # count ratio = 1.0
        sigma[RULE_SLICE]  = 0.1
        sigma[IN_SLICE]    = 3.0
        sigma[OUT_SLICE]   = 3.0
        return self._make_node("prior_identity_obj", mu, sigma)

    def _recolor(self) -> HFN:
        """Output objects keep position/shape but change colors."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 0.0
        mu[RULE_SLICE][10] = 0.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.15
        for slot in range(K):
            base_out = OUT_SLICE.start + slot * D_OBJ
            sigma[base_out + _IDX_COLOR]  = 2.0
            sigma[base_out + _IDX_ROW]    = 0.05
            sigma[base_out + _IDX_COL]    = 0.05
            sigma[base_out + _IDX_HEIGHT] = 0.05
            sigma[base_out + _IDX_WIDTH]  = 0.05
            sigma[base_out + _IDX_AREA]   = 0.05
        sigma[IN_SLICE] = 3.0
        return self._make_node("prior_recolor", mu, sigma)

    def _translate(self) -> HFN:
        """Objects are moved (translated) but keep color and shape."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 1.0
        mu[RULE_SLICE][10] = 0.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.2
        for slot in range(K):
            base_out = OUT_SLICE.start + slot * D_OBJ
            sigma[base_out + _IDX_COLOR]  = 0.05
            sigma[base_out + _IDX_ROW]    = 3.0
            sigma[base_out + _IDX_COL]    = 3.0
            sigma[base_out + _IDX_HEIGHT] = 0.05
            sigma[base_out + _IDX_WIDTH]  = 0.05
            sigma[base_out + _IDX_AREA]   = 0.05
        sigma[IN_SLICE] = 3.0
        return self._make_node("prior_translate", mu, sigma)

    def _reflect_h(self) -> HFN:
        """Objects are reflected horizontally (left-right flip)."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 1.0
        mu[RULE_SLICE][10] = 0.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.2
        for slot in range(K):
            base_out = OUT_SLICE.start + slot * D_OBJ
            sigma[base_out + _IDX_ROW]   = 0.05
            sigma[base_out + _IDX_COL]   = 0.15
            sigma[base_out + _IDX_COLOR] = 0.05
            sigma[base_out + _IDX_AREA]  = 0.05
        sigma[IN_SLICE] = 3.0
        return self._make_node("prior_reflect_h", mu, sigma)

    def _reflect_v(self) -> HFN:
        """Objects are reflected vertically (top-bottom flip)."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 1.0
        mu[RULE_SLICE][10] = 0.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.2
        for slot in range(K):
            base_out = OUT_SLICE.start + slot * D_OBJ
            sigma[base_out + _IDX_ROW]   = 0.15
            sigma[base_out + _IDX_COL]   = 0.05
            sigma[base_out + _IDX_COLOR] = 0.05
            sigma[base_out + _IDX_AREA]  = 0.05
        sigma[IN_SLICE] = 3.0
        return self._make_node("prior_reflect_v", mu, sigma)

    def _sort_by_size(self) -> HFN:
        """Objects are reordered by size (area), positions may change."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 1.0
        mu[RULE_SLICE][10] = 0.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.2
        sigma[IN_SLICE]    = 3.0
        sigma[OUT_SLICE]   = 3.0
        return self._make_node("prior_sort_by_size", mu, sigma)

    def _stamp(self) -> HFN:
        """A template object is stamped at each input object position."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][10] = 1.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.5
        sigma[IN_SLICE]    = 3.0
        sigma[OUT_SLICE]   = 3.0
        return self._make_node("prior_stamp", mu, sigma)

    def _fill_column(self) -> HFN:
        """Objects fill vertically along their column."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 1.0
        mu[RULE_SLICE][10] = 1.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.3
        for slot in range(K):
            base_out = OUT_SLICE.start + slot * D_OBJ
            sigma[base_out + _IDX_COLOR]  = 0.05
            sigma[base_out + _IDX_HEIGHT] = 1.0
            sigma[base_out + _IDX_WIDTH]  = 0.1
            sigma[base_out + _IDX_ROW]    = 1.0
        sigma[IN_SLICE] = 3.0
        return self._make_node("prior_fill_column", mu, sigma)

    def _fill_row(self) -> HFN:
        """Objects fill horizontally along their row."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 1.0
        mu[RULE_SLICE][10] = 1.0
        mu[RULE_SLICE][11] = 1.0
        sigma[RULE_SLICE]  = 0.3
        for slot in range(K):
            base_out = OUT_SLICE.start + slot * D_OBJ
            sigma[base_out + _IDX_COLOR]  = 0.05
            sigma[base_out + _IDX_HEIGHT] = 0.1
            sigma[base_out + _IDX_WIDTH]  = 1.0
            sigma[base_out + _IDX_COL]    = 1.0
        sigma[IN_SLICE] = 3.0
        return self._make_node("prior_fill_row", mu, sigma)

    def _count_rule(self) -> HFN:
        """Output object count encodes a property of the input."""
        mu, sigma = self._base_mu_sigma()
        mu[RULE_SLICE][9]  = 1.0
        mu[RULE_SLICE][10] = 0.0
        mu[RULE_SLICE][11] = 0.5
        sigma[RULE_SLICE]  = 0.5
        sigma[IN_SLICE]    = 3.0
        sigma[OUT_SLICE]   = 3.0
        return self._make_node("prior_count_rule", mu, sigma)


def build_object_level_priors() -> list[HFN]:
    """Convenience wrapper — returns the 10 object-level prior HFNs."""
    return ObjectPriorLoader().build_nodes()
