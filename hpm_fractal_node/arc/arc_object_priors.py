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


def _make_prior(name: str, mu: np.ndarray, sigma: np.ndarray) -> HFN:
    return HFN(mu=mu, sigma=sigma, id=name, use_diag=True)


def _base_mu_sigma(tight_input: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) both shaped (420,) with vague defaults."""
    mu = np.zeros(TOTAL_DIM)
    sigma = np.ones(TOTAL_DIM) * 5.0   # vague everywhere by default
    if tight_input:
        # Input objects: vague (we don't know what the input looks like)
        sigma[IN_SLICE] = 5.0
    return mu, sigma


# ─── 1. Identity ────────────────────────────────────────────────────────────
def _prior_identity_obj() -> HFN:
    """Output objects are the same as input objects (no change)."""
    mu, sigma = _base_mu_sigma()

    # Output objects should equal input objects → delta = 0
    # Encode: for each of K object slots, output descriptor ≈ input descriptor
    # Rule summary: no color change (bucket 4 = delta 0 → value ≈ 1), no move, no resize
    mu[RULE_SLICE][4] = 1.0    # color delta = 0 bucket
    mu[RULE_SLICE][9]  = 0.0   # no position change
    mu[RULE_SLICE][10] = 0.0   # no size change
    mu[RULE_SLICE][11] = 1.0   # count ratio = 1.0

    # Tight rule summary
    sigma[RULE_SLICE] = 0.1
    # Input and output object vectors should be similar — encode mild constraint
    sigma[IN_SLICE]  = 3.0
    sigma[OUT_SLICE] = 3.0

    return _make_prior("prior_identity_obj", mu, sigma)


# ─── 2. Recolor ─────────────────────────────────────────────────────────────
def _prior_recolor() -> HFN:
    """Output objects keep position/shape but change colors."""
    mu, sigma = _base_mu_sigma()

    # Position and shape dimensions tight between input and output
    # Color dimension loose (it changes). Rule: color delta != 0, no position change
    mu[RULE_SLICE][9]  = 0.0   # position unchanged
    mu[RULE_SLICE][10] = 0.0   # size unchanged
    mu[RULE_SLICE][11] = 1.0   # same count

    sigma[RULE_SLICE] = 0.15
    # Position/shape output ≈ input: tighten those dims, loosen color dims
    for slot in range(K):
        base_in  = IN_SLICE.start  + slot * D_OBJ
        base_out = OUT_SLICE.start + slot * D_OBJ
        # Color (dim 0): loose — it changes
        sigma[base_out + _IDX_COLOR] = 2.0
        # Position (dims 1-2): tight
        sigma[base_out + _IDX_ROW] = 0.05
        sigma[base_out + _IDX_COL] = 0.05
        # Size (dims 3-5): tight
        sigma[base_out + _IDX_HEIGHT] = 0.05
        sigma[base_out + _IDX_WIDTH]  = 0.05
        sigma[base_out + _IDX_AREA]   = 0.05

    sigma[IN_SLICE]  = 3.0

    return _make_prior("prior_recolor", mu, sigma)


# ─── 3. Translate ────────────────────────────────────────────────────────────
def _prior_translate() -> HFN:
    """Objects are moved (translated) but keep color and shape."""
    mu, sigma = _base_mu_sigma()

    mu[RULE_SLICE][9]  = 1.0   # position changes
    mu[RULE_SLICE][10] = 0.0   # size unchanged
    mu[RULE_SLICE][11] = 1.0   # same count

    sigma[RULE_SLICE] = 0.2
    # Color and shape tight between in/out; position loose
    for slot in range(K):
        base_out = OUT_SLICE.start + slot * D_OBJ
        sigma[base_out + _IDX_COLOR]  = 0.05   # color preserved
        sigma[base_out + _IDX_ROW]    = 3.0    # position changes
        sigma[base_out + _IDX_COL]    = 3.0
        sigma[base_out + _IDX_HEIGHT] = 0.05   # shape preserved
        sigma[base_out + _IDX_WIDTH]  = 0.05
        sigma[base_out + _IDX_AREA]   = 0.05

    sigma[IN_SLICE] = 3.0

    return _make_prior("prior_translate", mu, sigma)


# ─── 4. Reflect horizontal ──────────────────────────────────────────────────
def _prior_reflect_h() -> HFN:
    """Objects are reflected horizontally (left-right flip)."""
    mu, sigma = _base_mu_sigma()

    # Col centers flip: col_out ≈ (1 - col_in) when normalized 0-1
    # Row centers stay the same
    mu[RULE_SLICE][9]  = 1.0   # positions change
    mu[RULE_SLICE][10] = 0.0   # sizes unchanged
    mu[RULE_SLICE][11] = 1.0

    sigma[RULE_SLICE] = 0.2
    for slot in range(K):
        base_in  = IN_SLICE.start  + slot * D_OBJ
        base_out = OUT_SLICE.start + slot * D_OBJ
        # Row center: same
        sigma[base_out + _IDX_ROW]   = 0.05
        # Col center: reflects
        sigma[base_out + _IDX_COL]   = 0.15
        sigma[base_out + _IDX_COLOR] = 0.05
        sigma[base_out + _IDX_AREA]  = 0.05

    sigma[IN_SLICE] = 3.0

    return _make_prior("prior_reflect_h", mu, sigma)


# ─── 5. Reflect vertical ────────────────────────────────────────────────────
def _prior_reflect_v() -> HFN:
    """Objects are reflected vertically (top-bottom flip)."""
    mu, sigma = _base_mu_sigma()

    mu[RULE_SLICE][9]  = 1.0
    mu[RULE_SLICE][10] = 0.0
    mu[RULE_SLICE][11] = 1.0

    sigma[RULE_SLICE] = 0.2
    for slot in range(K):
        base_out = OUT_SLICE.start + slot * D_OBJ
        sigma[base_out + _IDX_ROW]   = 0.15   # row reflects
        sigma[base_out + _IDX_COL]   = 0.05   # col stays
        sigma[base_out + _IDX_COLOR] = 0.05
        sigma[base_out + _IDX_AREA]  = 0.05

    sigma[IN_SLICE] = 3.0

    return _make_prior("prior_reflect_v", mu, sigma)


# ─── 6. Sort by size ─────────────────────────────────────────────────────────
def _prior_sort_by_size() -> HFN:
    """Objects are reordered by size (area), positions may change."""
    mu, sigma = _base_mu_sigma()

    mu[RULE_SLICE][9]  = 1.0   # positions change (reordering)
    mu[RULE_SLICE][10] = 0.0   # individual sizes unchanged
    mu[RULE_SLICE][11] = 1.0   # same count

    sigma[RULE_SLICE] = 0.2
    sigma[IN_SLICE]  = 3.0
    sigma[OUT_SLICE] = 3.0   # positions and colors all loose

    return _make_prior("prior_sort_by_size", mu, sigma)


# ─── 7. Stamp ────────────────────────────────────────────────────────────────
def _prior_stamp() -> HFN:
    """A template object is stamped at each input object position."""
    mu, sigma = _base_mu_sigma()

    # Count may increase; color/shape of output objects may differ from input
    mu[RULE_SLICE][10] = 1.0   # size typically changes (stamping a pattern)
    mu[RULE_SLICE][11] = 1.0   # count roughly preserved

    sigma[RULE_SLICE] = 0.5   # loose — stamp rules are diverse
    sigma[IN_SLICE]  = 3.0
    sigma[OUT_SLICE] = 3.0

    return _make_prior("prior_stamp", mu, sigma)


# ─── 8. Fill column ──────────────────────────────────────────────────────────
def _prior_fill_column() -> HFN:
    """Objects fill vertically along their column."""
    mu, sigma = _base_mu_sigma()

    # Objects grow in height (fill column), same color
    mu[RULE_SLICE][9]  = 1.0   # position changes (center moves)
    mu[RULE_SLICE][10] = 1.0   # size changes
    mu[RULE_SLICE][11] = 1.0

    sigma[RULE_SLICE] = 0.3
    for slot in range(K):
        base_out = OUT_SLICE.start + slot * D_OBJ
        sigma[base_out + _IDX_COLOR]  = 0.05   # color preserved
        sigma[base_out + _IDX_HEIGHT] = 1.0    # height grows
        sigma[base_out + _IDX_WIDTH]  = 0.1    # width stays
        sigma[base_out + _IDX_ROW]    = 1.0    # row center shifts

    sigma[IN_SLICE] = 3.0

    return _make_prior("prior_fill_column", mu, sigma)


# ─── 9. Fill row ─────────────────────────────────────────────────────────────
def _prior_fill_row() -> HFN:
    """Objects fill horizontally along their row."""
    mu, sigma = _base_mu_sigma()

    mu[RULE_SLICE][9]  = 1.0
    mu[RULE_SLICE][10] = 1.0
    mu[RULE_SLICE][11] = 1.0

    sigma[RULE_SLICE] = 0.3
    for slot in range(K):
        base_out = OUT_SLICE.start + slot * D_OBJ
        sigma[base_out + _IDX_COLOR]  = 0.05
        sigma[base_out + _IDX_HEIGHT] = 0.1    # height stays
        sigma[base_out + _IDX_WIDTH]  = 1.0    # width grows
        sigma[base_out + _IDX_COL]    = 1.0    # col center shifts

    sigma[IN_SLICE] = 3.0

    return _make_prior("prior_fill_row", mu, sigma)


# ─── 10. Count rule ──────────────────────────────────────────────────────────
def _prior_count_rule() -> HFN:
    """Output object count encodes a property of the input (e.g., area → count)."""
    mu, sigma = _base_mu_sigma()

    # Count changes, positions of output objects differ from input
    mu[RULE_SLICE][9]  = 1.0
    mu[RULE_SLICE][10] = 0.0
    mu[RULE_SLICE][11] = 0.5   # count often differs significantly

    sigma[RULE_SLICE] = 0.5   # very loose — count rules are diverse
    sigma[IN_SLICE]  = 3.0
    sigma[OUT_SLICE] = 3.0

    return _make_prior("prior_count_rule", mu, sigma)


def build_object_level_priors() -> list[HFN]:
    """
    Build and return the full list of 420D object-level prior HFNs.

    Returns:
        list[HFN] — 10 prior nodes, each with mu.shape == (420,)
    """
    return [
        _prior_identity_obj(),
        _prior_recolor(),
        _prior_translate(),
        _prior_reflect_h(),
        _prior_reflect_v(),
        _prior_sort_by_size(),
        _prior_stamp(),
        _prior_fill_column(),
        _prior_fill_row(),
        _prior_count_rule(),
    ]
