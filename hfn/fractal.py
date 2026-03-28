"""
Fractal diagnostics for HFN node populations.

Box-counting dimension of the node population in μ-space measures whether
the Observer is building coherent hierarchical structure (low dimension,
nodes cluster around attractors) or just accumulating scattered points
(high dimension, nodes spread uniformly through observation space).

In an Iterated Function System (IFS) interpretation, each recombine()
call is a contracting affine map. The set of all learned nodes should
converge toward the attractor of this IFS. The fractal dimension of the
attractor is a convergence signal: decreasing dimension across passes
means structure is emerging.

Usage
-----
    from hfn.fractal import box_counting_dimension, population_dimension

    nodes = forest.active_nodes()
    dim = population_dimension(nodes, n_scales=8)
    print(f"Fractal dimension: {dim:.3f}")
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def box_counting_dimension(
    points: np.ndarray,
    n_scales: int = 8,
    eps_min: float | None = None,
    eps_max: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the box-counting (Minkowski) dimension of a point cloud.

    Uses a hash-based approach: for each box size ε, discretise each point
    by (point // ε).astype(int), then count unique discretised vectors.
    This scales O(n_points × D) and works in any dimensionality.

    Parameters
    ----------
    points : (n, D) array of floats
        Each row is a point in D-dimensional space.
    n_scales : int
        Number of ε values to evaluate (log-spaced).
    eps_min, eps_max : float or None
        Range of box sizes. Defaults to span the data range with
        geometrically spaced scales.

    Returns
    -------
    dimension : float
        Estimated box-counting dimension (slope of log N vs log 1/ε).
    log_inv_eps : (n_scales,) array
        log(1/ε) values used.
    log_counts : (n_scales,) array
        log(N(ε)) values measured.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    n, D = points.shape
    if n < 2:
        return 0.0, np.array([]), np.array([])

    # Data range along each axis
    ranges = points.max(axis=0) - points.min(axis=0)
    max_range = float(ranges.max())
    min_range = float(ranges[ranges > 0].min()) if (ranges > 0).any() else 1e-6

    if eps_max is None:
        eps_max = max_range * 0.5
    if eps_min is None:
        eps_min = max(min_range / n, max_range / (n * 10))

    if eps_min >= eps_max:
        eps_min = eps_max / 100

    epsilons = np.geomspace(eps_max, eps_min, n_scales)
    log_inv_eps = np.log(1.0 / epsilons)
    log_counts = np.zeros(n_scales)

    for i, eps in enumerate(epsilons):
        # Discretise: map each point to its box index
        boxes = np.floor(points / eps).astype(np.int32)
        # Count unique boxes using a set of tuples
        unique = len(set(map(tuple, boxes)))
        log_counts[i] = np.log(max(unique, 1))

    # Fit slope over the linear region (middle 60% of scales)
    trim = max(1, n_scales // 5)
    slope = _fit_slope(log_inv_eps[trim:-trim], log_counts[trim:-trim])

    return float(slope), log_inv_eps, log_counts


def population_dimension(nodes, n_scales: int = 8) -> float:
    """
    Estimate the box-counting dimension of an HFN node population.

    Stacks all node μ vectors into a point cloud and calls
    box_counting_dimension().

    Parameters
    ----------
    nodes : iterable of HFN
        Active nodes from a Forest.
    n_scales : int
        Number of ε scales to evaluate.

    Returns
    -------
    float
        Estimated fractal dimension of the μ-space point cloud.
        Returns 0.0 if fewer than 2 nodes are present.
    """
    mus = np.array([n.mu for n in nodes])
    if len(mus) < 2:
        return 0.0
    dim, _, _ = box_counting_dimension(mus, n_scales=n_scales)
    return dim


def dimension_profile(
    nodes_per_pass: list[list],
    n_scales: int = 8,
) -> np.ndarray:
    """
    Compute the fractal dimension at each pass.

    Parameters
    ----------
    nodes_per_pass : list of lists of HFN
        nodes_per_pass[i] is the list of active nodes after pass i.
    n_scales : int
        Number of ε scales per measurement.

    Returns
    -------
    (n_passes,) array of float
        Fractal dimension after each pass.
    """
    return np.array([population_dimension(nodes, n_scales) for nodes in nodes_per_pass])


def self_similarity_score(nodes, n_scales: int = 8) -> float:
    """
    Measure self-similarity of the node population in μ-space.

    For a true IFS attractor, the box counts N(ε_k) satisfy a power law, so
    the log-count differences Δlog N(ε_k) = log N(ε_k) - log N(ε_{k+1}) should
    be approximately constant across scales.  The coefficient of variation (CV)
    of those differences is therefore a scale-free self-similarity score:

        CV = std(Δ) / |mean(Δ)|   (lower ⟹ more self-similar)

    A perfectly self-similar attractor scores 0.0.  A uniform scatter scores
    close to 1.0 or higher because the count ratios fluctuate erratically.

    Parameters
    ----------
    nodes : iterable of HFN
        Active nodes from a Forest.
    n_scales : int
        Number of ε scales to evaluate (passed to box_counting_dimension).

    Returns
    -------
    float
        CV of log-count differences; NaN if fewer than 4 nodes.
    """
    mus = np.array([n.mu for n in nodes])
    if len(mus) < 4:
        return float("nan")
    _, _log_inv_eps, log_counts = box_counting_dimension(mus, n_scales=n_scales)
    diffs = np.diff(log_counts)
    if np.all(diffs == 0):
        return 0.0
    return float(np.std(diffs) / (np.abs(np.mean(diffs)) + 1e-9))


def hausdorff_distance(nodes_a, nodes_b) -> float:
    """
    Compute the Hausdorff distance between two node populations in μ-space.

    Hausdorff(A, B) = max(directed(A→B), directed(B→A))
    where directed(A→B) = max_{a∈A} min_{b∈B} ||a.μ - b.μ||

    This measures how far the "worst-case" node in each set is from the
    nearest node in the other set.  When used to track how learned nodes
    relate to prior nodes over passes:

        decreasing Hausdorff(learned, priors) = learned nodes are getting
        closer to the prior attractor (convergence)

        stable or increasing = learned nodes are exploring novel territory

    Parameters
    ----------
    nodes_a, nodes_b : iterables of HFN
        Two node populations.

    Returns
    -------
    float
        Hausdorff distance in μ-space.  Returns inf if either set is empty.
    """
    mus_a = np.array([n.mu for n in nodes_a])
    mus_b = np.array([n.mu for n in nodes_b])
    if len(mus_a) == 0 or len(mus_b) == 0:
        return float("inf")

    # directed A→B: for each point in A, find minimum distance to any point in B
    dists_ab = np.array([
        float(np.min(np.linalg.norm(mus_b - a, axis=1)))
        for a in mus_a
    ])
    # directed B→A
    dists_ba = np.array([
        float(np.min(np.linalg.norm(mus_a - b, axis=1)))
        for b in mus_b
    ])
    return float(max(dists_ab.max(), dists_ba.max()))


def correlation_dimension(
    nodes,
    n_scales: int = 8,
    eps_min: float | None = None,
    eps_max: float | None = None,
) -> float:
    """
    Estimate the correlation dimension of an HFN node population (Grassberger-Procaccia).

    More accurate than box-counting for small populations (n < 50). Measures
    how the fraction of node pairs within distance r scales with r:

        C(r) = (# pairs with ||μᵢ - μⱼ|| < r) / (n*(n-1)/2)
        correlation_dimension ≈ slope of log C(r) vs log r

    Parameters
    ----------
    nodes : iterable of HFN
    n_scales : int
        Number of r values to evaluate.

    Returns
    -------
    float
        Estimated correlation dimension. Returns 0.0 if fewer than 3 nodes.
    """
    mus = np.array([n.mu for n in nodes])
    if len(mus) < 3:
        return 0.0

    # Pairwise distances (upper triangle only)
    n = len(mus)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(mus[i] - mus[j])))
    dists = np.array(dists)

    max_d = dists.max()
    min_d = dists[dists > 0].min() if (dists > 0).any() else 1e-9
    if eps_max is None:
        eps_max = max_d * 0.5
    if eps_min is None:
        eps_min = min_d
    if eps_min >= eps_max:
        eps_min = eps_max / 100

    radii = np.geomspace(eps_min, eps_max, n_scales)
    log_r = np.log(radii)
    log_c = np.zeros(n_scales)
    total_pairs = len(dists)

    for i, r in enumerate(radii):
        count = float(np.sum(dists < r))
        log_c[i] = np.log(max(count / total_pairs, 1e-12))

    trim = max(1, n_scales // 5)
    return float(_fit_slope(log_r[trim:-trim], log_c[trim:-trim]))


def information_dimension(nodes, weights: dict[str, float], n_scales: int = 8) -> float:
    """
    Estimate the information dimension of an HFN node population.

    Unlike box-counting (which counts occupied boxes equally), information
    dimension weights each box by the total weight of nodes inside it:

        I(ε) = -Σ p_i * log(p_i)   where p_i = weight_i / total_weight
        information_dimension ≈ slope of I(ε) vs log(1/ε)

    This measures the fractal dimension of the *active* representation —
    heavily-used nodes contribute more than dormant ones.

    Parameters
    ----------
    nodes : iterable of HFN
    weights : dict mapping node.id → float weight (from Observer._weights)
    n_scales : int

    Returns
    -------
    float
        Estimated information dimension. Returns 0.0 if fewer than 2 nodes.
    """
    node_list = list(nodes)
    if len(node_list) < 2:
        return 0.0

    mus = np.array([n.mu for n in node_list])
    ws = np.array([max(weights.get(n.id, 0.0), 1e-12) for n in node_list])
    ws = ws / ws.sum()

    ranges = mus.max(axis=0) - mus.min(axis=0)
    max_range = float(ranges.max())
    if max_range == 0:
        return 0.0

    eps_max = max_range * 0.5
    eps_min = max_range / (len(mus) * 10)
    if eps_min >= eps_max:
        eps_min = eps_max / 100

    epsilons = np.geomspace(eps_max, eps_min, n_scales)
    log_inv_eps = np.log(1.0 / epsilons)
    log_info = np.zeros(n_scales)

    for i, eps in enumerate(epsilons):
        boxes = np.floor(mus / eps).astype(np.int32)
        box_weights: dict[tuple, float] = {}
        for idx, box in enumerate(map(tuple, boxes)):
            box_weights[box] = box_weights.get(box, 0.0) + ws[idx]
        ps = np.array(list(box_weights.values()))
        ps = ps[ps > 0]
        entropy = float(-np.sum(ps * np.log(ps)))
        log_info[i] = entropy

    trim = max(1, n_scales // 5)
    return float(_fit_slope(log_inv_eps[trim:-trim], log_info[trim:-trim]))


def intrinsic_dimensionality(points: np.ndarray) -> float:
    """
    Estimate the intrinsic dimensionality of a point cloud using the TwoNN estimator.

    Measures the true degrees of freedom in the data, independent of the ambient
    dimension D. For ARC observations this answers: how many independent axes of
    variation exist in the grid patterns?

        If intrinsic_dim ≈ 2–3: patterns vary along a few dominant axes
        If intrinsic_dim ≈ D:   patterns fill the full observation space

    Uses the Two Nearest Neighbours (TwoNN) method (Facco et al. 2017):
        μᵢ = d(x, 2nd-nearest) / d(x, 1st-nearest)
        empirical CDF of μ fits P(μ) = 1 - μ^(-d)
        d = 1 / mean(log μ)

    Parameters
    ----------
    points : (n, D) array
        Point cloud in ambient space (e.g. stacked observation vectors).

    Returns
    -------
    float
        Estimated intrinsic dimensionality. Returns nan if fewer than 3 points.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    n = len(points)
    if n < 3:
        return float("nan")

    mu_vals = []
    for i in range(n):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf  # exclude self
        sorted_d = np.sort(dists)
        r1, r2 = sorted_d[0], sorted_d[1]
        if r1 > 0:
            mu_vals.append(r2 / r1)

    if not mu_vals:
        return float("nan")
    return float(1.0 / np.mean(np.log(mu_vals)))


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y vs x."""
    if len(x) < 2:
        return 0.0
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(slope)
    except Exception:
        return 0.0
