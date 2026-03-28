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
