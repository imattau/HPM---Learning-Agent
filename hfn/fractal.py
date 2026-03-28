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


def persistence_scores(nodes, weights: dict[str, float]) -> dict[str, float]:
    """
    Per-node persistence using a simplified elder-rule approach.

    Persistence = how structurally isolated a node is from higher-weighted nodes.
    Computed as the distance to the nearest node with a strictly higher weight.

    - High persistence: far from any more significant node → structurally important
    - Low persistence: close to a more significant node → potentially redundant

    The highest-weight node gets persistence = inf (no more significant node exists).

    Parameters
    ----------
    nodes : iterable of HFN
    weights : dict mapping node.id → float weight

    Returns
    -------
    dict mapping node.id → float persistence
    """
    node_list = sorted(nodes, key=lambda n: -weights.get(n.id, 0.0))
    if not node_list:
        return {}
    mus = np.array([n.mu for n in node_list])
    scores: dict[str, float] = {}
    for i, node in enumerate(node_list):
        if i == 0:
            scores[node.id] = float("inf")
        else:
            prev_mus = mus[:i]
            dist = float(np.min(np.linalg.norm(prev_mus - mus[i], axis=1)))
            scores[node.id] = dist
    return scores


class RecurrenceTracker:
    """
    Measures how repetitive the observation stream is.

    Maintains a rolling buffer of recent observations and tracks what fraction
    of new observations fall within epsilon of a previous one (recurrence_rate).

    High recurrence_rate → stream is repetitive → compress more aggressively.
    Low recurrence_rate → stream is novel → keep observations distinct.
    """

    def __init__(self, buffer_size: int = 50, epsilon: float = 0.3):
        self._buffer: list[np.ndarray] = []
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.recurrence_rate: float = 0.0

    def update(self, x: np.ndarray) -> float:
        """
        Add a new observation and return the updated recurrence rate.

        Parameters
        ----------
        x : np.ndarray
            New observation vector.

        Returns
        -------
        float
            Current recurrence_rate in [0, 1].
        """
        if self._buffer:
            hits = sum(1 for b in self._buffer if np.linalg.norm(x - b) < self.epsilon)
            self.recurrence_rate = 0.9 * self.recurrence_rate + 0.1 * (hits / len(self._buffer))
        self._buffer.append(x.copy())
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)
        return self.recurrence_rate

    def recommended_threshold(self, base: int, min_t: int = 2, max_t: int = 10) -> int:
        """
        Return an adjusted compression co-occurrence threshold.

        High recurrence → lower threshold (compress more aggressively).
        Low recurrence → higher threshold (observations are novel, preserve them).

        Parameters
        ----------
        base : int
            Default/base threshold value.
        min_t, max_t : int
            Clamp the output to this range.

        Returns
        -------
        int
            Adjusted threshold.
        """
        adjusted = base * (1.0 - 0.5 * self.recurrence_rate)
        return max(min_t, min(max_t, round(adjusted)))


def lacunarity(nodes, n_scales: int = 8) -> np.ndarray:
    """
    Compute the lacunarity of a node population in μ-space.

    Lacunarity λ(ε) = σ²/μ² of box occupancy counts at each scale ε.

    Low lacunarity → uniformly distributed nodes (regular spacing).
    High lacunarity → clustered with gaps (heterogeneous structure).

    This is a diagnostic-only measure — it does not drive Observer decisions.

    Parameters
    ----------
    nodes : iterable of HFN
    n_scales : int
        Number of ε scales to evaluate.

    Returns
    -------
    (n_scales, 2) array of (ε, λ(ε)) pairs, or empty array if < 2 nodes.
    """
    mus = np.array([n.mu for n in nodes])
    if len(mus) < 2:
        return np.array([])
    ranges = mus.max(axis=0) - mus.min(axis=0)
    max_range = float(ranges.max())
    if max_range == 0:
        return np.array([])
    eps_max = max_range * 0.5
    eps_min = max_range / (len(mus) * 5)
    if eps_min >= eps_max:
        eps_min = eps_max / 100
    epsilons = np.geomspace(eps_max, eps_min, n_scales)
    results = []
    for eps in epsilons:
        boxes: dict[tuple, int] = {}
        for mu in mus:
            key = tuple(np.floor(mu / eps).astype(int))
            boxes[key] = boxes.get(key, 0) + 1
        counts = np.array(list(boxes.values()), dtype=float)
        mean_c = counts.mean()
        if mean_c > 0:
            lac = counts.var() / (mean_c ** 2)
        else:
            lac = 0.0
        results.append((eps, lac))
    return np.array(results)


def multifractal_spectrum(
    nodes,
    q_values: np.ndarray | None = None,
    n_scales: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the multifractal spectrum (generalised dimensions D_q).

    D_q is the generalised Rényi dimension for moment order q:
    - D_0 = box-counting dimension
    - D_1 = information dimension
    - D_2 = correlation dimension

    For a monofractal all D_q are equal; a spread in D_q indicates multifractal
    structure (the node population has qualitatively different densities at
    different scales).

    This is a diagnostic-only measure — it does not drive Observer decisions.

    Parameters
    ----------
    nodes : iterable of HFN
    q_values : array of q moments; defaults to linspace(-3, 5, 17)
    n_scales : int
        Number of ε scales to evaluate.

    Returns
    -------
    (q_values, D_q) : two arrays of the same length.
        D_q values are NaN when the node population is too small (< 3 nodes).
    """
    if q_values is None:
        q_values = np.linspace(-3, 5, 17)
    q_values = np.asarray(q_values, dtype=float)

    mus = np.array([n.mu for n in nodes])
    if len(mus) < 3:
        return q_values, np.full(len(q_values), float("nan"))

    ranges = mus.max(axis=0) - mus.min(axis=0)
    max_range = float(ranges.max())
    if max_range == 0:
        return q_values, np.full(len(q_values), float("nan"))

    eps_max = max_range * 0.5
    eps_min = max_range / (len(mus) * 5)
    if eps_min >= eps_max:
        eps_min = eps_max / 100
    epsilons = np.geomspace(eps_max, eps_min, n_scales)

    D_q = np.full(len(q_values), float("nan"))

    for qi, q in enumerate(q_values):
        log_inv_eps = np.log(1.0 / epsilons)
        log_Z = np.zeros(n_scales)

        for i, eps in enumerate(epsilons):
            boxes: dict[tuple, int] = {}
            for mu in mus:
                key = tuple(np.floor(mu / eps).astype(int))
                boxes[key] = boxes.get(key, 0) + 1
            counts = np.array(list(boxes.values()), dtype=float)
            total = counts.sum()
            if total == 0:
                log_Z[i] = float("nan")
                continue
            ps = counts / total
            ps = ps[ps > 0]

            if abs(q - 1.0) < 1e-6:
                # q=1: information dimension — use Shannon entropy
                log_Z[i] = float(-np.sum(ps * np.log(ps)))
            else:
                z = float(np.sum(ps ** q))
                log_Z[i] = np.log(max(z, 1e-300))

        valid = np.isfinite(log_Z)
        if valid.sum() < 2:
            continue

        trim = max(1, n_scales // 5)
        x = log_inv_eps[trim:-trim][valid[trim:-trim]]
        y = log_Z[trim:-trim][valid[trim:-trim]]
        if len(x) < 2:
            continue

        slope = _fit_slope(x, y)
        if abs(q - 1.0) < 1e-6:
            D_q[qi] = slope  # for q=1, slope of entropy vs log(1/eps) = D_1
        else:
            D_q[qi] = slope / (q - 1.0)

    return q_values, D_q


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
