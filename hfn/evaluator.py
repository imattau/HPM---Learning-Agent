"""
HPM Evaluator — pure evaluation service for HFN node populations.

The Evaluator is stateless: no Forest mutation, no decisions, no side effects.
It takes nodes and Observer state as inputs and returns numeric results.

Three responsibility classes:
  1. Fractal geometry: crowding, density_ratio, nearest_prior_dist,
                       hausdorff_candidates, persistence_scores
  2. Gap detection: coverage_gap, underrepresented_regions
  3. HPM framework: accuracy, description_length, score, coherence,
                    curiosity, boredom, reinforcement_signal

HPM Framework alignment
-----------------------
The Evaluator is the HPM layer 3 component (Pattern evaluators/gatekeepers):
deciding which patterns are worth keeping, signalling novelty/redundancy,
and providing hooks for external reinforcement.

Gap detection methods (coverage_gap, underrepresented_regions) provide
the Observer with meta-awareness of its knowledge boundaries — the
precondition for curiosity-driven learning and later integration with
external substrates/fields.

reinforcement_signal() is designed to be subclassed or injected to connect
external reward models, social context evaluators, or field signals without
modifying Observer or Evaluator internals.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from hfn.hfn import HFN


class Evaluator:
    """
    Pure evaluation service for HFN node populations.

    Stateless — no Forest mutation, no decisions.  All methods accept
    explicit arguments; no instance state is stored or mutated.
    """

    # ------------------------------------------------------------------
    # 1. Fractal geometry evaluations
    # ------------------------------------------------------------------

    def crowding(
        self,
        node_mu: np.ndarray,
        nodes: Sequence[HFN],
        radius: float,
    ) -> int:
        """
        Count non-protected nodes within *radius* of *node_mu*.

        Used by multifractal-guided absorption: high crowding = hot spot.
        Protected nodes are passed separately and excluded by the caller
        via the *nodes* argument — this method counts everything in *nodes*.

        Parameters
        ----------
        node_mu : (D,) array
            Centre point for the neighbourhood query.
        nodes : sequence of HFN
            Candidate nodes to count (caller should exclude protected ones).
        radius : float
            Euclidean neighbourhood radius in μ-space.

        Returns
        -------
        int
            Number of nodes within radius of node_mu.
        """
        count = 0
        node_mu = np.asarray(node_mu, dtype=float)
        for n in nodes:
            if float(np.linalg.norm(np.asarray(n.mu, dtype=float) - node_mu)) < radius:
                count += 1
        return count

    def density_ratio(
        self,
        x: np.ndarray,
        nodes: Sequence[HFN],
        radius: float,
    ) -> float:
        """
        Ratio of local node density at *x* to the mean density across all nodes.

        Used by lacunarity-guided creation suppression: if ratio >
        lacunarity_creation_factor the region is already dense and a new node
        would be redundant.

        Returns 0.0 if fewer than 3 nodes exist (always allow creation).

        Parameters
        ----------
        x : (D,) array
            Query point.
        nodes : sequence of HFN
            Active node population.
        radius : float
            Neighbourhood radius for local density estimate.

        Returns
        -------
        float
            Local density / mean density ratio.
        """
        node_list = list(nodes)
        if len(node_list) < 3:
            return 0.0
        x = np.asarray(x, dtype=float)
        mus = np.array([n.mu for n in node_list], dtype=float)
        dists_to_x = np.linalg.norm(mus - x, axis=1)
        local_count = float(np.sum(dists_to_x < radius))
        # Mean nearest-neighbour distance as a proxy for global density
        mean_nn = float(np.mean([
            np.sort(np.linalg.norm(mus - mus[i], axis=1))[1]
            for i in range(len(mus))
        ]))
        expected_local = radius / (mean_nn + 1e-9)
        return local_count / (expected_local + 1e-9)

    def nearest_prior_dist(
        self,
        mu: np.ndarray,
        prior_mus: np.ndarray,
    ) -> float:
        """
        Euclidean distance from *mu* to the nearest prior node in μ-space.

        Used to rank compression candidates (nearest_prior strategy).

        Parameters
        ----------
        mu : (D,) array
            Query μ vector.
        prior_mus : (K, D) array
            Stacked μ vectors of prior (protected) nodes.

        Returns
        -------
        float
            Minimum distance, or inf if prior_mus is empty.
        """
        prior_mus = np.asarray(prior_mus, dtype=float)
        if prior_mus.ndim == 1:
            prior_mus = prior_mus.reshape(1, -1)
        if len(prior_mus) == 0:
            return float("inf")
        mu = np.asarray(mu, dtype=float)
        return float(np.min(np.linalg.norm(prior_mus - mu, axis=1)))

    def hausdorff_candidates(
        self,
        nodes: Sequence[HFN],
        weights: dict[str, float],
        threshold: float,
        weight_floor: float,
        protected_ids: set[str],
    ) -> list[tuple[HFN, HFN]]:
        """
        Return (node, best_match) pairs where *node* is close to a
        better-weighted node and its weight is below *weight_floor*.

        Observer decides whether to absorb each pair; this method only
        identifies the geometric candidates.

        Parameters
        ----------
        nodes : sequence of HFN
            Active node snapshot (caller should pass list(forest.active_nodes())).
        weights : dict
            Node id → weight mapping.
        threshold : float
            Maximum μ-space distance to qualify as a match.
        weight_floor : float
            Only nodes with weight < weight_floor are considered as candidates.
        protected_ids : set of str
            Node ids exempt from all dynamics.

        Returns
        -------
        list of (node, best_match) tuples
            Each node is a weak node close to the stronger best_match.
        """
        node_list = list(nodes)
        pairs: list[tuple[HFN, HFN]] = []
        for node in node_list:
            if node.id in protected_ids:
                continue
            if weights.get(node.id, 0.0) >= weight_floor:
                continue
            for other in node_list:
                if other.id == node.id or other.id in protected_ids:
                    continue
                if weights.get(other.id, 0.0) <= weights.get(node.id, 0.0):
                    continue
                dist = float(np.linalg.norm(
                    np.asarray(node.mu, dtype=float) - np.asarray(other.mu, dtype=float)
                ))
                if dist < threshold:
                    pairs.append((node, other))
                    break  # one best_match per weak node
        return pairs

    def persistence_scores(
        self,
        nodes: Sequence[HFN],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Thin wrapper over hfn.fractal.persistence_scores.

        Returns persistence score per node id.

        Parameters
        ----------
        nodes : sequence of HFN
            Active node population.
        weights : dict
            Node id → weight mapping.

        Returns
        -------
        dict mapping node id → persistence score (float).
        """
        from hfn.fractal import persistence_scores as _ps
        return _ps(list(nodes), weights)

    # ------------------------------------------------------------------
    # 2. Gap / unknown detection
    # ------------------------------------------------------------------

    def coverage_gap(
        self,
        x: np.ndarray,
        nodes: Sequence[HFN],
        radius: float,
    ) -> float:
        """
        How sparse is the region around observation *x* given current nodes.

        High value = Observer has little knowledge near this point.
        Used by Observer to raise readiness to create a new node.

        Defined as 1 / (1 + local_count) where local_count is the number
        of nodes within *radius* of *x*.  Returns 1.0 when no nodes are
        present (maximum gap).

        Parameters
        ----------
        x : (D,) array
            Observation point.
        nodes : sequence of HFN
            Active node population.
        radius : float
            Neighbourhood radius for local density estimate.

        Returns
        -------
        float in [0, 1]
            Gap signal; 1.0 = completely uncovered, approaches 0 as density → ∞.
        """
        x = np.asarray(x, dtype=float)
        local_count = 0
        for n in nodes:
            if float(np.linalg.norm(np.asarray(n.mu, dtype=float) - x)) < radius:
                local_count += 1
        return 1.0 / (1.0 + local_count)

    def underrepresented_regions(
        self,
        nodes: Sequence[HFN],
    ) -> list[np.ndarray]:
        """
        Regions of observation space with low node density.

        Derived from lacunarity analysis across the full node set.  Returns
        representative μ-vectors for each gap region.  Observer can expose
        these to external substrates/fields as "I need more observations here."

        Uses lacunarity at the coarsest scale to identify sparse boxes, then
        returns the centre of each empty/sparse box as a gap representative.

        Returns an empty list if fewer than 3 nodes are present.

        Parameters
        ----------
        nodes : sequence of HFN
            Active node population.

        Returns
        -------
        list of (D,) arrays
            Representative μ-vectors for underrepresented regions.
        """
        from hfn.fractal import lacunarity as _lac
        node_list = list(nodes)
        if len(node_list) < 3:
            return []
        mus = np.array([n.mu for n in node_list], dtype=float)
        # Lacunarity values at multiple scales — use coarsest scale
        lac_values = _lac(node_list, n_scales=6)
        if lac_values is None or len(lac_values) == 0:
            return []
        # Find the scale with maximum lacunarity (most gap structure)
        max_lac_idx = int(np.argmax(lac_values))
        # Derive box size for that scale
        ranges = mus.max(axis=0) - mus.min(axis=0)
        max_range = float(ranges.max())
        if max_range == 0.0:
            return []
        eps_max = max_range * 0.5
        eps_min = max(max_range / (len(mus) * 10), 1e-9)
        if eps_min >= eps_max:
            eps_min = eps_max / 100
        epsilons = np.geomspace(eps_max, eps_min, 6)
        eps = float(epsilons[min(max_lac_idx, len(epsilons) - 1)])
        # Find occupied boxes
        boxes_idx = np.floor(mus / eps).astype(np.int32)
        occupied = set(map(tuple, boxes_idx))
        # Generate candidate gap boxes (grid around occupied region)
        min_box = boxes_idx.min(axis=0)
        max_box = boxes_idx.max(axis=0)
        gap_centres: list[np.ndarray] = []
        # Iterate over a coarse grid — limit to 1000 candidates
        grid_pts = np.array(np.meshgrid(
            *[np.arange(min_box[d], max_box[d] + 1) for d in range(mus.shape[1])]
        )).reshape(mus.shape[1], -1).T
        for pt in grid_pts[:1000]:
            if tuple(pt) not in occupied:
                gap_centres.append((pt + 0.5) * eps)
        return gap_centres

    # ------------------------------------------------------------------
    # 3. HPM framework evaluations
    # ------------------------------------------------------------------

    def accuracy(self, x: np.ndarray, node: HFN) -> float:
        """
        Functional utility of *node* for observation *x*.

        Normalised log-probability: higher = better fit.

        Parameters
        ----------
        x : (D,) array
            Observation point.
        node : HFN
            Node to evaluate.

        Returns
        -------
        float in (0, 1]
        """
        lp = node.log_prob(x)
        return float(1.0 / (1.0 + abs(lp)))

    def description_length(self, node: HFN) -> float:
        """
        Complexity proxy for *node*.

        Thin wrapper over node.description_length().  Used in score computation.

        Parameters
        ----------
        node : HFN

        Returns
        -------
        float
        """
        return float(node.description_length())

    def score(self, x: np.ndarray, node: HFN, lambda_complexity: float) -> float:
        """
        Combined utility evaluation.

        score = accuracy(x, node) - lambda_complexity * description_length(node)

        Parameters
        ----------
        x : (D,) array
            Observation point.
        node : HFN
            Node to evaluate.
        lambda_complexity : float
            Regularisation weight on description length.

        Returns
        -------
        float
        """
        return self.accuracy(x, node) - lambda_complexity * self.description_length(node)

    def coherence(self, node: HFN) -> float:
        """
        Internal consistency of *node*'s pattern.

        Derived from sigma matrix properties: condition number of sigma
        as a proxy for eigenvalue spread — low spread = high coherence.

        Returns a value in [0, 1] where 1 = perfectly coherent (identity sigma)
        and approaches 0 for highly ill-conditioned covariance.

        Parameters
        ----------
        node : HFN

        Returns
        -------
        float in [0, 1]
        """
        sigma = np.asarray(node.sigma, dtype=float)
        try:
            cond = float(np.linalg.cond(sigma))
        except np.linalg.LinAlgError:
            return 0.0
        if not np.isfinite(cond) or cond <= 0.0:
            return 0.0
        # Map condition number to [0, 1]: 1 = cond 1, approaches 0 as cond → ∞
        return float(1.0 / cond)

    def curiosity(
        self,
        x: np.ndarray,
        nodes: Sequence[HFN],
        weights: dict[str, float],
    ) -> float:
        """
        Novelty signal for observation *x*.

        How far is *x* from well-weighted nodes?  High value = Observer has
        not seen patterns like this.

        Complements coverage_gap (geometric) with a weight-aware novelty measure.
        Defined as the minimum weighted distance to any node:

            min_dist / (weight_of_nearest + ε)

        Normalised to [0, 1] via sigmoid.

        Parameters
        ----------
        x : (D,) array
            Observation point.
        nodes : sequence of HFN
            Active node population.
        weights : dict
            Node id → weight mapping.

        Returns
        -------
        float in [0, 1]
        """
        x = np.asarray(x, dtype=float)
        node_list = list(nodes)
        if not node_list:
            return 1.0
        min_weighted_dist = float("inf")
        for n in node_list:
            dist = float(np.linalg.norm(np.asarray(n.mu, dtype=float) - x))
            w = weights.get(n.id, 0.0) + 1e-9
            weighted = dist / w
            if weighted < min_weighted_dist:
                min_weighted_dist = weighted
        # Sigmoid normalisation: maps (0, ∞) → (0, 1)
        return float(1.0 - 1.0 / (1.0 + min_weighted_dist))

    def boredom(
        self,
        node: HFN,
        weights: dict[str, float],
        scores: dict[str, float],
    ) -> float:
        """
        Redundancy signal for *node*.

        High weight but low score = this node is over-represented; the
        Observer is not learning from it anymore.

        Defined as: max(0, weight - |score|) / (weight + ε)

        A node with high weight and near-zero score returns a value near 1
        (very bored).  A node with high weight and high positive score
        returns near 0 (still useful).

        Parameters
        ----------
        node : HFN
        weights : dict
            Node id → weight mapping.
        scores : dict
            Node id → score mapping.

        Returns
        -------
        float in [0, 1]
        """
        w = weights.get(node.id, 0.0)
        s = abs(scores.get(node.id, 0.0))
        return float(max(0.0, w - s) / (w + 1e-9))

    def reinforcement_signal(self, node_id: str) -> float:
        """
        Hook for external reward / reinforcement input.

        Default implementation returns 0.0 (no external signal).
        Subclass Evaluator or inject a custom instance to connect
        external evaluators (reward models, social context, field signals).
        Observer uses this in weight update decisions.

        Parameters
        ----------
        node_id : str
            Id of the node being evaluated.

        Returns
        -------
        float
            Reinforcement signal in any range (default 0.0).
        """
        return 0.0
