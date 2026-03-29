"""
HPM Observer — watches the Forest and drives all learning dynamics.

The Observer is stateful across queries. It holds:
- weights: accumulated predictive success per node
- scores: accuracy minus complexity per node
- co-occurrence: how often node pairs appear together in explanation trees
- miss counts: how many consecutive queries a node lost to an overlapping node

It runs the expansion loop (querying the Forest) and handles:
- Weight updates after each query
- Structural absorption when persistent overlap is detected
- Node creation from residual surprise or recurring co-occurrence

## HPM Framework boundaries

After the Evaluator/Recombination refactor:

  1. Surprise computation + expansion  — pure perception mechanics (Observer)
  2. Weight / score dynamics           — pattern dynamics, HPM layer 2 (Observer)
  3. Structural evaluation             — evaluator/gatekeeper, HPM layer 3 (Evaluator)
  4. Structural execution              — absorption + compression (Recombination)

Observer.evaluator holds all fractal geometry, gap detection, and HPM
framework evaluations.  Observer.recombination executes all structural merges.
Observer retains all HPM strategy flags — they are Observer configuration
for when to invoke which Evaluator method.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.evaluator import Evaluator
from hfn.recombination import Recombination


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class ExplanationResult(NamedTuple):
    explanation_tree: list[HFN]
    accuracy_scores: dict[str, float]
    residual_surprise: float


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class Observer:
    """
    The Observer watches a Forest (world model) and drives its dynamics.

    Separation of concerns:
    - Forest (HFN): pure structural container. No mutable dynamic state.
    - Observer: all dynamic state — weights, scores, co-occurrence, absorption.
    - Evaluator: pure evaluation service — stateless queries, no Forest mutation.
    - Recombination: structural executor — absorb/compress operations.

    The Observer can watch any HFN, including a Forest-of-Forests, using
    the same interface. Fractal uniformity applies to observation too.
    """

    def __init__(
        self,
        forest: Forest,
        tau: float = 1.0,                          # surprise threshold for expansion
        budget: int = 10,                           # max expansions per query
        lambda_complexity: float = 0.1,             # S = accuracy - λ·complexity
        alpha_gain: float = 0.1,                    # weight gain for explaining nodes
        beta_loss: float = 0.05,                    # weight loss for overlapping non-explaining nodes
        absorption_overlap_threshold: float = 0.5,  # κ threshold for absorption
        absorption_miss_threshold: int = 5,         # consecutive misses before absorption
        residual_surprise_threshold: float = 2.0,   # threshold to spawn new leaf
        compression_cooccurrence_threshold: int = 3, # M: co-occurrences to trigger compression
        w_init: float = 0.1,
        protected_ids: set[str] | None = None,      # nodes exempt from all dynamics
        recombination_strategy: str = "cooccurrence",  # "cooccurrence" | "nearest_prior"
        hausdorff_absorption_threshold: float | None = None,  # geometric absorption distance
        hausdorff_absorption_weight_floor: float = 0.2,       # min weight to trigger geometric absorption
        persistence_guided_absorption: bool = False,  # scale miss threshold by node persistence
        adaptive_compression: bool = False,           # adjust compression threshold by recurrence rate
        recurrence_buffer_size: int = 50,             # RecurrenceTracker buffer size
        recurrence_epsilon: float = 0.3,              # RecurrenceTracker neighbourhood radius
        lacunarity_guided_creation: bool = False,     # suppress node creation in already-dense regions
        lacunarity_creation_radius: float = 0.08,     # neighbourhood radius for local density check
        lacunarity_creation_factor: float = 2.0,      # suppress if local density > factor * mean density
        multifractal_guided_absorption: bool = False, # lower absorption threshold in hot spots
        multifractal_crowding_radius: float = 0.12,   # radius for local crowding count
        multifractal_crowding_threshold: int = 3,     # nodes within radius = hot spot → absorb sooner
        evaluator: Evaluator | None = None,           # injectable evaluator (default: Evaluator())
        recombination: Recombination | None = None,   # injectable recombination (default: Recombination())
    ):
        self.forest = forest
        self.tau = tau
        self.budget = budget
        self.lambda_complexity = lambda_complexity
        self.alpha_gain = alpha_gain
        self.beta_loss = beta_loss
        self.absorption_overlap_threshold = absorption_overlap_threshold
        self.absorption_miss_threshold = absorption_miss_threshold
        self.residual_surprise_threshold = residual_surprise_threshold
        self.compression_cooccurrence_threshold = compression_cooccurrence_threshold
        self.w_init = w_init
        self.recombination_strategy = recombination_strategy
        self.hausdorff_absorption_threshold = hausdorff_absorption_threshold
        self.hausdorff_absorption_weight_floor = hausdorff_absorption_weight_floor
        self.persistence_guided_absorption = persistence_guided_absorption
        self.adaptive_compression = adaptive_compression
        self.lacunarity_guided_creation = lacunarity_guided_creation
        self.lacunarity_creation_radius = lacunarity_creation_radius
        self.lacunarity_creation_factor = lacunarity_creation_factor
        self.multifractal_guided_absorption = multifractal_guided_absorption
        self.multifractal_crowding_radius = multifractal_crowding_radius
        self.multifractal_crowding_threshold = multifractal_crowding_threshold

        # Collaborators (HPM layer 3 + structural executor)
        self.evaluator: Evaluator = evaluator if evaluator is not None else Evaluator()
        self.recombination: Recombination = recombination if recombination is not None else Recombination()

        if adaptive_compression:
            from hfn.fractal import RecurrenceTracker
            self._recurrence: "RecurrenceTracker | None" = RecurrenceTracker(
                recurrence_buffer_size, recurrence_epsilon
            )
        else:
            self._recurrence = None

        self._weights: dict[str, float] = {}
        self._scores: dict[str, float] = {}
        self._miss_counts: dict[str, int] = defaultdict(int)
        self._cooccurrence: dict[frozenset, int] = defaultdict(int)

        # Initialise weights for any nodes already in the forest
        for node in forest.active_nodes():
            self._init_node(node)

        # Track absorbed node ids for test visibility
        self.absorbed_ids: set[str] = set()

        # Protected nodes are exempt from all dynamics (weight change, absorption).
        # Use for priors that represent invariant structural knowledge.
        self.protected_ids: set[str] = set(protected_ids) if protected_ids else set()

    # --- Node lifecycle ---

    def register(self, node: HFN) -> None:
        """Register a node in the Forest and initialise its Observer state."""
        self.forest.register(node)
        self._init_node(node)

    def _init_node(self, node: HFN) -> None:
        if node.id not in self._weights:
            self._weights[node.id] = self.w_init
            self._scores[node.id] = 0.0

    # --- Weight / score access ---

    def get_weight(self, node_id: str) -> float:
        return self._weights.get(node_id, 0.0)

    def get_score(self, node_id: str) -> float:
        return self._scores.get(node_id, 0.0)

    # --- Observation loop ---

    def observe(self, x: np.ndarray) -> ExplanationResult:
        """
        Run one observation pass for signal x:
        1. Retrieve candidate nodes from the Forest
        2. Expand nodes greedily by surprise + utility
        3. Update weights, scores, co-occurrence
        4. Trigger absorption and node creation if warranted
        Returns the explanation result.
        """
        result = self._expand(x)
        if self._recurrence is not None:
            self._recurrence.update(x)
        self._update_weights(x, result)
        self._update_scores(result)
        self._track_cooccurrence(result.explanation_tree)
        self._check_absorption()
        self._check_node_creation(x, result)
        return result

    def _expand(self, x: np.ndarray) -> ExplanationResult:
        """
        Run the expansion loop and return the explanation tree.

        A node is a GOOD EXPLAINER (added to explanation_tree) when its
        surprise is BELOW tau — it adequately explains x.

        A node is SURPRISING (above tau) — it needs expanding into its
        children to find a better explanation. If it's a leaf, it can't
        be expanded and contributes to residual surprise.
        """
        frontier = list(self.forest.retrieve(x, k=5))
        explanation_tree: list[HFN] = []
        accuracy_scores: dict[str, float] = {}
        seen_ids: set[str] = set()
        surprising_leaves: list[HFN] = []
        budget = self.budget

        while frontier:
            good     = [n for n in frontier if self._kl_surprise(x, n) < self.tau]
            surprising = [n for n in frontier if self._kl_surprise(x, n) >= self.tau]

            # Good explainers go into the explanation tree
            for n in good:
                if n.id not in seen_ids:
                    explanation_tree.append(n)
                    accuracy_scores[n.id] = self.evaluator.accuracy(x, n)
                    seen_ids.add(n.id)

            # No surprising nodes left — fully explained
            if not surprising or budget == 0:
                break

            # Expand the most surprising node (highest potential utility)
            best = max(surprising, key=lambda n: self._kl_surprise(x, n))
            frontier = [n for n in frontier if n.id != best.id]
            children = best.children()
            if children:
                frontier.extend(c for c in children if c.id not in seen_ids)
                budget -= 1
            else:
                # Leaf with high surprise — can't expand, becomes residual
                seen_ids.add(best.id)
                surprising_leaves.append(best)

        # Residual: mean surprise of surprising leaf nodes that couldn't be expanded
        # plus any remaining frontier nodes still above tau
        residual_nodes = surprising_leaves + [
            n for n in frontier if n.id not in seen_ids
            and self._kl_surprise(x, n) >= self.tau
        ]
        residual = float(np.mean([self._kl_surprise(x, n) for n in residual_nodes])) \
            if residual_nodes else 0.0

        return ExplanationResult(
            explanation_tree=explanation_tree,
            accuracy_scores=accuracy_scores,
            residual_surprise=residual,
        )

    def _kl_surprise(self, x: np.ndarray, node: HFN) -> float:
        """KL divergence proxy: negative log probability under node's Gaussian."""
        return -node.log_prob(x)

    # --- Dynamic updates ---

    def _update_weights(self, x: np.ndarray, result: ExplanationResult) -> None:
        explaining_ids = {n.id for n in result.explanation_tree}

        for node in self.forest.active_nodes():
            nid = node.id
            if nid not in self._weights:
                self._init_node(node)
            if nid in self.protected_ids:
                continue

            if nid in explaining_ids:
                acc = result.accuracy_scores.get(nid, 0.0)
                self._weights[nid] += self.alpha_gain * (1.0 - self._weights[nid]) * acc
                self._weights[nid] = min(self._weights[nid], 1.0)
                self._miss_counts[nid] = 0
            else:
                for explaining_node in result.explanation_tree:
                    kappa = node.overlap(explaining_node)
                    self._weights[nid] -= self.beta_loss * kappa * self._weights[nid]
                self._weights[nid] = max(self._weights[nid], 0.0)
                if any(
                    node.overlap(n) > self.absorption_overlap_threshold
                    for n in result.explanation_tree
                ):
                    self._miss_counts[nid] += 1

    def _update_scores(self, result: ExplanationResult) -> None:
        explaining_ids = {n.id for n in result.explanation_tree}
        for node in self.forest.active_nodes():
            nid = node.id
            if nid in explaining_ids:
                # Use the accuracy from the explanation tree for explaining nodes
                x_proxy = None  # score formula via evaluator.score requires x
                # Delegate to evaluator.score using stored accuracy
                acc = result.accuracy_scores.get(nid, 0.0)
                self._scores[nid] = acc - self.lambda_complexity * self.evaluator.description_length(node)
            else:
                self._scores[nid] = 0.0 - self.lambda_complexity * self.evaluator.description_length(node)

    def _track_cooccurrence(self, explanation_tree: list[HFN]) -> None:
        ids = [n.id for n in explanation_tree]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self._cooccurrence[frozenset([ids[i], ids[j]])] += 1

    # --- Absorption ---

    def _build_prior_mus_matrix(self) -> np.ndarray | None:
        """Stack μ vectors of all protected (prior) nodes into a (K, D) matrix."""
        prior_nodes = [
            n for n in self.forest.active_nodes() if n.id in self.protected_ids
        ]
        if not prior_nodes:
            return None
        return np.array([n.mu for n in prior_nodes], dtype=float)

    def _check_absorption(self) -> None:
        # Take a snapshot to prevent mid-pass invalidation when nodes are deregistered
        snapshot = list(self.forest.active_nodes())

        # Compute persistence scores once per absorption pass (not per node)
        p_scores: dict[str, float] = {}
        p_max: float = 1.0
        if self.persistence_guided_absorption:
            p_scores = self.evaluator.persistence_scores(snapshot, self._weights)
            p_max = max(
                (v for v in p_scores.values() if v != float("inf")),
                default=1.0,
            )

        # Hausdorff geometric absorption — use evaluator.hausdorff_candidates
        if self.hausdorff_absorption_threshold is not None:
            candidates = self.evaluator.hausdorff_candidates(
                snapshot,
                self._weights,
                threshold=self.hausdorff_absorption_threshold,
                weight_floor=self.hausdorff_absorption_weight_floor,
                protected_ids=self.protected_ids,
            )
            for node, best_match in candidates:
                # Only absorb if both are still active (snapshot may be stale after earlier absorbs)
                if node.id not in self.forest or best_match.id not in self.forest:
                    continue
                new_node = self.recombination.absorb(
                    absorbed=node, dominant=best_match, forest=self.forest
                )
                self._init_node(new_node)
                self._weights[new_node.id] = self._weights.get(best_match.id, self.w_init)
                self._scores[new_node.id] = self._scores.get(best_match.id, 0.0)
                self.absorbed_ids.add(node.id)

        for node in snapshot:
            if node.id in self.protected_ids:
                continue
            # Skip if already absorbed in this pass
            if node.id not in self.forest:
                continue

            # Original miss-count absorption
            effective_miss_threshold = self.absorption_miss_threshold
            if self.persistence_guided_absorption and p_max > 0:
                node_p = p_scores.get(node.id, 0.0)
                norm_p = min(node_p, p_max) / p_max  # in [0, 1]
                effective_miss_threshold = int(
                    round(self.absorption_miss_threshold * (1.0 + norm_p))
                )

            # Multifractal crowding: lower threshold for nodes in hot spots
            if self.multifractal_guided_absorption:
                # Exclude protected nodes from crowding count
                non_protected = [n for n in snapshot if n.id not in self.protected_ids]
                crowding = self.evaluator.crowding(
                    node.mu, non_protected, self.multifractal_crowding_radius
                )
                if crowding >= self.multifractal_crowding_threshold:
                    # Hot spot — halve the threshold (absorb sooner)
                    effective_miss_threshold = max(1, effective_miss_threshold // 2)

            if self._miss_counts.get(node.id, 0) < effective_miss_threshold:
                continue

            best_overlap = 0.0
            best_node = None
            for other in snapshot:
                if other.id == node.id:
                    continue
                if other.id in self.protected_ids:
                    continue
                if other.id not in self.forest:
                    continue
                kappa = node.overlap(other)
                if kappa > self.absorption_overlap_threshold and kappa > best_overlap:
                    best_overlap = kappa
                    best_node = other

            if best_node is None:
                continue

            new_node = self.recombination.absorb(
                absorbed=node, dominant=best_node, forest=self.forest
            )
            self._init_node(new_node)
            self._weights[new_node.id] = self._weights.get(best_node.id, self.w_init)
            self._scores[new_node.id] = self._scores.get(best_node.id, 0.0)
            self.absorbed_ids.add(node.id)

    # --- Node creation ---

    def _check_node_creation(self, x: np.ndarray, result: ExplanationResult) -> None:
        self._check_residual_surprise(x, result)
        self._check_compression_candidates()

    def _check_residual_surprise(self, x: np.ndarray, result: ExplanationResult) -> None:
        # Bootstrap: empty forest always needs a first node
        if len(self.forest) == 0 or result.residual_surprise >= self.residual_surprise_threshold:
            # Lacunarity-guided suppression: skip creation in already-dense regions
            if self.lacunarity_guided_creation and len(self.forest) > 0:
                ratio = self.evaluator.density_ratio(
                    x,
                    list(self.forest.active_nodes()),
                    self.lacunarity_creation_radius,
                )
                if ratio > self.lacunarity_creation_factor:
                    return  # region is dense — redirect to compression, not creation
            D = x.shape[0]
            new_node = HFN(
                mu=x.copy(),
                sigma=np.eye(D),
                id=f"leaf_{len(self.forest)}",
            )
            self.register(new_node)

    def _check_compression_candidates(self) -> None:
        threshold = self.compression_cooccurrence_threshold
        if self._recurrence is not None:
            threshold = self._recurrence.recommended_threshold(threshold)
        # Collect all qualifying pairs
        candidates: list[tuple[frozenset, list[str]]] = []
        for pair, count in list(self._cooccurrence.items()):
            if count < threshold:
                continue
            ids = list(pair)
            if ids[0] not in self.forest or ids[1] not in self.forest:
                continue
            compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
            if compressed_id in self.forest:
                continue
            candidates.append((pair, ids))

        if not candidates:
            return

        # Rank by nearest-prior proximity when strategy is set
        if self.recombination_strategy == "nearest_prior" and self.protected_ids:
            prior_mus = self._build_prior_mus_matrix()
            if prior_mus is not None:
                def _rank(item: tuple) -> float:
                    _, ids = item
                    a_mu = self.forest.get(ids[0]).mu
                    b_mu = self.forest.get(ids[1]).mu
                    midpoint = (a_mu + b_mu) / 2.0
                    return self.evaluator.nearest_prior_dist(midpoint, prior_mus)
                candidates.sort(key=_rank)

        # Process all candidates in ranked order (nearest_prior sorts first,
        # cooccurrence preserves dict iteration order).
        for pair, ids in candidates:
            node_a = self.forest.get(ids[0])
            node_b = self.forest.get(ids[1])
            compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
            # Re-check compressed_id is not already active (may have been added in earlier iteration)
            if compressed_id in self.forest:
                continue
            new_node = self.recombination.compress(
                node_a, node_b, self.forest, compressed_id
            )
            self._init_node(new_node)
            self._cooccurrence[pair] = 0
