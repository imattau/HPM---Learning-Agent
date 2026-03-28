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

## HPM Framework boundary note

The Observer currently combines three distinct responsibilities:

  1. Surprise computation + expansion  — pure perception mechanics
  2. Weight / score dynamics           — pattern dynamics (HPM layer 2)
  3. Absorption + compression          — structural evaluation (HPM layer 3)

Items 1 and 2 are unambiguously the Observer's job. Item 3 is evaluator
territory — deciding *which patterns are worth keeping* relative to prior
knowledge. The fractal strategies (recombination_strategy, hausdorff_absorption_threshold)
expose this: they require the Observer to know about the prior structure to
make structural decisions.

For the hfn library as a standalone tool this is acceptable — the strategies
are opt-in and the Observer remains domain-agnostic.

When integrating hfn into a full HPM AI, the structural decisions (absorption,
compression) should move up to the HPM AI's evaluator layer, which has access
to both Observer state AND fractal diagnostics AND domain knowledge. The
intended interface at that point:

    HPM AI Evaluator
        ↓ uses fractal diagnostics to decide
    Observer  ← exposes candidates, executes decisions on request
        ↓
    Forest

The Observer would expose candidate lists (nodes eligible for absorption, pairs
eligible for compression) and accept explicit directives — obs.absorb(node),
obs.compress(A, B) — rather than making those decisions itself. The evaluator
layer would call the fractal tools to inform its choices.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

from hfn.hfn import HFN
from hfn.forest import Forest


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

        # Cache of prior node mu-vectors for fast nearest-prior lookup.
        # Rebuilt whenever protected_ids is used in recombination_strategy.
        self._prior_mus: np.ndarray | None = None

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
                    accuracy_scores[n.id] = self._accuracy(x, n)
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

    def _accuracy(self, x: np.ndarray, node: HFN) -> float:
        """Accuracy: normalised log probability (higher = better fit)."""
        lp = node.log_prob(x)
        return float(1.0 / (1.0 + abs(lp)))

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
            acc = result.accuracy_scores.get(nid, 0.0) if nid in explaining_ids else 0.0
            self._scores[nid] = acc - self.lambda_complexity * node.description_length()

    def _track_cooccurrence(self, explanation_tree: list[HFN]) -> None:
        ids = [n.id for n in explanation_tree]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self._cooccurrence[frozenset([ids[i], ids[j]])] += 1

    # --- Absorption ---

    def _check_absorption(self) -> None:
        for node in list(self.forest.active_nodes()):
            if node.id in self.protected_ids:
                continue

            absorbed = False

            # Geometric shortcut: absorb if node is close to a better-weighted
            # non-protected node and its own weight is below the floor.
            if (
                self.hausdorff_absorption_threshold is not None
                and self._weights.get(node.id, 0.0) < self.hausdorff_absorption_weight_floor
            ):
                for other in self.forest.active_nodes():
                    if other.id == node.id or other.id in self.protected_ids:
                        continue
                    if self._weights.get(other.id, 0.0) <= self._weights.get(node.id, 0.0):
                        continue
                    dist = float(np.linalg.norm(node.mu - other.mu))
                    if dist < self.hausdorff_absorption_threshold:
                        new_parent = other.recombine(node)
                        self.forest.deregister(other.id)
                        self.forest.deregister(node.id)
                        self.forest.register(new_parent)
                        self._init_node(new_parent)
                        self._weights[new_parent.id] = self._weights.get(other.id, self.w_init)
                        self._scores[new_parent.id] = self._scores.get(other.id, 0.0)
                        self.absorbed_ids.add(node.id)
                        absorbed = True
                        break

            if absorbed:
                continue

            # Original miss-count absorption
            effective_miss_threshold = self.absorption_miss_threshold
            if self.persistence_guided_absorption:
                from hfn.fractal import persistence_scores
                p_scores = persistence_scores(
                    list(self.forest.active_nodes()), self._weights
                )
                max_p = max(
                    (v for v in p_scores.values() if v != float("inf")),
                    default=1.0,
                )
                if max_p > 0:
                    node_p = p_scores.get(node.id, 0.0)
                    norm_p = min(node_p, max_p) / max_p  # in [0, 1]
                    effective_miss_threshold = int(
                        round(self.absorption_miss_threshold * (1.0 + norm_p))
                    )

            # Multifractal crowding: lower threshold for nodes in hot spots
            if self.multifractal_guided_absorption:
                crowding = self._local_crowding(node.mu)
                if crowding >= self.multifractal_crowding_threshold:
                    # Hot spot — halve the threshold (absorb sooner)
                    effective_miss_threshold = max(1, effective_miss_threshold // 2)

            if self._miss_counts.get(node.id, 0) < effective_miss_threshold:
                continue

            best_overlap = 0.0
            best_node = None
            for other in self.forest.active_nodes():
                if other.id == node.id:
                    continue
                if other.id in self.protected_ids:
                    continue
                kappa = node.overlap(other)
                if kappa > self.absorption_overlap_threshold and kappa > best_overlap:
                    best_overlap = kappa
                    best_node = other

            if best_node is None:
                continue

            new_parent = best_node.recombine(node)
            self.forest.deregister(best_node.id)
            self.forest.deregister(node.id)
            self.forest.register(new_parent)
            self._init_node(new_parent)
            self._weights[new_parent.id] = self._weights.get(best_node.id, self.w_init)
            self._scores[new_parent.id] = self._scores.get(best_node.id, 0.0)
            self.absorbed_ids.add(node.id)

    # --- Fractal geometry helpers ---

    def _build_prior_mus(self) -> np.ndarray | None:
        """Stack μ vectors of all protected (prior) nodes into a matrix."""
        prior_nodes = [
            n for n in self.forest.active_nodes() if n.id in self.protected_ids
        ]
        if not prior_nodes:
            return None
        return np.array([n.mu for n in prior_nodes])

    def _nearest_prior_dist(self, mu: np.ndarray) -> float:
        """Distance from mu to the nearest prior node in μ-space."""
        if self._prior_mus is None:
            self._prior_mus = self._build_prior_mus()
        if self._prior_mus is None:
            return float("inf")
        return float(np.min(np.linalg.norm(self._prior_mus - mu, axis=1)))

    def _local_density_ratio(self, x: np.ndarray) -> float:
        """
        Ratio of local node density at x to the mean density across all nodes.

        Used by lacunarity_guided_creation: if ratio > lacunarity_creation_factor,
        the region is already dense and a new node would be redundant.

        Returns 0.0 if fewer than 3 nodes exist (always allow creation).
        """
        nodes = list(self.forest.active_nodes())
        if len(nodes) < 3:
            return 0.0
        mus = np.array([n.mu for n in nodes])
        dists_to_x = np.linalg.norm(mus - x, axis=1)
        local_count = float(np.sum(dists_to_x < self.lacunarity_creation_radius))
        # Mean nearest-neighbour distance as a proxy for global density
        mean_nn = float(np.mean([
            np.sort(np.linalg.norm(mus - mus[i], axis=1))[1]
            for i in range(len(mus))
        ]))
        expected_local = self.lacunarity_creation_radius / (mean_nn + 1e-9)
        return local_count / (expected_local + 1e-9)

    def _local_crowding(self, node_mu: np.ndarray) -> int:
        """
        Count non-protected nodes within multifractal_crowding_radius of node_mu.

        Used by multifractal_guided_absorption: high crowding = hot spot = absorb sooner.
        """
        count = 0
        for other in self.forest.active_nodes():
            if other.id in self.protected_ids:
                continue
            if float(np.linalg.norm(other.mu - node_mu)) < self.multifractal_crowding_radius:
                count += 1
        return count

    # --- Node creation ---

    def _check_node_creation(self, x: np.ndarray, result: ExplanationResult) -> None:
        self._check_residual_surprise(x, result)
        self._check_compression_candidates()

    def _check_residual_surprise(self, x: np.ndarray, result: ExplanationResult) -> None:
        # Bootstrap: empty forest always needs a first node
        if len(self.forest) == 0 or result.residual_surprise >= self.residual_surprise_threshold:
            # Lacunarity-guided suppression: skip creation in already-dense regions
            if self.lacunarity_guided_creation and len(self.forest) > 0:
                if self._local_density_ratio(x) > self.lacunarity_creation_factor:
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
            self._prior_mus = self._build_prior_mus()
            def _rank(item: tuple) -> float:
                _, ids = item
                a_mu = self.forest._registry[ids[0]].mu
                b_mu = self.forest._registry[ids[1]].mu
                midpoint = (a_mu + b_mu) / 2.0
                return self._nearest_prior_dist(midpoint)
            candidates.sort(key=_rank)

        # Process all candidates in ranked order (nearest_prior sorts first,
        # cooccurrence preserves dict iteration order).
        to_process = candidates

        for pair, ids in to_process:
            node_a = self.forest._registry[ids[0]]
            node_b = self.forest._registry[ids[1]]
            new_node = node_a.recombine(node_b)
            compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
            new_node.id = compressed_id  # type: ignore[misc]
            self.register(new_node)
            self._cooccurrence[pair] = 0
