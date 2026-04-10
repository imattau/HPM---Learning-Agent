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
from pathlib import Path
from typing import NamedTuple

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.evaluator import Evaluator
from hfn.recombination import Recombination
from hfn.policy import (
    DecisionPolicy,
    CreateContext,
    AbsorptionContext,
    StructureArbitrationContext,
    SearchPolicyConfig,
    LearningPolicyConfig,
    StructurePolicyConfig,
    NoveltyPolicyConfig,
)
from hfn.query import Query
from hfn.converter import Converter
from hfn.retriever import Retriever, GeometricRetriever
from hfn.observer_state import ObserverStateStore
from hfn.tiered_forest import TieredForest


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class ExplanationResult(NamedTuple):
    explanation_tree: list[HFN]
    accuracy_scores: dict[str, float]
    residual_surprise: float
    surprising_leaves: list[HFN]


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
        policy: DecisionPolicy | None = None,         # injectable decision policy (default: DecisionPolicy())
        query: Query | None = None,                   # gap query plugin (default: None = disabled)
        converter: Converter | None = None,           # raw-to-vector encoder (default: None = disabled)
        gap_query_threshold: float = 0.7,             # min coverage gap to trigger a query
        max_expand_depth: int = 2,                    # retry budget multiplier for gap queries
        vocab: list | None = None,                    # optional token vocabulary for gap_hash lookup
        prior_plasticity: bool = False,        # enable density-based prior revision
        prior_drift_rate: float = 0.01,        # mu drift step when revision triggered
        prior_revision_threshold: int = 200,   # misses before eligible for drift
        node_use_diag: bool = False,           # use O(D) diagonal sigma for all dynamically created nodes
        node_prefix: str = "leaf_",            # prefix for dynamically created leaf nodes
        weight_decay_rate: float = 0.0,        # global weight decay rate (0.0 = disabled)
        retriever: Retriever = None,           # injectable retriever (default: GeometricRetriever)
        decoder = None,                        # injectable decoder for predictive coding
    ):
        self.forest = forest
        self.retriever = retriever or GeometricRetriever(forest)
        self.decoder = decoder
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

        # Gap query collaborators
        self.query: Query | None = query
        self.converter: Converter | None = converter
        self.gap_query_threshold = gap_query_threshold
        self.max_expand_depth = max_expand_depth
        self.vocab: list | None = vocab
        self._in_gap_query: bool = False

        # Collaborators (HPM layer 3 + structural executor)
        self.evaluator: Evaluator = evaluator if evaluator is not None else Evaluator()
        self.recombination: Recombination = recombination if recombination is not None else Recombination()
        self.policy: DecisionPolicy = policy if policy is not None else DecisionPolicy(
            search=SearchPolicyConfig(tau=tau, budget=budget),
            learning=LearningPolicyConfig(
                alpha_gain=alpha_gain,
                beta_loss=beta_loss,
                lambda_complexity=lambda_complexity,
                weight_decay_rate=weight_decay_rate,
            ),
            structure=StructurePolicyConfig(
                absorption_miss_threshold=absorption_miss_threshold,
                compression_cooccurrence_threshold=compression_cooccurrence_threshold,
            ),
            novelty=NoveltyPolicyConfig(
                residual_surprise_threshold=residual_surprise_threshold,
                gap_query_threshold=gap_query_threshold,
                lacunarity_enabled=lacunarity_guided_creation,
                lacunarity_creation_factor=lacunarity_creation_factor,
            ),
        )

        if adaptive_compression:
            from hfn.fractal import RecurrenceTracker
            self._recurrence: "RecurrenceTracker | None" = RecurrenceTracker(
                recurrence_buffer_size, recurrence_epsilon
            )
        else:
            self._recurrence = None

        # --- Meta-forest: NodeState HFNs ---
        # Each domain node gets a companion 4D HFN: mu=[weight, score, miss_count, hit_count]
        import tempfile
        self._meta_dir = Path(tempfile.mkdtemp(prefix="hfn_meta_"))
        self.meta_forest: TieredForest = TieredForest(
            D=4, cold_dir=self._meta_dir / "cold", forest_id="meta"
        )
        self.state_store = ObserverStateStore(self.meta_forest)

        # Cooccurrence tracking via meta_forest (prefix "cooc:")
        # mu=[count, recency, pair_score, 0] — padded to D=4
        self._observation_step: int = 0

        # Initialise weights for any nodes already in the forest
        for node in forest.active_nodes():
            self._init_node(node)

        # Track absorbed node ids for test visibility
        self.absorbed_ids: set[str] = set()

        # Protected nodes are exempt from all dynamics (weight change, absorption).
        # Use for priors that represent invariant structural knowledge.
        self.protected_ids: set[str] = set(protected_ids) if protected_ids else set()

        # Prior plasticity (graduated protection based on density)
        self.prior_plasticity: bool = prior_plasticity
        self.prior_drift_rate: float = prior_drift_rate
        self.prior_revision_threshold: int = prior_revision_threshold
        self._prior_miss_counts: dict[str, int] = defaultdict(int)
        self._prior_hit_counts: dict[str, int] = defaultdict(int)

        # Diagonal sigma storage for dynamically created nodes
        self.node_use_diag: bool = node_use_diag
        self.node_prefix: str = node_prefix
        self.weight_decay_rate: float = weight_decay_rate

    # --- Node lifecycle ---

    def register(self, node: HFN) -> None:
        """Register a node in the Forest and initialise its Observer state."""
        self.forest.register(node)
        self._init_node(node)

    def _init_node(self, node: HFN) -> None:
        state_id = f"state:{node.id}"
        if state_id not in self.meta_forest:
            state_node = HFN(
                mu=np.array([self.w_init, 0.0, 0.0, 0.0]),
                sigma=np.ones(4),
                id=state_id,
                use_diag=True,
            )
            self.meta_forest.register(state_node)

    # --- NodeState HFN helpers ---
    # mu layout: [0]=weight, [1]=score, [2]=miss_count, [3]=hit_count

    def _get_state(self, node_id: str) -> HFN | None:
        return self.meta_forest.get(f"state:{node_id}")

    def _set_state_field(self, node_id: str, idx: int, value: float) -> None:
        s = self._get_state(node_id)
        if s is not None:
            s.mu[idx] = value

    def _get_state_field(self, node_id: str, idx: int, default: float = 0.0) -> float:
        s = self._get_state(node_id)
        return float(s.mu[idx]) if s is not None else default

    # --- Weight / score access ---

    def get_weight(self, node_id: str) -> float:
        return self._get_state_field(node_id, 0)

    def get_score(self, node_id: str) -> float:
        return self._get_state_field(node_id, 1)

    def penalize_id(self, node_id: str, penalty: float = 0.5) -> None:
        """Manually decrease the weight of a node (e.g. after a reasoning failure)."""
        s = self._get_state(node_id)
        if s is not None:
            s.mu[0] *= (1.0 - penalty)
            s.mu[0] = max(s.mu[0], 1e-6)

    def boost_id(self, node_id: str, gain: float = 0.1) -> None:
        """Increase the weight of a node after a successful outcome."""
        s = self._get_state(node_id)
        if s is not None:
            s.mu[0] += self.alpha_gain * gain * (1.0 - s.mu[0])
            s.mu[0] = min(s.mu[0], 1.0)

    def save_state(self, path) -> None:
        """Persist weights and scores to a JSON file (reads from meta_forest)."""
        import json
        weights = {}
        scores = {}
        for node in self.meta_forest.active_nodes():
            if node.id.startswith("state:"):
                domain_id = node.id[len("state:"):]
                weights[domain_id] = float(node.mu[0])
                scores[domain_id] = float(node.mu[1])
        state = {"weights": weights, "scores": scores}
        Path(path).write_text(json.dumps(state))

    def load_state(self, path) -> None:
        """Restore weights and scores from a JSON file if it exists."""
        import json
        p = Path(path)
        if not p.exists():
            return
        state = json.loads(p.read_text())
        for nid, w in state.get("weights", {}).items():
            s = self._get_state(nid)
            if s is not None:
                s.mu[0] = w
        for nid, sc in state.get("scores", {}).items():
            s = self._get_state(nid)
            if s is not None:
                s.mu[1] = sc

    def prune(self, min_weight: float = 1e-4) -> int:
        """
        Remove leaf nodes whose weight falls below min_weight.
        Protected nodes are never pruned. Returns the count removed.
        """
        to_remove = []
        for node in self.meta_forest.active_nodes():
            if not node.id.startswith("state:"):
                continue
            domain_id = node.id[len("state:"):]
            w = float(node.mu[0])
            if w >= min_weight or domain_id in self.protected_ids:
                continue
            domain_node = self.forest.get(domain_id)
            if domain_node is not None and domain_node.children():
                continue  # only prune leaves
            to_remove.append(domain_id)
        for nid in to_remove:
            self.forest.deregister(nid)
            self.meta_forest.deregister(f"state:{nid}")
        return len(to_remove)

    # --- Observation loop ---

    def observe(self, x: np.ndarray, exhaustive: bool = False) -> ExplanationResult:
        """
        Run one observation pass for signal x.
        If exhaustive=True, explores deep into the hierarchy even if parents match.
        """
        result = self._expand(x, exhaustive=exhaustive)
        if self._recurrence is not None:
            self._recurrence.update(x)
            self._sync_recurrence_hfn()
        self._update_weights(x, result)
        self._check_prior_plasticity(x)
        self._update_scores(result)
        self._track_cooccurrence(result.explanation_tree)
        self._check_absorption()
        self._check_node_creation(x, result)
        return result

    def _expand(self, x: np.ndarray, exhaustive: bool = False) -> ExplanationResult:
        """
        Run the expansion loop and return the explanation tree.
        """
        query_node = HFN(mu=x, sigma=np.ones_like(x), id="__query__", use_diag=True)
        frontier = list(self.retriever.retrieve(query_node, k=10))
        explanation_tree: list[HFN] = []
        accuracy_scores: dict[str, float] = {}
        seen_ids: set[str] = set()
        surprising_leaves: list[HFN] = []
        budget = self.budget

        # Memoize surprise scores to avoid re-computing during sort + pop
        surprise_cache: dict[str, float] = {}

        def surprise(n: HFN) -> float:
            if n.id not in surprise_cache:
                surprise_cache[n.id] = self._kl_surprise(x, n)
            return surprise_cache[n.id]

        while frontier:
            # Priority = surprise - node_weight  (high surprise + low weight = expand first)
            # This makes the search cost-aware: trusted nodes explored last
            frontier.sort(
                key=lambda n: self.policy.expand_score(surprise(n), self.get_weight(n.id)),
                reverse=True
            )

            n = frontier.pop(0)
            if n.id in seen_ids:
                continue

            s = surprise(n)

            if s < self.tau:
                explanation_tree.append(n)
                accuracy_scores[n.id] = self.evaluator.accuracy(x, n)
                seen_ids.add(n.id)
                if not exhaustive:
                    continue

            children = n.children()
            if children and budget > 0:
                # Beam: only add top-k children by surprise to bound branching
                unseen = [c for c in children if c.id not in seen_ids]
                if len(unseen) > 5:
                    unseen.sort(key=lambda c: surprise(c), reverse=True)
                    unseen = unseen[:5]
                frontier.extend(unseen)
                budget -= 1
            elif s >= self.tau:
                seen_ids.add(n.id)
                surprising_leaves.append(n)

        residual_nodes = surprising_leaves + [
            n for n in frontier if n.id not in seen_ids and surprise(n) >= self.tau
        ]
        residual = float(np.mean([surprise(n) for n in residual_nodes])) if residual_nodes else 0.0

        if hasattr(self.retriever, 'notify_active'):
            self.retriever.notify_active([n.id for n in explanation_tree])

        return ExplanationResult(
            explanation_tree=explanation_tree,
            accuracy_scores=accuracy_scores,
            residual_surprise=residual,
            surprising_leaves=surprising_leaves
        )

    def predict(self, result: ExplanationResult) -> np.ndarray | None:
        """Generate a prediction from current explanation (predictive coding step)."""
        if not result.explanation_tree or self.decoder is None:
            return None
        best = result.explanation_tree[0]
        dec = self.decoder.decode(best)
        if isinstance(dec, list) and dec:
            return dec[0].mu
        return None

    def _kl_surprise(self, x: np.ndarray, node: HFN) -> float:
        """Dimension-normalized KL divergence proxy."""
        D = x.shape[0]
        # Normalize by D to keep surprise in a range comparable to tau (usually ~1.0)
        return -node.log_prob(x) / D

    # --- Dynamic updates ---

    def _update_weights(self, x: np.ndarray, result: ExplanationResult) -> None:
        effective_explaining_ids = self.policy.active_explaining_ids(result.accuracy_scores)

        for node in self.forest.active_nodes():
            nid = node.id
            s = self._get_state(nid)
            if s is None:
                self._init_node(node)
                s = self._get_state(nid)

            # Apply global decay to all non-protected nodes
            if self.weight_decay_rate > 0 and nid not in self.protected_ids:
                s.mu[0] *= (1.0 - self.weight_decay_rate)

            if nid in self.protected_ids:

                if self.prior_plasticity:
                    if nid in effective_explaining_ids:
                        self._prior_hit_counts[nid] += 1
                        self._prior_miss_counts[nid] = 0
                    else:
                        self._prior_miss_counts[nid] += 1
                continue

            if nid in effective_explaining_ids:
                acc = result.accuracy_scores.get(nid, 0.0)
                s.mu[0] = self.policy.weight_update(
                    current_weight=float(s.mu[0]),
                    explaining=True,
                    accuracy=acc,
                    overlap_sum=0.0,
                )
                s.mu[2] = 0  # reset miss_count
                s.mu[3] += 1  # increment hit_count
            else:
                overlap_sum = 0.0
                for explaining_node in result.explanation_tree:
                    kappa = node.overlap(explaining_node)
                    overlap_sum += kappa
                s.mu[0] = self.policy.weight_update(
                    current_weight=float(s.mu[0]),
                    explaining=False,
                    accuracy=0.0,
                    overlap_sum=overlap_sum,
                )
                if any(
                    n.id in effective_explaining_ids and node.overlap(n) > self.absorption_overlap_threshold
                    for n in result.explanation_tree
                ):
                    s.mu[2] += 1  # increment miss_count

    def _check_prior_plasticity(self, x: np.ndarray) -> None:
        """Drift low-density priors toward observations they keep missing (HPM Section 2.6).

        A prior is eligible for revision when:
        - miss_count > prior_revision_threshold (consistently missing)
        - hit_rate < 0.5 (less than half encounters explain the observation)

        Drift: mu += prior_drift_rate * (x - mu)
        Counts reset after drift so prior gets fresh chance to stabilise.
        """
        if not self.prior_plasticity:
            return
        for nid in list(self.protected_ids):
            miss = self._prior_miss_counts[nid]
            if miss < self.prior_revision_threshold:
                continue
            hit = self._prior_hit_counts[nid]
            total = hit + miss
            if total == 0:
                continue
            hit_rate = hit / total
            if hit_rate >= 0.5:
                continue
            node = self.forest.get(nid)
            if node is None:
                continue
            node.mu = node.mu + self.prior_drift_rate * (x - node.mu)
            self._prior_miss_counts[nid] = 0
            self._prior_hit_counts[nid] = 0

    def _update_scores(self, result: ExplanationResult) -> None:
        explaining_ids = self.policy.active_explaining_ids(result.accuracy_scores)
        for node in self.forest.active_nodes():
            nid = node.id
            s = self._get_state(nid)
            if s is None:
                continue
            if nid in explaining_ids:
                acc = result.accuracy_scores.get(nid, 0.0)
                s.mu[1] = self.policy.node_utility(
                    acc,
                    self.evaluator.description_length(node),
                    self.lambda_complexity,
                )
            else:
                s.mu[1] = self.policy.node_utility(
                    0.0,
                    self.evaluator.description_length(node),
                    self.lambda_complexity,
                )

    def _cooc_id(self, pair: frozenset) -> str:
        """Deterministic id for a cooccurrence edge HFN."""
        a, b = sorted(pair)
        return f"cooc:{a}|{b}"

    def _get_cooc(self, pair: frozenset) -> HFN | None:
        return self.meta_forest.get(self._cooc_id(pair))

    def _get_cooc_count(self, pair: frozenset) -> int:
        c = self._get_cooc(pair)
        return int(c.mu[0]) if c is not None else 0

    def _track_cooccurrence(self, explanation_tree: list[HFN]) -> None:
        self._observation_step += 1
        ids = [n.id for n in explanation_tree]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = frozenset([ids[i], ids[j]])
                cid = self._cooc_id(pair)
                c = self.meta_forest.get(cid)
                if c is None:
                    c = HFN(
                        mu=np.array([1.0, float(self._observation_step), 0.0, 0.0]),
                        sigma=np.ones(4),
                        id=cid,
                        use_diag=True,
                    )
                    self.meta_forest.register(c)
                else:
                    c.mu[0] += 1  # count
                    c.mu[1] = float(self._observation_step)  # recency

    def _sync_recurrence_hfn(self) -> None:
        """Sync RecurrenceTracker statistics into a RecurrencePattern HFN in meta_forest."""
        if self._recurrence is None:
            return
        rec_id = "recurrence:global"
        rate = self._recurrence.recurrence_rate
        # Mean distance: average pairwise distance of buffer entries (approximated)
        buf = self._recurrence._buffer
        if len(buf) >= 2:
            dists = [float(np.linalg.norm(buf[-1] - b)) for b in buf[:-1]]
            mean_dist = float(np.mean(dists))
        else:
            mean_dist = 0.0
        rec_threshold = float(self._recurrence.recommended_threshold(
            self.compression_cooccurrence_threshold
        ))
        rec_node = self.meta_forest.get(rec_id)
        if rec_node is None:
            rec_node = HFN(
                mu=np.array([rate, mean_dist, rec_threshold, 0.0]),
                sigma=np.ones(4),
                id=rec_id,
                use_diag=True,
            )
            self.meta_forest.register(rec_node)
        else:
            rec_node.mu[0] = rate
            rec_node.mu[1] = mean_dist
            rec_node.mu[2] = rec_threshold

    # --- Absorption ---

    def _weights_dict(self) -> dict[str, float]:
        """Build a {node_id: weight} dict from meta_forest for evaluator calls."""
        return self.state_store.weights_dict()

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
            p_scores = self.evaluator.persistence_scores(snapshot, self._weights_dict())
            p_max = max(
                (v for v in p_scores.values() if v != float("inf")),
                default=1.0,
            )

        # Hausdorff geometric absorption — use evaluator.hausdorff_candidates
        if self.hausdorff_absorption_threshold is not None:
            candidates = self.evaluator.hausdorff_candidates(
                snapshot,
                self._weights_dict(),
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
                bm_s = self._get_state(best_match.id)
                new_s = self._get_state(new_node.id)
                if bm_s is not None and new_s is not None:
                    new_s.mu[0] = bm_s.mu[0]
                    new_s.mu[1] = bm_s.mu[1]
                self.absorbed_ids.add(node.id)

        for node in snapshot:
            if node.id in self.protected_ids:
                continue
            # Skip if already absorbed in this pass
            if node.id not in self.forest:
                continue

            # Original miss-count absorption
            # Multifractal crowding: lower threshold for nodes in hot spots
            crowding_hotspot = False
            if self.multifractal_guided_absorption:
                non_protected = [n for n in snapshot if n.id not in self.protected_ids]
                crowding = self.evaluator.crowding(
                    node.mu, non_protected, self.multifractal_crowding_radius
                )
                crowding_hotspot = crowding >= self.multifractal_crowding_threshold

            coh = self.evaluator.coherence(node)
            ctx = AbsorptionContext(
                miss_count=self.state_store.miss_count(node.id),
                base_miss_threshold=self.absorption_miss_threshold,
                persistence_guided=self.persistence_guided_absorption,
                node_persistence=p_scores.get(node.id, 0.0),
                persistence_max=p_max,
                crowding_hotspot=crowding_hotspot,
                coherence=coh,
            )
            effective_miss_threshold = self.policy.effective_miss_threshold(ctx)
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

            absorb_score = self.policy.absorb_score(
                miss_count=ctx.miss_count,
                effective_threshold=effective_miss_threshold,
                overlap=best_overlap,
            )
            if not self.policy.should_absorb(absorb_score):
                continue

            if best_node is None:
                continue

            new_node = self.recombination.absorb(
                absorbed=node, dominant=best_node, forest=self.forest
            )
            self._init_node(new_node)
            bn_s = self._get_state(best_node.id)
            nn_s = self._get_state(new_node.id)
            if bn_s is not None and nn_s is not None:
                nn_s.mu[0] = bn_s.mu[0]
                nn_s.mu[1] = bn_s.mu[1]
            self.absorbed_ids.add(node.id)

    # --- Node creation ---

    def _check_node_creation(self, x: np.ndarray, result: ExplanationResult) -> None:
        create_allowed, create_score = self._create_signal(x, result)

        comp_threshold = self.policy.compression_threshold(
            self.compression_cooccurrence_threshold,
            self._recurrence.recommended_threshold(self.compression_cooccurrence_threshold)
            if self._recurrence is not None else None,
        )
        compression_candidates = self._collect_compression_candidates(comp_threshold)
        compress_allowed = len(compression_candidates) > 0
        compress_score = max(
            (
                self.policy.compress_score(int(self.meta_forest.get(cid).mu[0]), comp_threshold)
                for cid, _ in compression_candidates
                if self.meta_forest.get(cid) is not None
            ),
            default=float("-inf"),
        )

        coverage_gap = 1.0 - max(result.accuracy_scores.values(), default=0.0)
        gap_allowed = self.policy.should_query_gap(coverage_gap, self.gap_query_threshold)

        actions = self.policy.arbitrate_structure_actions(
            StructureArbitrationContext(
                create_allowed=create_allowed,
                create_score=create_score,
                compress_allowed=compress_allowed,
                compress_score=compress_score,
                gap_allowed=gap_allowed,
            )
        )

        did_create = False
        did_compress = False
        if actions.compress_first and compress_allowed:
            self._apply_compression_candidates(compression_candidates)
            did_compress = True
            if create_allowed:
                self._create_node(x)
                did_create = True
        elif actions.create_first and create_allowed:
            self._create_node(x)
            did_create = True
            if compress_allowed:
                self._apply_compression_candidates(compression_candidates)
                did_compress = True
        elif compress_allowed:
            self._apply_compression_candidates(compression_candidates)
            did_compress = True
        elif create_allowed:
            self._create_node(x)
            did_create = True

        if actions.run_gap_query and not did_create and not did_compress:
            self._check_gap_query(x, result)

    def _create_signal(self, x: np.ndarray, result: ExplanationResult) -> tuple[bool, float]:
        density_ratio = 0.0
        if self.lacunarity_guided_creation and len(self.forest) > 0:
            density_ratio = self.evaluator.density_ratio(
                x,
                list(self.forest.active_nodes()),
                self.lacunarity_creation_radius,
            )
        create_ctx = CreateContext(
            forest_size=len(self.forest),
            residual_surprise=result.residual_surprise,
            residual_threshold=self.residual_surprise_threshold,
            lacunarity_enabled=self.lacunarity_guided_creation,
            density_ratio=float(density_ratio),
            density_factor=self.lacunarity_creation_factor,
        )
        return (
            self.policy.should_create(create_ctx),
            self.policy.create_score(
                residual_surprise=result.residual_surprise,
                density_ratio=float(density_ratio),
                lacunarity_enabled=self.lacunarity_guided_creation,
            ),
        )

    def _create_node(self, x: np.ndarray) -> None:
        D = x.shape[0]
        node_id = f"{self.node_prefix}{len(self.forest)}"
        while node_id in self.forest:
            node_id = f"{node_id}_{np.random.randint(1000)}"

        if self.node_use_diag:
            new_node = HFN(
                mu=x.copy(),
                sigma=np.ones(D),
                id=node_id,
                use_diag=True,
            )
        else:
            new_node = HFN(
                mu=x.copy(),
                sigma=np.eye(D),
                id=node_id,
            )
        self.register(new_node)

    def _collect_compression_candidates(self, threshold: float) -> list[tuple[str, list[str]]]:
        # Collect all qualifying pairs from cooccurrence HFNs in meta_forest
        candidates: list[tuple[str, list[str]]] = []
        for cooc_node in list(self.meta_forest.active_nodes()):
            if not cooc_node.id.startswith("cooc:"):
                continue
            count = int(cooc_node.mu[0])
            if not self.policy.should_compress(count, threshold):
                continue
            # Parse ids from "cooc:idA|idB"
            pair_str = cooc_node.id[len("cooc:"):]
            ids = pair_str.split("|", 1)
            if len(ids) != 2:
                continue
            if ids[0] not in self.forest or ids[1] not in self.forest:
                continue
            compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
            if compressed_id in self.forest:
                continue
            candidates.append((cooc_node.id, ids))
        return candidates

    def _apply_compression_candidates(self, candidates: list[tuple[str, list[str]]]) -> None:
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

        # Process all candidates in ranked order
        for cooc_id, ids in candidates:
            node_a = self.forest.get(ids[0])
            node_b = self.forest.get(ids[1])
            if node_a is None or node_b is None:
                continue
            compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
            if compressed_id in self.forest:
                continue
            new_node = self.recombination.compress(
                node_a, node_b, self.forest, compressed_id
            )
            self._init_node(new_node)
            # Reset cooccurrence count
            cooc_node = self.meta_forest.get(cooc_id)
            if cooc_node is not None:
                cooc_node.mu[0] = 0

    def _check_compression_candidates(self) -> None:
        threshold = self.policy.compression_threshold(
            self.compression_cooccurrence_threshold,
            self._recurrence.recommended_threshold(self.compression_cooccurrence_threshold)
            if self._recurrence is not None else None,
        )
        candidates = self._collect_compression_candidates(threshold)
        self._apply_compression_candidates(candidates)

    # --- Gap query ---

    def _expand_with_budget(self, x: np.ndarray, budget: int) -> ExplanationResult:
        """Run _expand with a temporary budget override."""
        original_budget = self.budget
        self.budget = budget
        try:
            return self._expand(x)
        finally:
            self.budget = original_budget

    def _check_gap_query(self, x: np.ndarray, result: ExplanationResult) -> None:
        """
        Third step of _check_node_creation: if coverage is low and a Query/Converter
        are configured, fetch external knowledge and inject it into the Forest.
        """
        if self.query is None or self.converter is None:
            return
        if self._in_gap_query:
            return

        # Coverage gap = 1 - best accuracy in explanation tree
        coverage_gap = 1.0 - max(result.accuracy_scores.values(), default=0.0)
        if not self.policy.should_query_gap(coverage_gap, self.gap_query_threshold):
            return

        gap_hash = str(int(np.argmax(x)))
        query_id = f"query_{gap_hash}"
        if query_id in self.forest:
            return

        # Retry expansion with increasing budget to get a richer explanation context
        raw: list[str] = []
        for depth in range(self.max_expand_depth):
            expanded_result = self._expand_with_budget(x, self.budget * (depth + 1))
            raw = self.query.fetch(x, context=expanded_result)
            if raw:
                break

        if not raw:
            return

        D = x.shape[0]

        # Register the Query HFN (not protected — can be absorbed by dynamics)
        if self.node_use_diag:
            query_node = HFN(mu=x.copy(), sigma=np.ones(D), id=query_id, use_diag=True)
        else:
            query_node = HFN(mu=x.copy(), sigma=np.eye(D), id=query_id)
        self.register(query_node)

        # Separate signature raws from context raws
        sig_raws = [r for r in raw if r.startswith("sig: ")]
        ctx_raws = [r for r in raw if not r.startswith("sig: ")]

        sig_node = None
        if sig_raws:
            # Build bag-of-tokens mu for the Signature HFN
            sig_vecs = []
            for r in sig_raws:
                vecs = self.converter.encode(r, D)
                sig_vecs.extend(vecs)
            if sig_vecs:
                sig_mu = np.mean(sig_vecs, axis=0)
                sig_id = f"sig_{gap_hash}"
                if self.node_use_diag:
                    sig_node = HFN(mu=sig_mu, sigma=np.ones(D), id=sig_id, use_diag=True)
                else:
                    sig_node = HFN(mu=sig_mu, sigma=np.eye(D), id=sig_id)
                self.register(sig_node)
                self.protected_ids.add(sig_id)
                # Wire: sig_node is a child of query_node
                query_node.add_child(sig_node)

                # If a function token name is known, wire it as child of sig_node
                if self.vocab is not None and int(gap_hash) < len(self.vocab):
                    token_name = f"word_{self.vocab[int(gap_hash)]}"
                    if token_name in self.forest:
                        sig_node.add_child(self.forest.get(token_name))

        parent_node = sig_node if sig_node is not None else query_node

        # Inject knowledge-graph observations under the parent node
        self._in_gap_query = True
        try:
            for i, r in enumerate(ctx_raws):
                vecs = self.converter.encode(r, D)
                for j, vec in enumerate(vecs):
                    kg_id = f"gap_{gap_hash}_{i}_{j}"
                    if self.node_use_diag:
                        kg_node = HFN(mu=vec, sigma=np.ones(D), id=kg_id, use_diag=True)
                    else:
                        kg_node = HFN(mu=vec, sigma=np.eye(D), id=kg_id)
                    self.register(kg_node)
                    self.protected_ids.add(kg_id)
                    parent_node.add_child(kg_node)

                    # Run inner observation pipeline for this KG node
                    inner_result = self._expand_with_budget(vec, self.budget)
                    self._update_weights(vec, inner_result)
                    self._track_cooccurrence(inner_result.explanation_tree)
                    self._check_absorption()
                    create_allowed, _ = self._create_signal(vec, inner_result)
                    if create_allowed:
                        self._create_node(vec)
                    self._check_compression_candidates()
        finally:
            self._in_gap_query = False
