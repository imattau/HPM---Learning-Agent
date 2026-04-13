"""
HPM Fractal Node — minimal structural implementation.

This module implements the HFN data structure only: Gaussian identity +
DAG polygraph body. No learning, no compression, no evaluator logic.
"""

from __future__ import annotations
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from hfn.probabilistic_models import ProbabilisticModel, FlatGaussianModel


class Edge(NamedTuple):
    source: HFN
    target: HFN
    relation: str


@dataclass(eq=False)
class HFN:
    """
    A single HPM Fractal Node.

    Two faces:
    - Compressed: Gaussian N(mu, sigma) — the node's predictive identity.
    - Structural: DAG of child HFN nodes with typed edges — internal composition.

    Invariants:
    - No parent references.
    - No mutation from queries.
    - Same node can be child of multiple parents simultaneously.
    - Identical interface at every depth (fractal uniformity).

    Memory modes:
    - use_diag=False (default): sigma is a D×D matrix. Backward-compatible.
    - use_diag=True: sigma is a D-vector (diagonal). O(D) storage instead of O(D²).
    """
    mu: np.ndarray
    sigma: np.ndarray
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    use_diag: bool = False
    _children: list[HFN] = field(default_factory=list, repr=False)
    _edges: list[Edge] = field(default_factory=list, repr=False)
    
    # NEW fields for multi-arity composition
    inputs: list[HFN] = field(default_factory=list, repr=False)
    outputs: list[HFN] = field(default_factory=list, repr=False)
    relation_type: str | None = None
    relation_params: dict = field(default_factory=dict, repr=False)

    # Probabilistic model (default = flat Gaussian)
    prob_model: Optional[ProbabilisticModel] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Probabilistic model initialization
        if self.prob_model is None:
            self.prob_model = FlatGaussianModel(self.mu, self.sigma, self.use_diag)

        # Caches for backward compatibility (kept for code that might access them directly)
        # Note: These are only guaranteed to be accurate for FlatGaussianModel
        self._sigma_diag = getattr(self.prob_model, '_sigma_diag', None)
        self._log_det_cached = getattr(self.prob_model, '_log_det_cached', 0.0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HFN):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    # --- Identity operations ---

    def log_prob(self, x: np.ndarray) -> float:
        """Log-probability of x under the node's probabilistic model."""
        return self.prob_model.log_prob(x)

    def overlap(self, other: HFN) -> float:
        """Overlap integral between this node and another node's probabilistic model."""
        return self.prob_model.overlap(other.prob_model)

    def description_length(self) -> float:
        """Complexity measure of the node's probabilistic model."""
        return self.prob_model.description_length()

    def update(self, x: np.ndarray, weight: float = 1.0, learning_rate: float = 0.1) -> None:
        """Update the node's internal probabilistic parameters based on observation."""
        self.prob_model.update(x, weight, learning_rate)
        # Sync mu/sigma fields for backward compatibility if the model provides them
        if hasattr(self.prob_model, 'mu'):
            self.mu = self.prob_model.mu
        if hasattr(self.prob_model, 'sigma'):
            self.sigma = self.prob_model.sigma

    # --- Structure operations (read-only) ---

    def children(self) -> list[HFN]:
        """Immediate child nodes. Empty list for leaves."""
        return list(self._children)

    def edges(self) -> list[Edge]:
        """Typed edges between immediate children."""
        return list(self._edges)

    def expand(self, depth: int) -> HFN:
        """
        Return this node as the root of a sub-tree to the given depth.
        At depth 0, or if this is a leaf, returns self.
        Fractal property: the return type is always HFN regardless of depth.
        """
        if depth == 0 or not self._children:
            return self
        return self  # structure is already present; depth controls traversal in caller

    def is_leaf(self) -> bool:
        return len(self._children) == 0

    # --- Recombination ---

    def recombine(self, other: HFN) -> HFN:
        """
        Produce a new parent HFN whose children are self and other.
        Gaussian is derived from the mean of both children's mu and sigma.
        Neither input node is mutated.
        If both nodes use_diag, the parent also uses diagonal storage.
        """
        new_mu = 0.5 * (self.mu + other.mu)
        if self.use_diag and other.use_diag:
            new_sigma = 0.5 * (self.sigma + other.sigma)
            parent = HFN(mu=new_mu, sigma=new_sigma, use_diag=True)
        else:
            # Expand diag node to full matrix if mixed
            s_sigma = np.diag(self.sigma) if self.use_diag else self.sigma
            o_sigma = np.diag(other.sigma) if other.use_diag else other.sigma
            new_sigma = 0.5 * (s_sigma + o_sigma)
            parent = HFN(mu=new_mu, sigma=new_sigma)
        parent._children = [self, other]
        parent._edges = [Edge(source=self, target=other, relation="recombined")]
        return parent

    # --- Construction helpers ---

    def add_relation(self, inputs: list[HFN], outputs: list[HFN] | None = None) -> None:
        """Link this node to multi-arity inputs and optional outputs."""
        self.inputs = inputs
        if outputs is not None:
            self.outputs = outputs

    def add_child(self, child: HFN, relation: str | None = None) -> None:
        """
        Add a child node to this node's polygraph.
        Optionally record an edge from the last existing child to this one.
        """
        if self._children and relation:
            self._edges.append(Edge(
                source=self._children[-1],
                target=child,
                relation=relation,
            ))
        self._children.append(child)

    def add_edge(self, source: HFN, target: HFN, relation: str) -> None:
        """Explicitly add a typed edge between two children."""
        self._edges.append(Edge(source=source, target=target, relation=relation))

    @staticmethod
    def query_node(
        known: np.ndarray,
        unknown_mask: np.ndarray,
        known_sigma: float = 0.5,
        unknown_sigma: float = 100.0,
        id: str = "query",
    ) -> HFN:
        """Create a goal node with tight sigma on known dims, loose on unknown.

        This is the standard HPM idiom for querying the Decoder: dimensions
        with high sigma are treated as "I don't know — fill this in."

        Parameters
        ----------
        known : (D,) array
            The mu vector (known values placed at their positions; unknown
            positions can be any value — they will be ignored by the Decoder
            because of the high sigma).
        unknown_mask : (D,) bool array
            True for dimensions that are unknown (will get high sigma).
        known_sigma : float
            Sigma value for known (pinned) dimensions.
        unknown_sigma : float
            Sigma value for unknown (free) dimensions.
        id : str
            Node id for the query node.
        """
        sigma = np.where(unknown_mask, unknown_sigma, known_sigma)
        return HFN(mu=known, sigma=sigma, id=id, use_diag=True)

    def __repr__(self) -> str:
        return f"HFN(id={self.id[:8]}, children={len(self._children)}, leaf={self.is_leaf()})"


# --- Factory helpers ---

def make_leaf(
    label: str,
    D: int = 4,
    rng: np.random.Generator | None = None,
    use_diag: bool = False,
) -> HFN:
    """Create a named leaf node with a stub Gaussian in R^D.

    use_diag=True: sigma stored as D-vector (diagonal), O(D) memory.
    use_diag=False (default): sigma stored as D×D matrix, backward-compatible.
    """
    rng = rng or np.random.default_rng(abs(hash(label)) % (2**31))
    mu = rng.standard_normal(D)
    variances = rng.uniform(0.5, 2.0, D)
    if use_diag:
        sigma = variances
    else:
        sigma = np.eye(D) * variances
    node = HFN(mu=mu, sigma=sigma, id=label, use_diag=use_diag)
    return node


def make_parent(
    label: str,
    children: list[HFN],
    edges: list[tuple] | None = None,
) -> HFN:
    """
    Create an internal node whose Gaussian is derived from its children.
    edges: list of (source_label, target_label, relation) strings.
    If all children use diagonal storage, the parent also uses diagonal storage.
    """
    mu = np.mean([c.mu for c in children], axis=0)
    all_diag = all(c.use_diag for c in children)
    if all_diag:
        sigma = np.mean([c.sigma for c in children], axis=0)
        node = HFN(mu=mu, sigma=sigma, id=label, use_diag=True)
    else:
        # Expand any diag children to full matrices before averaging
        sigmas = []
        for c in children:
            sigmas.append(np.diag(c.sigma) if c.use_diag else c.sigma)
        sigma = np.mean(sigmas, axis=0)
        node = HFN(mu=mu, sigma=sigma, id=label)
    node._children = list(children)
    if edges:
        child_by_id = {c.id: c for c in children}
        for src_id, tgt_id, rel in edges:
            node._edges.append(Edge(
                source=child_by_id[src_id],
                target=child_by_id[tgt_id],
                relation=rel,
            ))
    return node
