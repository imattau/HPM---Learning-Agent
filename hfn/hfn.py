"""
HPM Fractal Node — minimal structural implementation.

This module implements the HFN data structure only: Gaussian identity +
DAG polygraph body. No learning, no compression, no evaluator logic.
"""

from __future__ import annotations
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple


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

    def __post_init__(self) -> None:
        # Cache diagonal for O(D) log_prob fast path.
        # All prior nodes use diagonal sigma; this avoids O(D³) Cholesky per call.
        if self.use_diag:
            # sigma is already a D-vector diagonal — use directly
            self._sigma_diag: np.ndarray | None = np.maximum(self.sigma, 1e-9)
            self._log_det_cached: float = float(np.sum(np.log(self._sigma_diag)))
        else:
            diag = np.diag(self.sigma)
            if np.allclose(self.sigma, np.diag(diag)):
                self._sigma_diag = np.maximum(diag, 1e-9)
                self._log_det_cached = float(np.sum(np.log(self._sigma_diag)))
            else:
                self._sigma_diag = None
                self._log_det_cached = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HFN):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    # --- Identity operations ---

    def log_prob(self, x: np.ndarray) -> float:
        """Log-probability of x under N(mu, sigma). Lower = more surprising."""
        diff = np.asarray(x, dtype=float) - self.mu
        D = self.mu.shape[0]
        if self._sigma_diag is not None:
            # O(D) fast path for diagonal covariance (all prior nodes)
            z2 = float(np.dot(diff * diff, 1.0 / self._sigma_diag))
            return -0.5 * (z2 + self._log_det_cached + D * np.log(2.0 * np.pi))
        try:
            chol = np.linalg.cholesky(self.sigma)
            z = np.linalg.solve(chol, diff)
            log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        except np.linalg.LinAlgError:
            diag = np.maximum(np.diag(self.sigma), 1e-9)
            z = diff / np.sqrt(diag)
            log_det = float(np.sum(np.log(diag)))
        return float(-0.5 * (np.dot(z, z) + log_det + D * np.log(2.0 * np.pi)))

    def overlap(self, other: HFN) -> float:
        """Gaussian overlap integral approx: exp(-0.5 * Mahalanobis(mu_self, mu_other))."""
        diff = self.mu - other.mu
        # O(D) fast path when both nodes have diagonal sigma (all prior/learned nodes)
        if self._sigma_diag is not None and other._sigma_diag is not None:
            combined_diag = self._sigma_diag + other._sigma_diag
            return float(np.exp(-0.5 * float(np.dot(diff * diff, 1.0 / combined_diag))))
        # Mixed case: one diag, one full — expand diag to full matrix for the full path
        s_sigma = np.diag(self._sigma_diag) if (self.use_diag and self._sigma_diag is not None) else self.sigma
        o_sigma = np.diag(other._sigma_diag) if (other.use_diag and other._sigma_diag is not None) else other.sigma
        combined_sigma = s_sigma + o_sigma
        try:
            val = float(np.exp(-0.5 * diff @ np.linalg.solve(combined_sigma, diff)))
        except np.linalg.LinAlgError:
            val = 0.0
        return val

    def description_length(self) -> float:
        """Complexity proxy: non-zero mu components + off-diagonal sigma entries."""
        D = self.mu.shape[0]
        if self._sigma_diag is not None:
            # Diagonal sigma — off-diagonal is exactly zero, skip the expensive check
            return float(np.sum(np.abs(self.mu) > 1e-6)) + D
        return float(
            np.sum(np.abs(self.mu) > 1e-6)
            + np.sum(np.abs(self.sigma - np.diag(np.diag(self.sigma))) > 1e-6)
            + D
        )

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
