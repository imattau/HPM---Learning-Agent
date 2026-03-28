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
    """
    mu: np.ndarray
    sigma: np.ndarray
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _children: list[HFN] = field(default_factory=list, repr=False)
    _edges: list[Edge] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # Cache diagonal for O(D) log_prob fast path.
        # All prior nodes use diagonal sigma; this avoids O(D³) Cholesky per call.
        diag = np.diag(self.sigma)
        if np.allclose(self.sigma, np.diag(diag)):
            self._sigma_diag: np.ndarray | None = np.maximum(diag, 1e-9)
            self._log_det_cached: float = float(np.sum(np.log(self._sigma_diag)))
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
        combined_sigma = self.sigma + other.sigma
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
        """
        new_mu = 0.5 * (self.mu + other.mu)
        new_sigma = 0.5 * (self.sigma + other.sigma)
        parent = HFN(mu=new_mu, sigma=new_sigma)
        parent._children = [self, other]
        parent._edges = [Edge(source=self, target=other, relation="recombined")]
        return parent

    # --- Construction helpers ---

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

    def __repr__(self) -> str:
        return f"HFN(id={self.id[:8]}, children={len(self._children)}, leaf={self.is_leaf()})"


# --- Factory helpers ---

def make_leaf(label: str, D: int = 4, rng: np.random.Generator | None = None) -> HFN:
    """Create a named leaf node with a stub Gaussian in R^D."""
    rng = rng or np.random.default_rng(abs(hash(label)) % (2**31))
    mu = rng.standard_normal(D)
    sigma = np.eye(D) * rng.uniform(0.5, 2.0, D)
    node = HFN(mu=mu, sigma=sigma, id=label)
    return node


def make_parent(label: str, children: list[HFN], edges: list[tuple] | None = None) -> HFN:
    """
    Create an internal node whose Gaussian is derived from its children.
    edges: list of (source_label, target_label, relation) strings.
    """
    mu = np.mean([c.mu for c in children], axis=0)
    sigma = np.mean([c.sigma for c in children], axis=0)
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
