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
        try:
            chol = np.linalg.cholesky(self.sigma)
            z = np.linalg.solve(chol, diff)
            log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        except np.linalg.LinAlgError:
            diag = np.maximum(np.diag(self.sigma), 1e-9)
            z = diff / np.sqrt(diag)
            log_det = float(np.sum(np.log(diag)))
        D = self.mu.shape[0]
        return float(-0.5 * (np.dot(z, z) + log_det + D * np.log(2.0 * np.pi)))

    def overlap(self, other: HFN) -> float:
        """Gaussian overlap integral approx: exp(-0.5 * Mahalanobis(mu_self, mu_other))."""
        combined_sigma = self.sigma + other.sigma
        diff = self.mu - other.mu
        try:
            val = float(np.exp(-0.5 * diff @ np.linalg.solve(combined_sigma, diff)))
        except np.linalg.LinAlgError:
            val = 0.0
        return val

    def description_length(self) -> float:
        """Complexity proxy: non-zero mu components + off-diagonal sigma entries."""
        return float(
            np.sum(np.abs(self.mu) > 1e-6)
            + np.sum(np.abs(self.sigma - np.diag(np.diag(self.sigma))) > 1e-6)
            + self.sigma.shape[0]
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
