"""
HPM Forest — a world model represented as an HFN node.

A Forest IS an HFN whose children are the active pattern nodes it contains.
Its Gaussian is a compressed summary of all its children — the world model identity.

The Forest has no dynamics of its own. All weight tracking, co-occurrence
monitoring, absorption decisions, and node creation are the Observer's responsibility.
"""

from __future__ import annotations

import numpy as np
from hfn.hfn import HFN


class Forest(HFN):
    """
    A Forest is an HFN node whose children are all active patterns.

    It extends HFN with:
    - O(1) register/deregister by node id
    - Proximity retrieval in latent space Z
    - Gaussian kept in sync with its population (world model identity)

    It does NOT hold weights, scores, or any dynamic state.
    Those belong to the Observer.
    """

    def __init__(self, D: int = 4, forest_id: str = "forest"):
        super().__init__(
            mu=np.zeros(D),
            sigma=np.eye(D),
            id=forest_id,
        )
        self._D = D
        self._registry: dict[str, HFN] = {}

    # --- Registry ---

    def register(self, node: HFN) -> None:
        """Add a node to the Forest. Idempotent."""
        if node.id not in self._registry:
            self._registry[node.id] = node
            self._sync_gaussian()

    def deregister(self, node_id: str) -> None:
        """Remove a node from the Forest."""
        if node_id in self._registry:
            del self._registry[node_id]
            self._sync_gaussian()

    def active_nodes(self) -> list[HFN]:
        return list(self._registry.values())

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    # --- HFN interface overrides ---

    def children(self) -> list[HFN]:
        """All registered nodes are the Forest's children."""
        return self.active_nodes()

    # --- Retrieval ---

    def retrieve(self, x: np.ndarray, k: int = 5) -> list[HFN]:
        """
        Return the k nearest active nodes by Euclidean distance of μ to x.
        These are candidate roots for the Observer's expansion.
        """
        if not self._registry:
            return []
        scored = sorted(
            self._registry.values(),
            key=lambda n: float(np.linalg.norm(n.mu - x)),
        )
        return scored[:k]

    # --- Internal ---

    def _sync_gaussian(self) -> None:
        """Keep the Forest's Gaussian in sync with its population."""
        if not self._registry:
            self.mu = np.zeros(self._D)
            self.sigma = np.eye(self._D)
        else:
            mus = np.stack([n.mu for n in self._registry.values()])
            self.mu = mus.mean(axis=0)
            sigmas = np.stack([n.sigma for n in self._registry.values()])
            self.sigma = sigmas.mean(axis=0)
