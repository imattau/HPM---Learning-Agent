"""
HFNLoader — base class for building and loading HFN nodes into forests.

Subclass this to define a set of HFN priors or task encodings for a given
manifold. The base class handles forest registration and provides helpers
for constructing diagonal-covariance HFN nodes.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hfn.hfn import HFN

if TYPE_CHECKING:
    from hfn.forest import Forest


class HFNLoader(ABC):
    """
    Base class for anything that produces HFN nodes to be added to a forest.

    Subclasses define:
    - ``dim``: the manifold dimensionality
    - ``build()``: construct and return the list of HFN nodes

    The loader then handles registration via ``load_into(forest)``.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Manifold dimensionality shared by all nodes this loader produces."""
        ...

    @abstractmethod
    def build(self) -> list[HFN]:
        """Construct and return HFN nodes for this loader."""
        ...

    def load_into(self, forest: Forest) -> list[HFN]:
        """Build all HFNs and register them into *forest*."""
        nodes = self.build()
        for node in nodes:
            forest.register(node)
        return nodes

    # ── Helpers available to all subclasses ─────────────────────────────────

    def _make_node(self, name: str, mu: np.ndarray, sigma: np.ndarray) -> HFN:
        """Wrap (mu, sigma) into a named diagonal-covariance HFN."""
        return HFN(mu=mu, sigma=sigma, id=name, use_diag=True)

    def _base_mu_sigma(self, default_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Return zero mu and uniform sigma of shape (dim,)."""
        return np.zeros(self.dim), np.ones(self.dim) * default_sigma
