"""
HFNLoader — base class for building and loading HFN nodes into forests.

Subclass this to define a set of HFN priors or task encodings for a given
manifold. The base class handles forest registration and provides helpers
for constructing diagonal-covariance HFN nodes.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

from hfn.hfn import HFN

if TYPE_CHECKING:
    from hfn.forest import Forest
    from hfn.observer import Observer


Role = Literal["prior", "seed", "aux"]


@dataclass(frozen=True)
class LoadItem:
    node: HFN
    role: Role
    protected: bool = False


class HFNLoader(ABC):
    """
    Base class for anything that produces HFN nodes to be added to a forest.

    Subclasses define:
    - ``dim``: the manifold dimensionality
    - ``build()``: construct and return the list of HFN nodes

    The loader then handles registration via ``load_into(forest)``.
    """

    namespace: str = ""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Manifold dimensionality shared by all nodes this loader produces."""
        ...

    @abstractmethod
    def build(self) -> list[HFN] | list[LoadItem]:
        """Construct and return HFN nodes (or LoadItems) for this loader."""
        ...

    def load_into(self, forest: "Forest", observer: "Observer | None" = None) -> list[LoadItem]:
        """Build HFNs, validate, and register them into *forest* (and observer if provided)."""
        items = self.build_items()
        self._validate(items, forest)

        protected_ids: set[str] = set()
        for item in items:
            if observer is not None:
                observer.register(item.node)
            else:
                forest.register(item.node)
            if item.protected:
                protected_ids.add(item.node.id)

        if protected_ids:
            if observer is not None:
                observer.protected_ids.update(protected_ids)
                forest.set_protected(set(observer.protected_ids))
            else:
                existing = set(getattr(forest, "_protected_ids", set()))
                forest.set_protected(existing | protected_ids)

        return items

    # ── Helpers available to all subclasses ─────────────────────────────────

    def _make_node(self, name: str, mu: np.ndarray, sigma: np.ndarray) -> HFN:
        """Wrap (mu, sigma) into a named diagonal-covariance HFN."""
        node_id = f"{self.namespace}:{name}" if self.namespace else name
        return HFN(mu=mu, sigma=sigma, id=node_id, use_diag=True)

    def _base_mu_sigma(self, default_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Return zero mu and uniform sigma of shape (dim,)."""
        return np.zeros(self.dim), np.ones(self.dim) * default_sigma

    def connect(self, parent: HFN, children: Sequence[HFN], relations: Sequence[str] | None = None) -> HFN:
        """
        Attach children to parent using the public HFN API.

        If relations is provided, its length must be len(children) - 1 and is
        used to label edges from the previous child to the current one.
        """
        if relations is not None and len(relations) != max(0, len(children) - 1):
            raise ValueError("relations must be length len(children)-1")
        existing_ids = {c.id for c in parent.children()}
        new_ids = [c.id for c in children]
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("duplicate child ids in connect")
        if existing_ids.intersection(new_ids):
            raise ValueError("child already attached to parent")

        for idx, child in enumerate(children):
            relation = relations[idx - 1] if relations is not None and idx > 0 else None
            parent.add_child(child, relation=relation)
        return parent

    def _item(self, node: HFN, role: Role = "prior", protected: bool = False) -> LoadItem:
        return LoadItem(node=node, role=role, protected=protected)

    def build_items(self) -> list[LoadItem]:
        return self._normalize_items(self.build())

    def build_nodes(self) -> list[HFN]:
        return [item.node for item in self.build_items()]

    def _normalize_items(self, items: Iterable[HFN] | Iterable[LoadItem]) -> list[LoadItem]:
        normalized: list[LoadItem] = []
        for item in items:
            if isinstance(item, LoadItem):
                normalized.append(item)
            elif isinstance(item, HFN):
                normalized.append(LoadItem(node=item, role="prior", protected=False))
            else:
                raise TypeError("build() must return HFN or LoadItem instances")
        return normalized

    def _validate(self, items: Sequence[LoadItem], forest: "Forest") -> None:
        ids = [item.node.id for item in items]
        dupes = {nid for nid in ids if ids.count(nid) > 1}
        if dupes:
            raise ValueError(f"duplicate node ids: {sorted(dupes)}")

        for item in items:
            node = item.node
            if node.mu.shape != (self.dim,):
                raise ValueError(f"mu shape mismatch for {node.id}: {node.mu.shape} != ({self.dim},)")
            if node.use_diag:
                if node.sigma.shape != (self.dim,):
                    raise ValueError(f"sigma shape mismatch for {node.id}: {node.sigma.shape} != ({self.dim},)")
            else:
                if node.sigma.shape != (self.dim, self.dim):
                    raise ValueError(
                        f"sigma shape mismatch for {node.id}: {node.sigma.shape} != ({self.dim}, {self.dim})"
                    )

        batch_ids = set(ids)
        for item in items:
            node = item.node
            children = node.children()
            if not children:
                continue
            child_ids = {c.id for c in children}
            for cid in child_ids:
                if cid not in batch_ids and forest.get(cid) is None:
                    raise ValueError(f"missing child {cid} for parent {node.id}")

            for edge in node.edges():
                if edge.source.id not in child_ids or edge.target.id not in child_ids:
                    raise ValueError(f"edge references non-child in {node.id}")
