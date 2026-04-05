"""
Retriever — attention mechanism over the HFN graph.

A separate layer answering "what is considered?" — complementing HFN's "what is possible?".
Decouples retrieval strategy from Observer and Decoder.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfn.forest import Forest
    from hfn.hfn import HFN


class Retriever(ABC):
    """
    Attention mechanism selecting relevant HFN nodes from a forest.

    Answers: "Given this query, which parts of the graph matter right now?"
    """

    def __init__(self, forest: Forest):
        self.forest = forest

    @abstractmethod
    def retrieve(self, query: HFN, k: int = 10) -> list[HFN]:
        """Return up to k relevant nodes for this query."""
        ...


class GeometricRetriever(Retriever):
    """
    Default retriever: k-nearest by Euclidean distance of mu.
    Preserves existing forest.retrieve() behavior — backward compatible.
    """

    def retrieve(self, query: HFN, k: int = 10) -> list[HFN]:
        return self.forest.retrieve(query.mu, k=k)


class ContextualRetriever(Retriever):
    """
    Geometric retrieval + recency boost.

    Recently-active nodes are scored higher, implementing context-dependent selection.
    This allows the system to preferentially re-use patterns it's currently reasoning about.
    """

    def __init__(
        self,
        forest: Forest,
        recency_window: int = 20,
        recency_boost: float = 0.3,
    ):
        super().__init__(forest)
        self._recent: list[str] = []
        self._recency_window = recency_window
        self._recency_boost = recency_boost

    def notify_active(self, node_ids: list[str]) -> None:
        """Call after each observation with the explaining/active node IDs."""
        self._recent.extend(node_ids)
        self._recent = self._recent[-self._recency_window:]

    def retrieve(self, query: HFN, k: int = 10) -> list[HFN]:
        # Over-fetch to have candidates to re-rank
        candidates = self.forest.retrieve(query.mu, k=max(k * 2, 20))
        if not candidates:
            return []

        recent_set = set(self._recent)

        # Re-rank: penalize distance, boost recency
        def score(node: HFN) -> float:
            geo = float(np.sum((node.mu - query.mu) ** 2))
            boost = self._recency_boost if node.id in recent_set else 0.0
            return geo - boost  # lower = better

        candidates.sort(key=score)
        return candidates[:k]


class GoalConditionedRetriever(Retriever):
    """
    Retrieval weighted by a specific goal or target slice.

    Useful for intent-driven search where specific dimensions (e.g., Delta)
    must match precisely, while others (e.g., Input) are context.
    """

    def __init__(
        self,
        forest: Forest,
        target_slice: slice | None = None,
        target_weight: float = 10.0,
    ):
        super().__init__(forest)
        self.target_slice = target_slice or slice(None)
        self.target_weight = target_weight

    def retrieve(self, query: HFN, k: int = 10) -> list[HFN]:
        # Fetch a larger pool via standard forest lookup (fast index)
        candidates = self.forest.retrieve(query.mu, k=max(k * 5, 50))
        if not candidates:
            return []

        def goal_score(node: HFN) -> float:
            # Weighted Euclidean distance
            # High weight on target_slice increases its influence on ranking
            diff = node.mu - query.mu
            
            # Apply target weight to the specific slice
            weighted_diff = diff.copy()
            weighted_diff[self.target_slice] *= self.target_weight
            
            return float(np.sum(weighted_diff**2))

        candidates.sort(key=goal_score)
        return candidates[:k]
