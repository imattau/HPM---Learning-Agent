"""
Retriever — attention mechanism over the HFN graph.

A separate layer answering "what is considered?" — complementing HFN's "what is possible?".
Decouples retrieval strategy from Observer and Decoder.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

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
        weight_provider: Callable[[str], float] | None = None,
        weight_penalty: float = 100.0,
    ):
        super().__init__(forest)
        self.target_slice = target_slice or slice(None)
        self.target_weight = target_weight
        self.weight_provider = weight_provider
        self.weight_penalty = weight_penalty

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
            
            dist = float(np.sum(weighted_diff**2))
            
            # Apply penalty for low-weight nodes if a provider is present
            if self.weight_provider:
                weight = self.weight_provider(node.id)
                # Multiplicative penalty: dist / (weight + epsilon)
                # This ensures low-weight nodes are heavily suppressed
                epsilon = 1e-6
                return dist / (weight + epsilon)
                
            # Structural Similarity Bonus (multi-arity macros)
            struct_bonus = 0.0
            if node.inputs and query.inputs:
                struct_bonus = len(node.inputs) / (1.0 + len(query.inputs))

            return dist - struct_bonus

        candidates.sort(key=goal_score)
        return candidates[:k]


class StructuralRetriever(Retriever):
    """
    Retrieves nodes by structural fingerprint similarity of their DAG.
    Fingerprint includes:
      - number of children, inputs, edges
      - histogram of relation types
      - depth of the sub-tree (approximated)
    """
    def __init__(self, forest: Forest, fingerprint_dim: int = 16):
        super().__init__(forest)
        self.fingerprint_dim = fingerprint_dim
        self._fingerprint_cache = {}  # node_id -> np.ndarray

    def _structural_fingerprint(self, node: HFN) -> np.ndarray:
        """Return a fixed-length vector encoding the node's DAG structure."""
        if node.id in self._fingerprint_cache:
            return self._fingerprint_cache[node.id]

        vec = np.zeros(self.fingerprint_dim)
        vec[0] = len(node.children())
        vec[1] = len(node.inputs) if node.inputs else 0
        vec[2] = len(node.edges())
        # Relation type histogram (if relation_type is set)
        rel_types = ["macro", "sequence", "grounded_op", "string_op", "meta_schema"]
        for i, rt in enumerate(rel_types):
            if node.relation_type == rt:
                vec[3 + i] = 1.0
        # Depth proxy: if leaf, 0; else 1 + max child depth (capped)
        if node.is_leaf():
            vec[10] = 0.0
        else:
            max_child_depth = 0
            for child in node.children():
                child_fp = self._structural_fingerprint(child)
                max_child_depth = max(max_child_depth, child_fp[10])
            vec[10] = min(1.0, max_child_depth / 10.0)
        # Additional features: has_children, has_inputs, has_edges
        vec[11] = 1.0 if node.children() else 0.0
        vec[12] = 1.0 if node.inputs else 0.0
        vec[13] = 1.0 if node.edges() else 0.0
        # Flag for macro vs primitive
        vec[14] = 1.0 if node.relation_type == "macro" else 0.0
        vec[15] = 1.0 if node.inputs and not node.children() else 0.0  # multi-arity leaf

        self._fingerprint_cache[node.id] = vec
        return vec

    def retrieve(self, query: HFN, k: int = 10) -> list[HFN]:
        q_fp = self._structural_fingerprint(query)
        scored = []
        for node in self.forest.active_nodes():
            fp = self._structural_fingerprint(node)
            dist = np.linalg.norm(q_fp - fp)
            # Lower distance = more structurally similar
            scored.append((dist, node))
        scored.sort(key=lambda x: x[0])
        return [node for _, node in scored[:k]]


class HybridRetriever(Retriever):
    """
    Combines geometric (mu) and structural similarity.
    Weights can be adjusted.
    """
    def __init__(self, forest: Forest,
                 geometric_weight: float = 0.5,
                 structural_weight: float = 0.5,
                 geometric_retriever: Retriever = None,
                 structural_retriever: Retriever = None):
        super().__init__(forest)
        self.geometric_weight = geometric_weight
        self.structural_weight = structural_weight
        self.geometric_retriever = geometric_retriever or GeometricRetriever(forest)
        self.structural_retriever = structural_retriever or StructuralRetriever(forest)

    def retrieve(self, query: HFN, k: int = 10) -> list[HFN]:
        # Get two candidate lists (over-fetch)
        geo_candidates = self.geometric_retriever.retrieve(query, k=k*3)
        struct_candidates = self.structural_retriever.retrieve(query, k=k*3)
        # Combine and deduplicate
        combined = {}
        for node in geo_candidates:
            combined[node.id] = node
        for node in struct_candidates:
            combined[node.id] = node

        # Score each node
        scored = []
        for node in combined.values():
            # Geometric score: 1/(1+Euclidean distance)
            geo_dist = np.linalg.norm(node.mu - query.mu)
            geo_score = 1.0 / (1.0 + geo_dist)
            # Structural score: 1/(1+structural distance)
            struct_dist = np.linalg.norm(
                self.structural_retriever._structural_fingerprint(node) -
                self.structural_retriever._structural_fingerprint(query)
            )
            struct_score = 1.0 / (1.0 + struct_dist)
            total = self.geometric_weight * geo_score + self.structural_weight * struct_score
            scored.append((total, node))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [node for _, node in scored[:k]]
