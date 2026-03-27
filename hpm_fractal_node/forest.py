"""
HPM Forest — the environment all HFN nodes inhabit.

Responsibilities:
- Registry: maintains all active nodes with weights and scores
- Retrieval: proximity search in latent space Z
- Competitive dynamics: score-weighted weight updates per query
- Structural absorption: persistent overlap resolves into hierarchy
- Node creation: residual surprise and query-induced compression
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

from hpm_fractal_node.hfn import HFN, make_leaf, make_parent


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class QueryResult(NamedTuple):
    """Output of a single Observer query passed back to the Forest."""
    explanation_tree: list[HFN]        # nodes that participated in explaining x
    accuracy_scores: dict[str, float]  # node_id -> accuracy (higher = better)
    residual_surprise: float           # unexplained surprise remaining


@dataclass
class NodeRecord:
    node: HFN
    weight: float = 0.1               # starts low, must be earned
    score: float = 0.0
    queries_explained: int = 0
    queries_missed: int = 0           # consecutive queries where node lost to overlapping node


# ---------------------------------------------------------------------------
# Forest
# ---------------------------------------------------------------------------

class Forest:
    """
    The HPM Forest: maintains the population of active HFN nodes and drives
    learning dynamics through competitive weight updates and structural absorption.

    Forest invariant: never writes to or mutates any HFN node. All structural
    changes produce new node objects.
    """

    def __init__(
        self,
        lambda_complexity: float = 0.1,   # compression pressure in S = acc - λ·complexity
        alpha_gain: float = 0.1,           # weight gain rate for explaining nodes
        beta_loss: float = 0.05,           # weight loss rate for overlapping non-explaining nodes
        absorption_overlap_threshold: float = 0.5,   # κ threshold for structural absorption
        absorption_query_threshold: int = 5,          # consecutive misses before absorption
        residual_surprise_threshold: float = 2.0,     # threshold to spawn new leaf
        compression_cooccurrence_threshold: int = 3,  # M: recurrences to trigger compression
        w_init: float = 0.1,               # initial weight for new nodes
    ):
        self._records: dict[str, NodeRecord] = {}
        self.lambda_complexity = lambda_complexity
        self.alpha_gain = alpha_gain
        self.beta_loss = beta_loss
        self.absorption_overlap_threshold = absorption_overlap_threshold
        self.absorption_query_threshold = absorption_query_threshold
        self.residual_surprise_threshold = residual_surprise_threshold
        self.compression_cooccurrence_threshold = compression_cooccurrence_threshold
        self.w_init = w_init

        # Tracks co-occurrence of node pairs across explanation trees
        # key: frozenset of two node ids, value: count of co-appearances
        self._cooccurrence: dict[frozenset, int] = defaultdict(int)

        # Tracks which node ids have been absorbed (for test visibility)
        self.absorbed_ids: set[str] = set()

    # --- Registry ---

    def register(self, node: HFN) -> None:
        """Add a node to the Forest with initial weight."""
        if node.id not in self._records:
            self._records[node.id] = NodeRecord(node=node, weight=self.w_init)

    def deregister(self, node_id: str) -> None:
        """Remove a node from the active registry."""
        self._records.pop(node_id, None)

    def get_weight(self, node_id: str) -> float:
        return self._records[node_id].weight if node_id in self._records else 0.0

    def get_score(self, node_id: str) -> float:
        return self._records[node_id].score if node_id in self._records else 0.0

    def active_nodes(self) -> list[HFN]:
        return [r.node for r in self._records.values()]

    def __len__(self) -> int:
        return len(self._records)

    # --- Retrieval ---

    def retrieve(self, x: np.ndarray, k: int = 5) -> list[HFN]:
        """
        Return the k nearest active nodes by Euclidean distance of μ to x.
        These are candidate roots for the Observer.
        """
        if not self._records:
            return []
        scored = [
            (np.linalg.norm(r.node.mu - x), r.node)
            for r in self._records.values()
        ]
        scored.sort(key=lambda t: t[0])
        return [node for _, node in scored[:k]]

    # --- Post-query update ---

    def update(self, x: np.ndarray, result: QueryResult) -> None:
        """
        Process the output of one Observer query:
        1. Update weights and scores
        2. Track co-occurrence for compression
        3. Trigger structural absorption if conditions met
        4. Create new nodes if needed
        """
        explaining_ids = {n.id for n in result.explanation_tree}

        self._update_weights(result, explaining_ids)
        self._update_scores(result, explaining_ids)
        self._track_cooccurrence(result.explanation_tree)
        self._check_absorption()
        self._check_node_creation(x, result)

    def _update_weights(self, result: QueryResult, explaining_ids: set[str]) -> None:
        for record in list(self._records.values()):
            nid = record.node.id
            if nid in explaining_ids:
                acc = result.accuracy_scores.get(nid, 0.0)
                record.weight += self.alpha_gain * (1.0 - record.weight) * acc
                record.weight = min(record.weight, 1.0)
                record.queries_explained += 1
                record.queries_missed = 0
            else:
                # lose weight proportional to overlap with explaining nodes
                for explaining_node in result.explanation_tree:
                    kappa = record.node.overlap(explaining_node)
                    record.weight -= self.beta_loss * kappa * record.weight
                record.weight = max(record.weight, 0.0)
                if any(
                    record.node.overlap(n) > self.absorption_overlap_threshold
                    for n in result.explanation_tree
                ):
                    record.queries_missed += 1

    def _update_scores(self, result: QueryResult, explaining_ids: set[str]) -> None:
        for record in self._records.values():
            nid = record.node.id
            acc = result.accuracy_scores.get(nid, 0.0) if nid in explaining_ids else 0.0
            complexity = record.node.description_length()
            record.score = acc - self.lambda_complexity * complexity

    def _track_cooccurrence(self, explanation_tree: list[HFN]) -> None:
        """Increment co-occurrence count for every pair in the explanation tree."""
        ids = [n.id for n in explanation_tree]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = frozenset([ids[i], ids[j]])
                self._cooccurrence[key] += 1

    # --- Structural absorption ---

    def _check_absorption(self) -> None:
        """
        For each node that has missed too many queries while overlapping a
        stronger node: absorb it as a child of the stronger node.
        """
        for record in list(self._records.values()):
            if record.queries_missed < self.absorption_query_threshold:
                continue
            # Find the strongest overlapping node
            best_overlap = 0.0
            best_record = None
            for other_record in self._records.values():
                if other_record.node.id == record.node.id:
                    continue
                kappa = record.node.overlap(other_record.node)
                if kappa > self.absorption_overlap_threshold and kappa > best_overlap:
                    best_overlap = kappa
                    best_record = other_record

            if best_record is None:
                continue

            # Absorb: create new parent that contains both, replace winner in registry
            new_parent = best_record.node.recombine(record.node)
            new_parent_record = NodeRecord(
                node=new_parent,
                weight=best_record.weight,
                score=best_record.score,
                queries_explained=best_record.queries_explained,
            )
            self.deregister(best_record.node.id)
            self.deregister(record.node.id)
            self._records[new_parent.id] = new_parent_record
            self.absorbed_ids.add(record.node.id)

    # --- Node creation ---

    def _check_node_creation(self, x: np.ndarray, result: QueryResult) -> None:
        self._check_residual_surprise(x, result)
        self._check_compression_candidates()

    def _check_residual_surprise(self, x: np.ndarray, result: QueryResult) -> None:
        """Spawn a new leaf node if residual surprise exceeds threshold."""
        if result.residual_surprise >= self.residual_surprise_threshold:
            D = x.shape[0]
            new_node = HFN(
                mu=x.copy(),
                sigma=np.eye(D),
                id=f"leaf_{len(self._records)}",
            )
            self.register(new_node)

    def _check_compression_candidates(self) -> None:
        """
        Create a new internal node for any pair that has co-occurred
        compression_cooccurrence_threshold times and both are still active.
        """
        for pair, count in list(self._cooccurrence.items()):
            if count < self.compression_cooccurrence_threshold:
                continue
            ids = list(pair)
            if ids[0] not in self._records or ids[1] not in self._records:
                continue
            node_a = self._records[ids[0]].node
            node_b = self._records[ids[1]].node
            # Avoid creating duplicates: check if a compression node already exists
            compressed_id = f"compressed({ids[0][:8]},{ids[1][:8]})"
            if compressed_id in self._records:
                continue
            new_node = node_a.recombine(node_b)
            new_node.id = compressed_id  # type: ignore[misc]
            self.register(new_node)
            # Reset counter to avoid re-triggering
            self._cooccurrence[pair] = 0
