"""
HPM Agnostic Decoder — domain-agnostic top-down synthesis.
Resides in the core hfn/ folder. Knows only geometry and topology.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Union
from dataclasses import dataclass

if TYPE_CHECKING:
    from hfn.hfn import HFN, Edge
    from hfn.forest import Forest

from hfn.retriever import Retriever, GeometricRetriever

@dataclass
class ResolutionRequest:
    """Emitted by Decoder when a generative goal cannot be met by the current Target Forest."""
    missing_mu: np.ndarray
    missing_sigma: np.ndarray
    required_edges: list[Edge]

class Decoder:
    """
    Collapses abstract HFN nodes (high variance) into concrete leaf nodes (low variance)
    from a target manifold.
    """
    # Relation importance weights — higher = more critical to match
    _RELATION_WEIGHTS: dict[str, float] = {
        "MUST_SATISFY": 3.0,   # hard structural constraint
        "PART_OF": 2.0,         # parent-child composition
        "spatial": 1.5,
        "temporal": 1.5,
        "recombined": 0.5,      # soft/emergent, lower penalty for missing
    }
    _DEFAULT_RELATION_WEIGHT = 1.0

    def __init__(
        self,
        target_forest: Forest,
        sigma_threshold: float = 1e-3,
        k_candidates: int = 10,
        retriever: Retriever = None,
    ):
        self.target_forest = target_forest
        self.sigma_threshold = sigma_threshold
        self.k_candidates = k_candidates
        self.retriever = retriever or GeometricRetriever(target_forest)

    def decode(self, node: HFN) -> Union[List[HFN], ResolutionRequest]:
        """
        Recursively collapses node into concrete leaves from target_forest.
        Returns a ResolutionRequest if no suitable concrete candidate is found.
        """
        # 1. Concrete Check: Is this node already an output?
        # Use mean of diagonal sigma as variance proxy for uniformity
        variance = np.mean(node.sigma) if node.use_diag else np.mean(np.diag(node.sigma))
        
        if variance <= self.sigma_threshold:
            return [node]

        # 2. Explicit Expansion: Does it have children?
        children = node.children()
        if children:
            results = []
            for child in children:
                child_result = self.decode(child)
                if isinstance(child_result, ResolutionRequest):
                    # If any child fails, the whole expansion stalls. Pass request up.
                    return child_result
                results.extend(child_result)
            return results

        # 3. Implicit Resolution: It's abstract but has no children. Resolve it.
        # Check if all edge targets exist in forest
        for edge in node.edges():
            target_id = edge.target.id
            target_nodes = [n for n in self.target_forest.active_nodes() if n.id == target_id]
            if not target_nodes:
                return ResolutionRequest(
                    missing_mu=node.mu,
                    missing_sigma=node.sigma,
                    required_edges=node.edges()
                )

        # Find candidates in the target forest near this abstract node's mu
        candidates = self.retriever.retrieve(node, k=self.k_candidates)
        if not candidates:
            return ResolutionRequest(
                missing_mu=node.mu,
                missing_sigma=node.sigma,
                required_edges=node.edges()
            )

        # Score candidates by topological fit
        # CRITICAL: We only want candidates that are CONCRETE (below threshold)
        # to satisfy the generative goal.
        valid_candidates = []
        for cand in candidates:
            cand_var = np.mean(cand.sigma) if cand.use_diag else np.mean(np.diag(cand.sigma))
            if cand_var <= self.sigma_threshold:
                valid_candidates.append(cand)

        if not valid_candidates:
            # We found rules/priors, but no concrete leaves. This is a gap.
            return ResolutionRequest(
                missing_mu=node.mu,
                missing_sigma=node.sigma,
                required_edges=node.edges()
            )

        best_candidate = None
        best_score = -float('inf')

        for cand in valid_candidates:
            score = self._score_topological_fit(node, cand)
            if score > best_score:
                best_score = score
                best_candidate = cand

        # If best_score < 0.0, we found NO candidate that satisfies the topological requirements
        if best_score < 0.0 and node.edges():
             return ResolutionRequest(
                missing_mu=node.mu,
                missing_sigma=node.sigma,
                required_edges=node.edges()
            )

        return [best_candidate] if best_candidate else []

    def _score_topological_fit(self, abstract_node: HFN, concrete_node: HFN) -> float:
        """
        Scores how well a concrete node satisfies the constraints of an abstract node.
        Relation types are weighted: structural constraints penalise harder for mismatches.
        """
        abstract_edges = abstract_node.edges()
        if not abstract_edges:
            return 0.0

        # Map concrete node's edges: target_id → set of relation types
        concrete_edge_map: dict[str, set[str]] = {}
        for e in concrete_node.edges():
            concrete_edge_map.setdefault(e.target.id, set()).add(e.relation)

        score = 0.0
        for e in abstract_edges:
            w = self._RELATION_WEIGHTS.get(e.relation, self._DEFAULT_RELATION_WEIGHT)
            if e.target.id in concrete_edge_map:
                # Bonus if relation type also matches
                if e.relation in concrete_edge_map[e.target.id]:
                    score += w
                else:
                    score += w * 0.5   # target matches but relation differs
            else:
                score -= w * 2.0       # missing edge is critical — heavy penalty

        return score
