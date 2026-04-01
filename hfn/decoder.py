"""
HPM Agnostic Decoder — domain-agnostic top-down synthesis.
Resides in the core hfn/ folder. Knows only geometry and topology.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hfn.forest import Forest

class Decoder:
    """
    Collapses abstract HFN nodes (high variance) into concrete leaf nodes (low variance)
    from a target manifold.
    """
    def __init__(
        self, 
        target_forest: Forest, 
        sigma_threshold: float = 1e-3,
        k_candidates: int = 5
    ):
        self.target_forest = target_forest
        self.sigma_threshold = sigma_threshold
        self.k_candidates = k_candidates

    def decode(self, node: HFN) -> List[HFN]:
        """
        Recursively collapses node into concrete leaves from target_forest.
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
                results.extend(self.decode(child))
            return results

        # 3. Implicit Resolution: It's abstract but has no children. Resolve it.
        # Find candidates in the target forest near this abstract node's mu
        candidates = self.target_forest.retrieve(node.mu, k=self.k_candidates)
        if not candidates:
            return []

        # Score candidates by topological fit
        best_candidate = None
        best_score = -float('inf')

        for cand in candidates:
            score = self._score_topological_fit(node, cand)
            if score > best_score:
                best_score = score
                best_candidate = cand

        return [best_candidate] if best_candidate else []

    def _score_topological_fit(self, abstract_node: HFN, concrete_node: HFN) -> float:
        """
        Scores how well a concrete node satisfies the constraints of an abstract node.
        Logic: For every edge in abstract_node, does concrete_node share a path to the same target?
        """
        score = 0.0
        abstract_edges = abstract_node.edges()
        if not abstract_edges:
            return 0.0

        # We check the targets of the concrete node's edges
        concrete_edge_targets = {e.target.id for e in concrete_node.edges()}
        
        for e in abstract_edges:
            # If the abstract node requires a specific relationship to another node
            # we check if the candidate also has that relationship.
            if e.target.id in concrete_edge_targets:
                score += 1.0
            else:
                score -= 0.5 # Penalty for missing required constraint
        
        return score
