"""Librarian for HPM AI v3.2.1.

Tracks the Global Project Manifold and broadcasts Structural Shift Signals.
Implements Diversity-Driven Saliency and Metacognitive Tabooing to prevent 
stagnation loops.
"""
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from hpm_ai_v1.transpiler.mmr import ProjectTopology, MMRNode

class CodeLibrarian:
    def __init__(self, topology: ProjectTopology):
        self.topology = topology
        self.shift_history: List[Dict] = []
        self.taboo_list: Set[str] = set()
        self.failure_counts: Dict[str, int] = {}
        self.target_history: List[str] = []

    def broadcast_structural_shift(self, filepath: str, old_name: str, new_name: str):
        """Signals that a core relational node has moved in the manifold."""
        shift = {
            "filepath": filepath,
            "old_name": old_name,
            "new_name": new_name,
            "type": "rename"
        }
        self.shift_history.append(shift)
        print(f"Librarian: BROADCAST Structural Shift in {filepath}: {old_name} -> {new_name}")
        return self.topology.get_impacted_files(old_name)

    def update_manifold(self, filepath: str, new_root: MMRNode):
        """Updates the global brain with the new structural truth."""
        self.topology.add_module(filepath, new_root)
        # Reset failure count on success
        self.failure_counts[filepath] = 0
        if filepath in self.taboo_list:
            self.taboo_list.remove(filepath)

    def report_failure(self, filepath: str):
        """Metacognitive feedback from L5."""
        self.failure_counts[filepath] = self.failure_counts.get(filepath, 0) + 1
        if self.failure_counts[filepath] >= 3:
            print(f"Librarian: Module {filepath} entered TABOO state (Repeated failures).")
            self.taboo_list.add(filepath)

    def get_most_salient_target(self) -> Optional[str]:
        """
        Diversity-Driven Saliency Scanner.
        Prevents 'Saliency Traps' by factoring in historical targets and failures.
        """
        print("Librarian: Scanning Project Manifold for most salient refactor target...")
        targets = []
        
        for path, mmr in self.topology.modules.items():
            if path in self.taboo_list:
                continue
                
            # Base Saliency = Complexity / Connectivity
            node_count = self._get_node_count(mmr)
            in_degree = len(self.topology.in_edges.get(path, [])) + 1
            base_saliency = node_count / in_degree
            
            # Recency Penalty: Lower score if we just targeted this module
            recency_penalty = 0.0
            if self.target_history and self.target_history[-1] == path:
                recency_penalty = base_saliency * 0.8
            elif path in self.target_history:
                recency_penalty = base_saliency * 0.2
                
            score = base_saliency - recency_penalty
            targets.append((path, score))
                
        if not targets:
            if not self.taboo_list:
                return None
            # If all modules are taboo, clear oldest taboos
            print("Librarian: All salient targets tabooed. Clearing oldest taboos...")
            taboo_list_ordered = list(self.taboo_list)
            self.taboo_list = set(taboo_list_ordered[len(taboo_list_ordered)//2:])
            return self.get_most_salient_target()

        # Pick best target
        best_target, highest_score = max(targets, key=lambda x: x[1])
        
        self.target_history.append(best_target)
        if len(self.target_history) > 10:
            self.target_history.pop(0)
            
        print(f"Librarian: Saliency detected in {best_target} (Score={highest_score:.2f})")
        return best_target

    def _get_node_count(self, node: MMRNode) -> int:
        count = 1
        for child in node.children:
            count += self._get_node_count(child)
        return count
