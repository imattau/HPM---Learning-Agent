"""Librarian for HPM AI v3.2.

Tracks the Global Project Manifold and broadcasts Structural Shift Signals.
Implements Manifold-Directed Saliency to identify refactoring targets.
"""
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from hpm_ai_v1.transpiler.mmr import ProjectTopology, MMRNode

class CodeLibrarian:
    def __init__(self, topology: ProjectTopology):
        self.topology = topology
        self.shift_history: List[Dict] = []

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

    def analyze_manifold_redundancy(self) -> List[Tuple[str, str, float]]:
        """Identifies modules with high relational similarity."""
        redundancies = []
        filepaths = list(self.topology.modules.keys())
        for i in range(len(filepaths)):
            for j in range(i + 1, len(filepaths)):
                path_a, path_b = filepaths[i], filepaths[j]
                mmr_a, mmr_b = self.topology.modules[path_a], self.topology.modules[path_b]
                sim = float(np.dot(mmr_a.embedding, mmr_b.embedding))
                if sim > 0.95:
                    redundancies.append((path_a, path_b, sim))
        return sorted(redundancies, key=lambda x: x[2], reverse=True)

    def get_most_salient_target(self) -> Optional[str]:
        """
        Manifold-Directed Saliency Scanner.
        Returns the module with the highest Structural Entropy (MDL / Connectivity).
        """
        print("Librarian: Scanning Project Manifold for most salient refactor target...")
        best_target = None
        highest_saliency = -1.0
        
        for path, mmr in self.topology.modules.items():
            # Saliency = Structural Complexity / Stickiness (In-degree)
            node_count = self._get_node_count(mmr)
            in_degree = len(self.topology.in_edges.get(path, [])) + 1
            
            # Complex files with few dependents are 'Entropy Islands'
            saliency = node_count / in_degree
            
            if saliency > highest_saliency:
                highest_saliency = saliency
                best_target = path
                
        if best_target:
            print(f"Librarian: Saliency detected in {best_target} (Score={highest_saliency:.2f})")
        return best_target

    def _get_node_count(self, node: MMRNode) -> int:
        count = 1
        for child in node.children:
            count += self._get_node_count(child)
        return count
