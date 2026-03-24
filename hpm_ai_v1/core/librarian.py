"""Librarian for HPM AI v3.1.

Specialized monitor that tracks the Global Project Manifold and broadcasts 
Structural Shift Signals when modules are refactored.
Now includes Manifold Redundancy Analysis for SP25.
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
        
        # Identify all dependent files that need an 'Automatic Litmus Turn'
        impacted = self.topology.get_impacted_files(old_name)
        return impacted

    def update_manifold(self, filepath: str, new_root: MMRNode):
        """Updates the global brain with the new structural truth."""
        self.topology.add_module(filepath, new_root)

    def analyze_manifold_redundancy(self) -> List[Tuple[str, str, float]]:
        """
        Identifies modules with high relational similarity (MMR overlap).
        Used to drive Architectural Forging (merging redundant modules).
        """
        print("Librarian: Analyzing Project Manifold for structural redundancies...")
        redundancies = []
        filepaths = list(self.topology.modules.keys())
        
        for i in range(len(filepaths)):
            for j in range(i + 1, len(filepaths)):
                path_a, path_b = filepaths[i], filepaths[j]
                mmr_a, mmr_b = self.topology.modules[path_a], self.topology.modules[path_b]
                
                # Compute relational overlap using root embeddings
                sim = float(np.dot(mmr_a.embedding, mmr_b.embedding))
                
                if sim > 0.95: # High structural overlap
                    # Check if they share similar child node distributions
                    redundancies.append((path_a, path_b, sim))
                    
        return sorted(redundancies, key=lambda x: x[2], reverse=True)
