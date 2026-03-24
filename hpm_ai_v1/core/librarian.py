"""Librarian for HPM AI v3.0.

Specialized monitor that tracks the Global Project Manifold and broadcasts 
Structural Shift Signals when modules are refactored.
"""
from typing import Dict, List, Set, Optional
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
