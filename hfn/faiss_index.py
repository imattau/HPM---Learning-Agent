"""
FaissVectorIndex — A wrapper for FAISS approximate nearest neighbor search.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class FaissVectorIndex:
    """
    Manages a FAISS index for high-speed semantic retrieval.
    Automatically handles index rebuilding and dimensionality shifts.
    """
    def __init__(self, D: int):
        self.D = D
        self.index: Optional[faiss.Index] = None
        self.ids: List[str] = []
        
        if HAS_FAISS:
            # Use FlatIP for inner product (cosine similarity if normalized)
            # or FlatL2 for Euclidean distance.
            self.index = faiss.IndexFlatL2(D)
            
    def add(self, node_id: str, mu: np.ndarray) -> None:
        """Add a single vector to the index."""
        if not HAS_FAISS or self.index is None:
            return
            
        if mu.shape[0] != self.D:
            # Should not happen with proper bootstrapping, but handle it
            return
            
        self.ids.append(node_id)
        # FAISS expects float32
        self.index.add(np.array([mu], dtype=np.float32))
        
    def search(self, x: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search for top-k nearest neighbors."""
        if not HAS_FAISS or self.index is None or self.index.ntotal == 0:
            return []
            
        # Ensure x is float32 and matches D
        if x.shape[0] != self.D:
            new_x = np.zeros(self.D, dtype=np.float32)
            copy_len = min(x.shape[0], self.D)
            new_x[:copy_len] = x[:copy_len]
            x = new_x
        else:
            x = x.astype(np.float32)
            
        distances, indices = self.index.search(np.array([x]), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.ids):
                results.append((self.ids[idx], float(dist)))
        return results

    def reset(self, D: int) -> None:
        """Wipe and re-initialize the index with a new dimension."""
        self.D = D
        self.ids = []
        if HAS_FAISS:
            self.index = faiss.IndexFlatL2(D)

    def rebuild(self, id_to_mu: dict[str, np.ndarray]) -> None:
        """Full rebuild from a dictionary of vectors."""
        if not id_to_mu:
            return
            
        # Get first mu to check dimension
        first_mu = next(iter(id_to_mu.values()))
        self.reset(first_mu.shape[0])
        
        if not HAS_FAISS or self.index is None:
            return
            
        mu_list = []
        for nid, mu in id_to_mu.items():
            self.ids.append(nid)
            mu_list.append(mu)
            
        all_mu = np.array(mu_list, dtype=np.float32)
        self.index.add(all_mu)
