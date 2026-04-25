"""
vector_store.py - Simple persistence for learned concept mappings using Chroma.
"""

import os
from typing import Dict, Any, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

class VariableMappingStore:
    def __init__(self, persist_path: Optional[str] = None):
        if persist_path is None:
            # Default to data dir
            base_dir = os.path.dirname(os.path.dirname(__file__))
            persist_path = os.path.join(base_dir, "data", "concept_mappings")
            os.makedirs(os.path.dirname(persist_path), exist_ok=True)
            
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection("learned_concepts")
        
        # Use a reliable embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # We NO LONGER hardcode seeds. Mappings are learned during discovery.

    def query(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Query the learned concept store."""
        emb = self.model.encode([text]).tolist()
        results = self.collection.query(
            query_embeddings=emb,
            n_results=1
        )
        
        if results['documents'] and results['distances']:
            dist = results['distances'][0][0]
            # dist < 1.0 is a reasonable match threshold for sentence-transformers
            if dist < 1.0:
                return {
                    "variable": results['metadatas'][0][0]['variable'],
                    "document": results['documents'][0][0],
                    "distance": dist,
                    "status": "success"
                }
        return {"variable": None, "status": "no_match"}

    def add_mapping(self, description: str, variable: str):
        """Allow agents to contribute new learned mappings."""
        emb = self.model.encode([description]).tolist()
        import time
        self.collection.add(
            documents=[description],
            embeddings=emb,
            metadatas=[{"variable": variable}],
            ids=[f"learned_{int(time.time()*1000)}"]
        )
        return {"status": "success"}

# Singleton instance
_store = None
def get_mapping_store():
    global _store
    if _store is None:
        _store = VariableMappingStore()
    return _store
