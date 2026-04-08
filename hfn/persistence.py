"""
HPM Persistence Manager — handles saving/loading of Forest and Observer states.
Ensures cumulative knowledge base across experiments.
"""
import json
import os
import pickle
from pathlib import Path
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.tiered_forest import TieredForest

class PersistenceManager:
    """
    Standardizes how HFN agents interact with the disk.
    Builds a cumulative knowledge base by persisting Forest nodes and Observer weights.
    """
    def __init__(self, root_dir: str = "data/knowledge_base"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, forest: Forest, observer: Observer, experiment_id: str):
        """Saves current knowledge state."""
        exp_dir = self.root_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Forest Persistence
        # If it's a TieredForest, nodes are already in its cold_dir.
        # But we want a snapshot of the current registry for this experiment.
        nodes_path = exp_dir / "forest_snapshot.pkl"
        with open(nodes_path, "wb") as f:
            # We only save the ID mapping. The actual node content is in the cold store
            # if using TieredForest, or we save them fully if it's a standard Forest.
            pickle.dump(forest._registry, f)
            
        # 2. Observer Persistence (Weights and Scores)
        weights_path = exp_dir / "observer_weights.json"
        observer.save_state(weights_path)
        
        print(f"  [PERSISTENCE] State saved to {exp_dir}")

    def load(self, forest: Forest, observer: Observer, experiment_id: str):
        """Loads knowledge from a previous state."""
        exp_dir = self.root_dir / experiment_id
        nodes_path = exp_dir / "forest_snapshot.pkl"
        weights_path = exp_dir / "observer_weights.json"
        
        if nodes_path.exists():
            with open(nodes_path, "rb") as f:
                registry = pickle.load(f)
                for nid, node in registry.items():
                    if nid not in forest:
                        forest._registry[nid] = node
            forest._stale_index = True
            forest._sync_gaussian()
            if hasattr(forest, "rebuild_hierarchy_cache"):
                forest.rebuild_hierarchy_cache()
            print(f"  [PERSISTENCE] Loaded structural snapshot from '{experiment_id}'")
            
        if weights_path.exists():
            observer.load_state(weights_path)
            print(f"  [PERSISTENCE] Loaded observer weights from '{experiment_id}'")
