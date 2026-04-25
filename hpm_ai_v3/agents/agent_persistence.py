"""
agent_persistence.py - Unified checkpointing for HPM agents and their memory contents.
"""

import os
import torch
import json
from typing import Dict, Any, List, Optional

from .base_discovery import PureAgnosticDiscoveryAgent
from ..population import PatternPopulation
from ..tools.memory import save_vector_collection, load_vector_collection, set_kv_persistence


def save_population(population: PatternPopulation, path: str):
    """Save pattern population (policies and weights) to disk."""
    os.makedirs(path, exist_ok=True)
    # Save patterns metadata
    pattern_meta = []
    for i, p in enumerate(population.patterns):
        p_id = f"pattern_{i}_{p.id}"
        if hasattr(p, 'policy_net'):
            torch.save(p.policy_net.state_dict(), os.path.join(path, f"{p_id}.pt"))
        
        meta = {
            "id": p.id,
            "weight": p.weight,
            "accuracy": p.accuracy,
            "type": type(p).__name__,
            "action_type": getattr(p, 'action_type', None),
            "module": getattr(p, 'module', None),
            "function": getattr(p, 'function', None),
            "parameter_names": getattr(p, 'parameter_names', []),
            "arg_bindings": getattr(p, 'arg_bindings', {}),
            "is_positional": getattr(p, 'is_positional', False),
            "cost": getattr(p, 'cost', 0.01),
            "substrate_type": getattr(p, 'substrate_type', "unknown"),
            "source_code": getattr(p, 'source_code', None),
            "output_key": getattr(p, 'output_key', "result"),
            "required_keys": getattr(p, 'required_observation_keys', []),
            "model_file": f"{p_id}.pt" if hasattr(p, 'policy_net') else None
        }
        pattern_meta.append(meta)
    
    with open(os.path.join(path, "population_meta.json"), "w") as f:
        json.dump(pattern_meta, f)
    print(f"[Persistence] Saved {len(population.patterns)} patterns to {path}")


def load_population(path: str) -> Optional[PatternPopulation]:
    """Load pattern population from disk."""
    meta_path = os.path.join(path, "population_meta.json")
    if not os.path.exists(meta_path):
        return None
    
    with open(meta_path, "r") as f:
        pattern_meta = json.load(f)
    
    from .base_discovery import ActionPattern
    from ..symbolic_pattern import SymbolicPattern
    from ..tools.registry import ToolRegistry
    import torch
    import numpy as np
    
    patterns = []
    for meta in pattern_meta:
        p = None
        if meta["type"] == "ActionPattern":
            p = ActionPattern(
                action_type=meta["action_type"],
                module=meta["module"],
                function=meta["function"],
                pattern_id=meta["id"]
            )
            p.parameter_names = meta.get("parameter_names", [])
            p.arg_bindings = meta.get("arg_bindings", {})
            p.is_positional = meta.get("is_positional", False)
            p.cost = meta.get("cost", 0.01)
            p.substrate_type = meta.get("substrate_type", "functional_action")
            
        elif meta["type"] == "SymbolicPattern" or meta.get("substrate_type", "").startswith("symbolic"):
            # RECONSTRUCT SYMBOLIC FUNCTION
            source = meta.get("source_code")
            if source:
                namespace = {"ToolRegistry": ToolRegistry, "torch": torch, "np": np}
                try:
                    exec(source, namespace)
                    # Extract the function (pipeline, agent_pipeline, or compiled_action)
                    func_names = ["pipeline", "agent_pipeline", "compiled_action"]
                    forward_fn = next((namespace[n] for n in func_names if n in namespace), None)
                    
                    if forward_fn:
                        p = SymbolicPattern(
                            forward_fn=forward_fn,
                            required_keys=meta.get("required_keys", []),
                            pattern_id=meta["id"],
                            output_key=meta.get("output_key", "result")
                        )
                        p.source_code = source
                except Exception as e:
                    print(f"[Persistence] Failed to reconstruct symbolic pattern {meta['id']}: {e}")
            
        if p:
            p.weight = meta["weight"]
            p.accuracy = meta["accuracy"]
            p.substrate_type = meta.get("substrate_type", p.substrate_type)
            
            model_file = os.path.join(path, meta["model_file"]) if meta.get("model_file") else None
            if model_file and os.path.exists(model_file):
                if hasattr(p, 'policy_net'):
                    p.policy_net.load_state_dict(torch.load(model_file))
            patterns.append(p)
            
    return PatternPopulation(patterns)


def save_full_checkpoint(agent: PureAgnosticDiscoveryAgent, 
                         base_path: str, 
                         vector_collections: List[str] = None):
    """
    Save agent patterns and specific vector collections.
    """
    pop_path = os.path.join(base_path, "population")
    save_population(agent.population, pop_path)
    
    if vector_collections:
        for collection in vector_collections:
            coll_path = os.path.join(base_path, "memory", "vectors", collection)
            save_vector_collection(collection, coll_path)
            
    print(f"[Persistence] Full checkpoint saved to {base_path}")


def load_full_checkpoint(agent: PureAgnosticDiscoveryAgent, 
                         base_path: str, 
                         vector_collections: List[str] = None,
                         kv_path: str = None):
    """
    Load agent patterns and memory contents into an existing agent instance.
    """
    pop_path = os.path.join(base_path, "population")
    new_pop = load_population(pop_path)
    if new_pop:
        agent.population = new_pop
        
    if vector_collections:
        for collection in vector_collections:
            coll_path = os.path.join(base_path, "memory", "vectors", collection)
            load_vector_collection(collection, coll_path)
            
    if kv_path:
        set_kv_persistence(kv_path)
        
    print(f"[Persistence] Full checkpoint loaded from {base_path}")
