"""
SP77: Cross-Domain Analogy Using Shared L3 Schemas.

Demonstrates that a relational pattern (meta-schema) learned in List domain
can be transferred to Graph domain zero-shot.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
import numpy as np
import networkx as nx

from hpm_ai_v2.domains.list_domain import ListDomainConfig
from hpm_ai_v2.domains.list_renderer import ListRenderer
from hpm_ai_v2.domains.graph_domain import GraphDomainConfig
from hpm_ai_v2.domains.graph_renderer import GraphRenderer
from hpm_ai_v2.utils.oracle import ListOracle, GraphOracle, CountingOracle
from hpm_ai_v2.agents.agents import SocialAnalogicalAgent

def run_experiment():
    print("================================================================================")
    print("SP77: Cross-Domain Analogy Using Shared L3 Schemas")
    print("================================================================================")

    shared_root = Path("data/knowledge_base/sp77_analogy")
    if shared_root.exists():
        shutil.rmtree(shared_root)
    shared_root.mkdir(parents=True)

    shared_forest_dir = shared_root / "forest"
    shared_l4_dir = shared_root / "l4"
    shared_l5_dir = shared_root / "l5"

    # --------------------------------------------------------------------------
    # Phase 1: Expert Agent in List Domain
    # --------------------------------------------------------------------------
    print("\n[Phase 1] Training Agent A (Expert, List Domain)")
    list_config = ListDomainConfig()
    # Ensure BLOCK_END is in concepts
    if "BLOCK_END" not in list_config.concepts:
        list_config.concepts.append("BLOCK_END")
        list_config.DIM += 1
        
    agent_a = SocialAnalogicalAgent(
        config=list_config,
        cold_dir=str(shared_forest_dir),
        forward_model_cold_dir=str(shared_l4_dir),
        meta_cold_dir=str(shared_l5_dir),
    )
    agent_a.renderer = ListRenderer(list_config)
    agent_a.oracle = ListOracle(list_config)
    agent_a.counting_oracle = CountingOracle(agent_a.oracle)

    # Seed expert macros
    print("  Seeding expert list macros...")
    # Get priors by ID for easy construction
    priors = {n.id.replace("prior_rule_", ""): n for n in agent_a.forest.active_nodes() if "prior_rule_" in n.id}
    
    # Check if all needed concepts are present
    needed = ["VAR_INP", "LIST_INIT", "FOR_LOOP", "ITEM_ACCESS", "OP_ADD", "OP_MUL2", "LIST_APPEND", "BLOCK_END"]
    for c in needed:
        if c not in priors:
            print(f"  [ERROR] Missing prior: {c}")
            return
            
    # MAP_add1: [[2, 3]] -> [[3, 4]]
    path_add1 = [
        priors["VAR_INP"], priors["LIST_INIT"], priors["FOR_LOOP"], 
        priors["ITEM_ACCESS"], priors["OP_ADD"], priors["LIST_APPEND"], 
        priors["BLOCK_END"]
    ]
    agent_a.register_macro("MAP_add1", path_add1)
    
    # MAP_mul2: [[2, 3]] -> [[4, 6]]
    path_mul2 = [
        priors["VAR_INP"], priors["LIST_INIT"], priors["FOR_LOOP"], 
        priors["ITEM_ACCESS"], priors["OP_MUL2"], priors["LIST_APPEND"], 
        priors["BLOCK_END"]
    ]
    agent_a.register_macro("MAP_mul2", path_mul2)
    
    # Extract meta-schema
    print("  Discovering L3 meta-schema...")
    # Get solved paths from registered macros
    solved_paths = [node.inputs for node in agent_a.patterns.values() if node.relation_type == "macro"]
    schema = agent_a.discover_meta_schema(solved_paths)
    if schema:
        print(f"  [OK] Discovered L3 schema: {schema.id}")
    else:
        print("  [ERROR] Failed to discover L3 schema.")
        return

    agent_a.save_state()
    print(f"  Agent A state saved to {shared_root}\n")

    # --------------------------------------------------------------------------
    # Phase 2: Novice Agent in Graph Domain
    # --------------------------------------------------------------------------
    print("[Phase 2] Initializing Agent B (Novice, Graph Domain)")
    graph_config = GraphDomainConfig()
    agent_b = SocialAnalogicalAgent(
        config=graph_config,
        cold_dir=str(shared_forest_dir),
        forward_model_cold_dir=str(shared_l4_dir),
        meta_cold_dir=str(shared_l5_dir),
    )
    agent_b.renderer = GraphRenderer(graph_config)
    agent_b.oracle = GraphOracle(graph_config)
    agent_b.counting_oracle = CountingOracle(agent_b.oracle)

    # --------------------------------------------------------------------------
    # Phase 3: Zero-Shot Cross-Domain Analogy
    # --------------------------------------------------------------------------
    print("\n[Phase 3] Target Task: Graph Relabeling (Zero-Shot)")
    
    # Task: increment labels of a path graph
    # Use NON-OVERLAPPING labels to avoid node merging during iterative relabeling
    G_in = nx.Graph()
    G_in.add_edges_from([(10, 20), (20, 30)]) # 10-20-30
    
    mapping = {n: n + 1 for n in G_in.nodes()}
    G_out = nx.relabel_nodes(G_in, mapping, copy=True) # 11-21-31
    
    print("  Task Graph In:  Nodes", list(G_in.nodes()), "Edges", list(G_in.edges()))
    print("  Task Graph Out: Nodes", list(G_out.nodes()), "Edges", list(G_out.edges()))

    # Attempt solve zero-shot using analogy
    print("\n  Attempting solve via strategy: 'analogy'...")
    success, code, strat = agent_b.solve([G_in], [G_out], task_id="task_graph_relabel")
    
    if success:
        print(f"  [SUCCESS] Agent B solved graph task via {strat}!")
        print(f"  Synthesized Code:\n---\n{code}\n---")
    else:
        print("  [FAILURE] Agent B failed to solve via analogy.")
        # Debugging: check if schemas are visible
        schemas = [n for n in agent_b.forest.active_nodes() if n.relation_type == "meta_schema"]
        print(f"  Visible schemas: {[n.id for n in schemas]}")

    print("\n================================================================================")
    if success:
        print("[FINAL] SP77 – Cross-Domain Analogy validated!")
    else:
        print("[FINAL] SP77 – Failed.")
    print("================================================================================")

if __name__ == "__main__":
    run_experiment()
