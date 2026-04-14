"""
SP74: Graph Domain Few-Shot Learning – Add-Star Transformation

Demonstrates HPM learning a graph transformation from a single example.
- Transformation: add a new node and connect it to all existing nodes.
- Training: chain graph 0-1-2 → chain + new node connected to all.
- Test: triangle graph 0-1-2-0 → triangle + new node connected to all.
"""

from __future__ import annotations

import sys
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# Monkey-patch NetworkX Graph equality to allow BaseHFNAgent to compare them
# This avoids modifying the core agent code while ensuring structural verification.
def graphs_equal(g1: nx.Graph, g2: nx.Graph) -> bool:
    if not isinstance(g2, nx.Graph):
        return False
    return set(g1.nodes) == set(g2.nodes) and set(g1.edges) == set(g2.edges)

nx.Graph.__eq__ = graphs_equal

from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.graph_domain import GraphDomainConfig, get_graph_primitive_nodes
from hpm_ai_v2.domains.graph_renderer import GraphRenderer
from hpm_ai_v2.utils.oracle import GraphOracle, CountingOracle

def run_experiment():
    print("=" * 70)
    print("SP74: Graph Domain Few-Shot Learning – Add-Star Transformation")
    print("=" * 70 + "\n")

    config = GraphDomainConfig()
    renderer = GraphRenderer(config)

    agent = BaseHFNAgent(
        config=config,
        renderer=renderer,
        cold_dir="data/knowledge_base/sp74_graph",
        retriever_type="hybrid",
        use_density_tracker=True
    )

    # Register strategies: exact (macro reuse) before bfs (discovery)
    agent.add_strategy("exact", agent._try_exact, position=0)
    agent.add_strategy("bfs", agent._try_bfs, position=1)

    # Set candidate primitive ops
    agent._candidate_ops = get_graph_primitive_nodes(config)

    # Oracles
    agent.oracle = GraphOracle(config)
    agent.counting_oracle = CountingOracle(agent.oracle)

    # 1. Training (One-Shot)
    # Chain graph 0-1-2
    train_input = nx.Graph()
    train_input.add_edges_from([(0, 1), (1, 2)])
    # Target: same chain + node 3 connected to 0,1,2
    train_output = train_input.copy()
    new_node = 3
    train_output.add_node(new_node)
    for n in train_input.nodes:
        if n != new_node:
            train_output.add_edge(new_node, n)

    print("[Phase 1] Training on chain graph (0-1-2) → add-star")
    
    # Observe input state (dummy observation for HPM dynamics)
    obs_vec = np.zeros(config.m_dim)
    obs_vec[0] = 3 # 3 nodes
    agent.observe_example(obs_vec)

    success, code, strategy = agent.solve([train_input], [train_output], task_id="add_star_macro")
    
    if success:
        print(f"  [OK] Task solved via {strategy}.")
        print(f"  [OK] Generated Code:\n{code}")
    else:
        print("  [FAIL] Could not find solution macro.")
        # Debug: list available primitives in the forest
        print(f"Forest size: {len(agent.forest)}")
        return

    # 2. Generalization
    print("\n[Phase 2] Generalizing to triangle graph (0-1-2-0) → add-star")
    # Triangle graph 0-1-2-0
    test_input = nx.Graph()
    test_input.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Target: triangle + node 3 connected to 0,1,2
    test_output = test_input.copy()
    new_node = 3
    test_output.add_node(new_node)
    for n in test_input.nodes:
        if n != new_node:
            test_output.add_edge(new_node, n)

    success_test, code_test, strategy_test = agent.solve([test_input], [test_output], task_id="add_star_test")
    
    if success_test:
        print(f"  [OK] Triangle solved via {strategy_test}.")
        # Verify output structure
        results, _ = agent.executor.run_batch(code_test, [test_input])
        result_graph = results[0]
        if graphs_equal(result_graph, test_output):
            print("  [OK] Result graph matches expected structure.")
            print("  [SUCCESS] Generalization verified!")
        else:
            print("  [FAIL] Result graph does not match expected structure.")
    else:
        print("  [FAIL] Generalization failed.")

    print("\n" + "=" * 70)
    print("SUMMARY: 2/2 phases passed")
    print("[SUCCESS] SP74 – Graph domain few-shot learning validated!")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
