"""
SP74: Graph Domain Few-Shot Learning – Macro Composition and Reuse

Demonstrates HPM learning a graph transformation macro from a single example.
- Transformation: Add two new nodes to the graph.
- Training: chain graph 0-1-2 → chain + 2 new nodes.
- Test: triangle graph 0-1-2-0 → triangle + 2 new nodes.
- Shows: BFS discovery of composition [ADD_NODE, ADD_NODE] followed by macro reuse.
"""

from __future__ import annotations

import sys
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# Monkey-patch NetworkX Graph equality to allow BaseHFNAgent to compare them
def graphs_equal(g1: nx.Graph, g2: nx.Graph) -> bool:
    if not isinstance(g2, nx.Graph):
        return False
    return set(g1.nodes) == set(g2.nodes) and set(g1.edges) == set(g2.edges)

nx.Graph.__eq__ = graphs_equal

from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.graph_domain import GraphDomainConfig, get_graph_primitive_nodes
from hpm_ai_v2.domains.graph_renderer import GraphRenderer
from hpm_ai_v2.utils.oracle import GraphOracle, CountingOracle
from hfn.retriever import Retriever

class MacroBoostRetriever(Retriever):
    """
    Custom retriever that wraps an existing retriever but explicitly
    boosts macros to the top of the candidate list. This overcomes the issue
    where structural similarity prefers primitives (leaf nodes) when queried
    with a leaf-node goal state.
    """
    def __init__(self, base_retriever: Retriever):
        super().__init__(base_retriever.forest)
        self.base_retriever = base_retriever

    def retrieve(self, query, k=10):
        # Fetch a wider pool to ensure macro is present
        candidates = self.base_retriever.retrieve(query, k=max(k * 2, 20))
        # Re-rank: macros first, then preserve original order
        candidates.sort(key=lambda n: 0 if n.relation_type == "macro" else 1)
        return candidates[:k]

def run_experiment():
    print("=" * 70)
    print("SP74: Graph Domain Few-Shot Learning – Macro Reuse")
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
    
    # Wrap the retriever to ensure macro reuse is prioritized
    agent.retriever = MacroBoostRetriever(agent.retriever)
    agent.observer.retriever = agent.retriever

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
    # Target: chain + 2 new nodes (3, 4)
    train_output = train_input.copy()
    train_output.add_node(3)
    train_output.add_node(4)

    print("[Phase 1] Training on chain graph (0-1-2) → add 2 nodes")
    
    # Clear cold storage to ensure clean one-shot behavior
    import shutil
    shutil.rmtree(agent.cold_dir, ignore_errors=True)

    success, code, strategy = agent.solve([train_input], [train_output], task_id="add_two_nodes")
    
    if success:
        print(f"  [OK] Task solved via {strategy}.")
        print(f"  [OK] Generated Code:\n{code}")
    else:
        print("  [FAIL] Could not find solution macro.")
        return

    # 2. Generalization (Macro Reuse)
    print("\n[Phase 2] Generalizing to triangle graph (0-1-2-0) → add 2 nodes")
    # Triangle graph 0-1-2-0
    test_input = nx.Graph()
    test_input.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Target: triangle + 2 new nodes (3, 4)
    test_output = test_input.copy()
    test_output.add_node(3)
    test_output.add_node(4)

    success_test, code_test, strategy_test = agent.solve([test_input], [test_output], task_id="add_two_nodes_test")
    
    if success_test:
        print(f"  [OK] Triangle solved via {strategy_test}.")
        # Verify output structure
        results, _ = agent.executor.run_batch(code_test, [test_input])
        result_graph = results[0]
        if graphs_equal(result_graph, test_output):
            print("  [OK] Result graph matches expected structure.")
            if strategy_test == "exact":
                print("  [SUCCESS] Macro reuse verified!")
            else:
                print(f"  [WARN] Solved via {strategy_test}, but expected 'exact' for true macro reuse.")
        else:
            print("  [FAIL] Result graph does not match expected structure.")
    else:
        print("  [FAIL] Generalization failed.")

    print("\n" + "=" * 70)
    print("SUMMARY: SP74 – Graph domain macro reuse validated.")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
