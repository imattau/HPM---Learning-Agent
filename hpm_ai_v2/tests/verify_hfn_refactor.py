"""
Verification script for L4 (Forward Model) and L5 (Meta-Controller) HFN Refactor.
Ensures that the HFN-based components correctly store and retrieve state.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from pathlib import Path
import shutil
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hpm_ai_v2.domains.graph_domain import GraphDomainConfig, get_graph_primitive_nodes
from hpm_ai_v2.domains.graph_renderer import GraphRenderer
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l4_forward import L4ForwardModelMixin
from hfn.retriever import MacroPrioritizingRetriever

# Monkey-patch NetworkX Graph equality to allow BaseHFNAgent to compare them
def graphs_equal(g1: nx.Graph, g2: nx.Graph) -> bool:
    if not isinstance(g2, nx.Graph):
        return False
    return set(g1.nodes) == set(g2.nodes) and set(g1.edges) == set(g2.edges)

nx.Graph.__eq__ = graphs_equal
nx.Graph.__ne__ = lambda self, other: not self.__eq__(other)

class GraphImaginativeAgent(L4ForwardModelMixin, BaseHFNAgent):
    pass

def run_verification():
    print("=" * 70)
    print("Verifying L4/L5 HFN-based Refactor")
    print("=" * 70 + "\n")

    test_dir = Path("data/knowledge_base/test_hfn_refactor")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    config = GraphDomainConfig()
    renderer = GraphRenderer(config)

    # Initialize agent with HFN flags enabled
    agent = GraphImaginativeAgent(
        config=config,
        renderer=renderer,
        cold_dir=str(test_dir / "agent"),
        retriever_type="hybrid",
        use_hfn_forward_model=True,
        forward_model_cold_dir=str(test_dir / "l4"),
        use_hfn_meta_controller=True,
        meta_cold_dir=str(test_dir / "l5"),
        n_workers=1
    )
    
    # Primitives and strategies
    agent._candidate_ops = get_graph_primitive_nodes(config)
    agent.add_strategy("exact", agent._try_exact, position=0)
    agent.add_strategy("imagine", agent._try_imagine, position=1)
    agent.add_strategy("bfs", agent._try_bfs, position=2)
    
    # Retrieval
    agent.retriever = MacroPrioritizingRetriever(agent.retriever)
    agent.observer.retriever = agent.retriever

    # Oracles
    from hpm_ai_v2.utils.oracle import GraphOracle, CountingOracle
    agent.oracle = GraphOracle(config)
    agent.counting_oracle = CountingOracle(agent.oracle)

    # 1. Verify L5 HFN Meta Controller
    print("[1] Verifying L5 HFN Meta Controller")
    from hpm_ai_v2.utils.hfn_meta_controller import HFNMetaStrategyController
    assert isinstance(agent.meta, HFNMetaStrategyController), "Meta controller must be HFN-based"
    
    # Record a fake success
    from hpm_ai_v2.utils.meta_controller import SolveRecord
    rec = SolveRecord(task_id="test", goal_type="graph", n_macros=0, strategy="bfs", 
                      depth=1, oracle_calls=5, success=True, wall_ms=10)
    agent.meta.record(rec)
    
    ranked = agent.meta.rank_strategies("graph", 0)
    assert ranked[0] == "bfs", f"BFS should be ranked #1 after success, got {ranked[0]}"
    print("  [OK] L5 recording and ranking verified.")

    # 2. Verify L4 HFN Forward Model
    print("\n[2] Verifying L4 HFN Forward Model")
    from hpm_ai_v2.utils.hfn_forward_model import HFNStateTransitionModel
    assert isinstance(agent.forward_model, HFNStateTransitionModel), "Forward model must be HFN-based"
    
    # Learn ADD_NODE delta
    train_input = nx.Graph()
    train_input.add_node(0)
    train_output = train_input.copy()
    train_output.add_node(1)
    
    # Use BFS to solve and trigger recording
    path = agent._try_bfs([train_input], [train_output])
    if path is None:
        # Debugging
        goal_state = agent._outputs_to_goal_state([train_output])
        idx = agent.s_dim + agent.dim
        print(f"DEBUG: Goal state mu_delta[3:6] = {[f'{x:.4f}' for x in goal_state[idx+3:idx+6]]}")
        for p in agent._candidate_ops:
            code = agent.renderer.render(p)
            results, errors = agent.executor.run_batch(code, [train_input])
            state = agent.oracle.compute_state(results, errors, code)
            print(f"DEBUG: Prereq {p.id} -> state mu[3:6] = {[f'{x:.4f}' for x in state[3:6]]}")
    
    assert path is not None, "BFS must solve ADD_NODE"
    
    # Record transitions
    agent._record_transitions(path, [train_input])
    
    # Verify delta node exists
    node_id = path[0].id
    delta_node = agent.forward_model.delta_forest.get(f"delta:{node_id}")
    assert delta_node is not None, f"Delta node for {node_id} not created"
    assert np.linalg.norm(delta_node.mu) > 0, "Delta mu should be non-zero"
    print("  [OK] L4 transition recording verified.")

    # 3. Verify Mental Simulation (Imagination)
    print("\n[3] Verifying Imagination via HFN Forward Model")
    test_input = nx.Graph()
    test_input.add_node(0)
    test_input.add_node(1)
    test_output = test_input.copy()
    test_output.add_node(2) # Target is adding one node
    
    # Try imagination (should use learned delta node)
    path_im = agent._try_imagine([test_input], [test_output])
    assert path_im is not None, "Imagination must solve ADD_NODE using learned delta"
    print(f"  [OK] Imagination verified using HFN deltas. Strategy used: imagine")

    # 4. Verify Persistence
    print("\n[4] Verifying Fractal Persistence")
    agent.save_state()
    # Check if cold storage files were created (TieredForest saves .npz files)
    l4_files = list((test_dir / "l4").glob("*.npz"))
    l5_files = list((test_dir / "l5").glob("*.npz"))
    assert len(l4_files) > 0, "L4 cold storage (.npz) missing"
    assert len(l5_files) > 0, "L5 cold storage (.npz) missing"
    print("  [OK] HFN persistence verified.")

    print("\n" + "=" * 70)
    print("[SUCCESS] L4/L5 HFN-based Refactor Verified!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
