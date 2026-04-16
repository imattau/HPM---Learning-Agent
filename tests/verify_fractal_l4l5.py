import numpy as np
import time
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.list_domain import ListDomainConfig
from hpm_ai_v2.utils.meta_controller import SolveRecord
from hfn.hfn import HFN

def test_fractal_l5():
    print("--- Testing Fractal L5 Refactor ---")
    config = ListDomainConfig()
    agent = BaseHFNAgent(config=config, use_fractal_meta=True)
    
    # Create a dummy pattern node
    pnode = HFN(mu=np.zeros(config.m_dim), sigma=np.ones(config.m_dim), id="dummy_pattern")
    agent.forest.register(pnode)
    
    # Record a success
    rec = SolveRecord(
        task_id="task1",
        goal_type="map",
        n_macros=0,
        strategy="bfs",
        depth=2,
        oracle_calls=5,
        success=True,
        wall_ms=10.0
    )
    agent.meta.record(rec, pnode)
    
    # Record a failure
    rec2 = SolveRecord(
        task_id="task2",
        goal_type="map",
        n_macros=0,
        strategy="bfs",
        depth=2,
        oracle_calls=10,
        success=False,
        wall_ms=20.0
    )
    agent.meta.record(rec2, pnode)
    
    # Check ranking
    ranked = agent.meta.rank_strategies("map", 0)
    print(f"Ranked strategies for map: {ranked}")
    assert ranked[0] == "bfs", f"Expected bfs to be top, got {ranked[0]}"
    
    # Verify nodes in meta forest
    meta_nodes = list(agent.meta.meta_forest.active_nodes())
    print(f"Meta forest node IDs: {[n.id for n in meta_nodes]}")
    
    # Find the aggregate node
    agg_node = agent.meta.meta_forest.get("meta:map:0:bfs")
    assert agg_node is not None
    assert agg_node.relation_type == "meta_pattern"
    # mu = [avg_success_rate, avg_oracle_calls, count, last_timestamp]
    assert np.isclose(agg_node.mu[0], 0.5)  # 1 success, 1 failure
    assert np.isclose(agg_node.mu[1], 7.5)  # (5 + 10) / 2
    assert np.isclose(agg_node.mu[2], 2.0)  # 2 records
    
    # Verify inputs
    assert len(agg_node.inputs) == 2
    for n in agg_node.inputs:
        assert n.relation_type == "solve_record"
        assert len(n.inputs) == 1
        assert n.inputs[0].id == "dummy_pattern"
        
    print("  [OK] Fractal L5 solve records and aggregate nodes verified.")

def test_fractal_l4():
    print("\n--- Testing Fractal L4 Refactor ---")
    config = ListDomainConfig()
    # BaseHFNAgent doesn't have _record_transitions unless mixin is used
    from hpm_ai_v2.agents.agents import ImaginativeAgent
    agent = ImaginativeAgent(config=config, use_hfn_forward_model=True)
    
    pnode = HFN(mu=np.zeros(config.m_dim), sigma=np.ones(config.m_dim), id="p1")
    agent.forest.register(pnode)
    
    # Record a transition
    # States are compute_state outputs (20D for list domain usually)
    s0 = np.zeros(config.S_DIM)
    s1 = np.ones(config.S_DIM)
    
    # We need to simulate a path solve to call _record_transitions
    # or just call forward_model.record_path directly
    agent.forward_model.record_path([pnode], [s0, s1])
    
    delta_node = agent.forward_model.delta_forest.get(f"delta:p1")
    assert delta_node is not None
    assert delta_node.relation_type == "transition"
    assert len(delta_node.inputs) == 1
    assert delta_node.inputs[0].id == "p1"
    assert np.allclose(delta_node.mu, s1 - s0)
    
    print("  [OK] Fractal L4 delta nodes verified.")

if __name__ == "__main__":
    test_fractal_l5()
    test_fractal_l4()
