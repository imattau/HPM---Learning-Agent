"""
SP76: Social Sharing of L4/L5 HFN Nodes.
Demonstrates zero-shot transfer of Forward Model deltas and Meta-Controller strategy rankings
via shared HFN-native cold storage.
"""
from __future__ import annotations
import numpy as np
import shutil
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hpm_ai_v2.domains.base import DomainConfig
from hpm_ai_v2.utils.oracle import ListOracle, CountingOracle
from hpm_ai_v2.domains.list_renderer import ListRenderer
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l4_forward import L4ForwardModelMixin

class SocialHFNAgent(L4ForwardModelMixin, BaseHFNAgent):
    """An agent equipped with HFN-native Forward Model and Meta-Controller."""
    pass

def run_experiment():
    print("=" * 80)
    print("SP76: Social Sharing of L4/L5 HFN Nodes")
    print("=" * 80 + "\n")

    # Define shared cold storage directories
    shared_root = Path("data/knowledge_base/sp76_social_shared")
    if shared_root.exists():
        shutil.rmtree(shared_root)
    shared_root.mkdir(parents=True)

    shared_forest_dir = shared_root / "forest"
    shared_l4_dir = shared_root / "l4"
    shared_l5_dir = shared_root / "l5"

    config = DomainConfig(["VAR_INP", "OP_MUL2", "LIST_INIT", "FOR_LOOP", "ITEM_ACCESS", "LIST_APPEND", "RETURN"])
    renderer = ListRenderer(config)
    
    print("[Phase 1] Training Agent A (Expert)")
    agent_a = SocialHFNAgent(
        config=config,
        renderer=renderer,
        cold_dir=str(shared_forest_dir),
        use_hfn_forward_model=True,
        forward_model_cold_dir=str(shared_l4_dir),
        use_hfn_meta_controller=True,
        meta_cold_dir=str(shared_l5_dir),
        n_workers=1
    )
    # Oracles and strategies
    agent_a.oracle = ListOracle(config)
    agent_a.counting_oracle = CountingOracle(agent_a.oracle)
    agent_a.add_strategy("exact", agent_a._try_exact)
    agent_a.add_strategy("bfs", agent_a._try_bfs)
    
    # 1. Learn MUL2
    print("  Learning 'OP_MUL2'...")
    # Map [2, 3] -> [4, 6]
    success, code, strat = agent_a.solve([2, 3], [4, 6], task_id="task_mul2")
    print(f"  Agent A solve success={success}, strategy={strat}, code={code}")
    assert success, "Agent A failed to solve training task"
    
    # Persist state to shared storage
    agent_a.save_state()
    print(f"  Agent A training complete. State saved to {shared_root}\n")

    print("[Phase 2] Initializing Agent B (Novice)")
    # Agent B points to the exact same shared directories
    agent_b = SocialHFNAgent(
        config=config,
        renderer=renderer,
        cold_dir=str(shared_forest_dir),
        use_hfn_forward_model=True,
        forward_model_cold_dir=str(shared_l4_dir),
        use_hfn_meta_controller=True,
        meta_cold_dir=str(shared_l5_dir),
        n_workers=1
    )
    agent_b.oracle = ListOracle(config)
    agent_b.counting_oracle = CountingOracle(agent_b.oracle)
    agent_b.add_strategy("exact", agent_b._try_exact)
    agent_b.add_strategy("imagine", agent_b._try_imagine) # Enable imagination for Agent B
    agent_b.add_strategy("bfs", agent_b._try_bfs)
    
    # Verify Agent B has access to Agent A's macros
    macros = agent_b.forest.active_nodes()
    # Found 5 priors + 1 task macro
    assert len(macros) >= 6, f"Agent B should see Agent A's macros, found {len(macros)}"
    print(f"  [OK] Agent B sees shared forest: {len(macros)} nodes.")

    # 3. Test L4 Sharing
    print("\n[Phase 3] Verifying L4 Sharing (Forward Model)")
    # Find the shared macro (it's the only macro in the forest)
    macros = [n for n in agent_b.forest.active_nodes() if n.relation_type == "macro"]
    assert len(macros) > 0, "Agent B must see the shared macro node"
    macro_node = macros[0]
    print(f"  [OK] Found shared macro: {macro_node.id}")
    
    start_input = [2, 3]
    # Baseline (empty path)
    id_code = ""
    res_id, err_id = agent_b.executor.run_batch(id_code, start_input)
    start_state = agent_b.oracle.compute_state(res_id, err_id, id_code)
    
    # Prediction: recursively decomposes macro into VAR_INP and OP_MUL2 deltas
    predicted_state = agent_b.forward_model.predict(start_state, macro_node)
    
    # Compute true state of the macro
    macro_code = agent_b.renderer.render(macro_node)
    res_true, err_true = agent_b.executor.run_batch(macro_code, start_input)
    true_state = agent_b.oracle.compute_state(res_true, err_true, macro_code)
    
    dist = np.linalg.norm(predicted_state - true_state)
    print(f"  Macro prediction distance: {dist:.4f}")
    assert dist < 0.1, f"Agent B macro prediction should be accurate via shared deltas, dist={dist}"
    print("  [OK] Agent B successfully predicts using Agent A's shared knowledge base.")

    # 4. Test L5 Sharing
    print("\n[Phase 4] Verifying L5 Sharing (Meta-Controller)")
    # Agent B ranks strategies for a 'scalar' goal (since we solved a scalar-like list task)
    # Actually, solve uses 'scalar' as default goal_type.
    ranked = agent_b.meta.rank_strategies("scalar", n_macros=1) # n_macros=1 because Agent A learned one
    print(f"  Ranked strategies: {ranked}")
    assert ranked[0] in ["bfs", "exact"], f"Agent B should prioritize expert strategies, got {ranked[0]}"
    print("  [OK] Agent B sees Agent A's strategy performance stats.")

    # 5. Test Zero-Shot Transfer
    print("\n[Phase 5] Zero-Shot Transfer Solve")
    # Agent B solves a new task: [10] -> [20] (OP_MUL2)
    success, path, strat = agent_b.solve([10], [20], task_id="task_mul2_zeroshot")
    assert success, "Agent B failed to solve zero-shot task"
    print(f"  [OK] Agent B solved task zero-shot using shared knowledge. Strategy: {strat}")

    print("\n" + "=" * 80)
    print("[SUCCESS] SP76 – Social Sharing of HFN-native L4/L5 validated!")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
