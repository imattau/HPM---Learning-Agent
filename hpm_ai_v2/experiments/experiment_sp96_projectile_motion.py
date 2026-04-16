#!/usr/bin/env python3
"""
SP96: Noisy Projectile Motion – Structure vs Surface (Vertical Drop Edition).

Demonstrates:
1. Discovery of quadratic structure (t^2) from noisy data.
2. Compression into a reusable macro.
3. Robustness to surface changes (scaling/units).
4. Structural sensitivity (failure/re-discovery when physics change to linear).
"""

import sys
import tempfile
import random
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.projectile_domain import ProjectileDomainConfig
from hpm_ai_v2.domains.projectile_renderer import ProjectileRenderer
from hpm_ai_v2.utils.oracle.projectile_oracle import ProjectileOracle
from hpm_ai_v2.utils.oracle.base import CountingOracle


# ----------------------------------------------------------------------
# 1. Data Generation
# ----------------------------------------------------------------------
def generate_drop_data(n_trajectories=20, points_per_traj=2, g=9.8, noise_sigma=0.05):
    """Quadratic motion: y = 0.5 * g * t^2"""
    inputs, outputs = [], []
    for _ in range(n_trajectories):
        for _ in range(points_per_traj):
            t = random.uniform(0.1, 2.0)
            y = 0.5 * g * (t**2)
            y_noisy = y + np.random.normal(0, noise_sigma)
            inp = np.array([0.0, 0.0, t, g] + [0.0]*16)
            inputs.append(inp)
            outputs.append(max(0.0, y_noisy))
    return inputs, outputs

def generate_linear_data(n_trajectories=20, points_per_traj=2, v=10.0, noise_sigma=0.05):
    """Linear motion: y = v * t"""
    inputs, outputs = [], []
    for _ in range(n_trajectories):
        for _ in range(points_per_traj):
            t = random.uniform(0.1, 2.0)
            y = v * t
            y_noisy = y + np.random.normal(0, noise_sigma)
            inp = np.array([0.0, v, t, 0.0] + [0.0]*16)
            inputs.append(inp)
            outputs.append(max(0.0, y_noisy))
    return inputs, outputs

def create_primitives(agent, config, concepts):
    primitives = []
    for concept in concepts:
        mu = np.zeros(config.m_dim)
        mu[config.S_DIM + config.concept_idx[concept]] = 1.0
        node = HFN(mu=mu, sigma=np.ones(config.m_dim), id=f"prior_rule_{concept}", use_diag=True)
        agent.observer.register(node, protected=True)
        primitives.append(node)
    return primitives

# ----------------------------------------------------------------------
# 2. Main Experiment
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("SP96: Noisy Physical Discovery – Structure vs Surface")
    print("=" * 80)

    config = ProjectileDomainConfig(s_dim=20)
    renderer = ProjectileRenderer(config)
    oracle = ProjectileOracle(config)
    cold_dir = Path(tempfile.mkdtemp(prefix="sp96_"))
    forest = TieredForest(D=config.m_dim, cold_dir=cold_dir / "forest")

    agent = BaseHFNAgent(
        config=config, renderer=renderer, forest=forest,
        n_workers=8, tolerance=0.15
    )
    agent.oracle = oracle
    agent.counting_oracle = CountingOracle(oracle)
    
    # Register core strategies
    agent.add_strategy("bfs", agent._try_bfs)
    agent.add_strategy("greedy", agent._try_greedy_chain)
    
    # [PHASE 1] Discovery from Noisy Data
    print("\n[Phase 1] Discovery: Vertical Drop (0.5 g t^2)")
    t1_in, t1_out = generate_drop_data()
    t1_concepts = ["VAR_T", "VAR_G", "OP_CONST_05", "OP_MUL", "OP_SQUARE"]
    agent._candidate_ops = create_primitives(agent, config, t1_concepts)

    success1, t1_code, _ = agent.solve(t1_in, t1_out, goal_type="map", task_id="drop", beam_width=2000)
    if not success1:
        print("  FAILED discovery.")
        return
    
    macro_drop = agent.patterns["drop"]
    print(f"  SUCCESS! Macro registered: {macro_drop.id}")

    # [PHASE 2] Robustness to Surface Change (Scaling)
    print("\n[Phase 2] Robustness: Scaling (Double Gravity)")
    t2_in, t2_out = generate_drop_data(g=19.6)
    
    # We want to show macro reuse, so we use greedy or just let the meta-controller decide.
    # But first we need to make sure the macro is active in context.
    if hasattr(agent.retriever.base_retriever, "notify_active"):
        agent.retriever.base_retriever.notify_active([macro_drop.id])
        
    # Scaling Test with BFS (Depth 1 should find the macro)
    # Ensure macro is in candidate ops
    agent._candidate_ops = create_primitives(agent, config, t1_concepts) + [macro_drop]
    
    agent._strategy_order = ["bfs"]
    success2, t2_code, strat2 = agent.solve(t2_in, t2_out, goal_type="map", task_id="scaled_drop", max_depth=4)
    if success2:
        print(f"  Scaling Test: SUCCESS (Strategy: {strat2})")
        print(f"  Macro Reuse: {'BEGIN MACRO' in t2_code}")
    else:
        print(f"  Scaling Test: FAILURE")
        return

    # [PHASE 3] Structural Sensitivity (Linear Motion)
    print("\n[Phase 3] Structural Sensitivity: Linear Motion (y = vt)")
    t3_in, t3_out = generate_linear_data(v=15.0)
    
    # Attempt with old macro (should fail or be low utility)
    success3_greedy, _, _ = agent.solve(t3_in, t3_out, goal_type="map", task_id="linear_macro")
    print(f"  Old Macro (Greedy) attempt: {'SUCCESS' if success3_greedy else 'REJECTED'}")
    
    # Re-discovery
    t3_concepts = ["VAR_V0", "VAR_T", "OP_MUL"]
    agent._candidate_ops = create_primitives(agent, config, t3_concepts)
    success3_bfs, t3_code, _ = agent.solve(t3_in, t3_out, goal_type="map", task_id="linear_bfs", max_depth=4)
    print(f"  New Linear Discovery: {'SUCCESS' if success3_bfs else 'FAILURE'}")

    # Verdict
    print("\n" + "=" * 80)
    if success1 and success2 and not success3_greedy and success3_bfs:
        print("[VERDICT] SP96 VALIDATED: HPM extracted quadratic structure from noise and showed structural sensitivity.")
    else:
        print("[VERDICT] Mixed results. Audit required.")
    print("=" * 80)

if __name__ == "__main__":
    main()
