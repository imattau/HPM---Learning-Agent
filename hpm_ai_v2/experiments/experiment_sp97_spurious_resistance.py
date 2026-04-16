#!/usr/bin/env python3
"""
SP97: Spurious Correlation Resistance.

Demonstrates:
1. Agent being presented with easy-to-use spurious features (Z1 ≈ Y).
2. Evaluation under distribution shift where spurious correlations break.
3. Recovery and discovery of true physical invariants (t^2) despite distractors.
4. Preference for structure over surface statistics.
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
# 1. Data Generation (SP97 Specific)
# ----------------------------------------------------------------------
def generate_spurious_data(
    n_trajectories: int = 20,
    points_per_traj: int = 2,
    g: float = 9.8,
    noise_sigma: float = 0.05,
    confounded: bool = True
):
    """
    Generate data for Vertical Drop.
    If confounded=True, VAR_Z1 is highly correlated with Y.
    If confounded=False, VAR_Z1 is pure noise.
    """
    inputs, outputs = [], []
    for _ in range(n_trajectories):
        for _ in range(points_per_traj):
            t = random.uniform(0.1, 2.0)
            y_clean = 0.5 * g * (t**2)
            y_noisy = y_clean + np.random.normal(0, noise_sigma)
            
            # Spurious Z1: Correlated with Y if confounded
            if confounded:
                z1 = y_clean + np.random.normal(0, 0.02)
            else:
                z1 = random.uniform(0.0, 20.0) # Broken correlation
                
            # Spurious Z2: Correlated with T
            if confounded:
                z2 = t + np.random.normal(0, 0.05)
            else:
                z2 = random.uniform(0.0, 2.0)
                
            # Spurious Z3: Pure noise
            z3 = random.uniform(-1.0, 1.0)
            
            # Input: [theta, v0, t, g, z1, z2, z3, ...padding]
            inp = np.array([0.0, 0.0, t, g, z1, z2, z3] + [0.0]*13)
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
    print("SP97: Spurious Correlation Resistance")
    print("=" * 80)

    config = ProjectileDomainConfig(s_dim=20)
    renderer = ProjectileRenderer(config)
    oracle = ProjectileOracle(config)
    cold_dir = Path(tempfile.mkdtemp(prefix="sp97_"))
    forest = TieredForest(D=config.m_dim, cold_dir=cold_dir / "forest")

    agent = BaseHFNAgent(
        config=config, renderer=renderer, forest=forest,
        n_workers=8, tolerance=0.15
    )
    agent.oracle = oracle
    agent.counting_oracle = CountingOracle(oracle)
    agent.add_strategy("bfs", agent._try_bfs)
    agent.add_strategy("greedy", agent._try_greedy_chain)

    # Primitives: True physics + Spurious distractors
    concepts = ["VAR_T", "VAR_G", "VAR_Z1", "VAR_Z2", "VAR_Z3", "OP_CONST_05", "OP_MUL", "OP_SQUARE"]
    agent._candidate_ops = create_primitives(agent, config, concepts)

    # [PHASE 1] Confounded Discovery
    print("\n[Phase 1] Discovery on Confounded Data (Z1 ≈ Y)")
    t1_in, t1_out = generate_spurious_data(confounded=True)
    success1, t1_code, _ = agent.solve(t1_in, t1_out, goal_type="map", task_id="confounded_task", beam_width=1000)
    
    if success1:
        uses_z1 = "VAR_Z1" in t1_code or "inp[4]" in t1_code
        print(f"  Initial Solution found! Code uses Z1: {uses_z1}")
        if uses_z1:
            print("  (Expected: Agent exploited the easy spurious correlation)")
        else:
            print("  (Surprise: Agent found the true structure despite the distractor!)")
    else:
        print("  FAILED initial discovery.")
        return

    # [PHASE 2] Distribution Shift
    print("\n[Phase 2] Distribution Shift (Z1 broken)")
    t2_in, t2_out = generate_spurious_data(confounded=False) # Break Z1 correlation
    
    # Evaluate Phase 1 code on Shifted Data
    results, _ = agent.executor.run_batch(t1_code, t2_in)
    p2_success = agent._check_outputs(results, t2_out)
    print(f"  Phase 1 model performance on Shifted Data: {'STABLE' if p2_success else 'FAILED'}")
    
    if p2_success:
        print("  Verdict: Spurious model was surprisingly robust or agent found true structure.")
    else:
        print("  Verdict: Spurious model failed as expected. Triggering structural recovery.")

    # [PHASE 3] Structural Recovery
    print("\n[Phase 3] Structural Recovery on Shifted Data")
    
    # Force BFS for structural search to find the 5-node invariant
    agent._strategy_order = ["bfs"]
    # We'll use a direct call to agent._try_bfs to get the path for robust audit
    path3 = agent._try_bfs(t2_in, t2_out, beam_width=2000, max_depth=6)
    print(f"  Debug: path3 is {path3}")
    success3 = path3 is not None
    
    if success3:
        path_ids = [n.id for n in path3]
        uses_z1_3 = any("VAR_Z1" in pid for pid in path_ids)
        uses_square = any("OP_SQUARE" in pid for pid in path_ids)
        
        t3_code = agent.renderer.render(agent._compose_sequence(path3))
        
        print(f"  Recovery Solution found!")
        print(f"  Code Audit: uses Z1: {uses_z1_3}, uses SQUARED(t): {uses_square}")
        
        if not uses_z1_3 and uses_square:
            print("  SUCCESS: Agent abandoned spurious correlation and identified invariant structure.")
        else:
            print("  FAILURE: Agent still stuck on spurious features or failed to find structure.")
    else:
        print("  FAILED structural recovery.")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    # Validation logic (Robust): 
    recovery_valid = False
    if success3:
        path_ids = [n.id for n in path3]
        recovery_valid = not any("VAR_Z1" in pid for pid in path_ids) and any("OP_SQUARE" in pid for pid in path_ids)
    
    if success1 and recovery_valid:
        print("[VERDICT] SP97 VALIDATED: HPM proved capable of escaping spurious correlations via structural recovery.")
    else:
        print("[VERDICT] SP97 FAILED: Agent unable to distinguish structure from statistics.")
    print("=" * 80)

if __name__ == "__main__":
    main()
