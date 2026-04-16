import numpy as np
import random
import sys
import tempfile
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l2_macro import L2MacroMixin

# Physics Domain Imports
from hpm_ai_v2.domains.projectile_domain import ProjectileDomainConfig
from hpm_ai_v2.domains.projectile_renderer import ProjectileRenderer
from hpm_ai_v2.utils.oracle.projectile_oracle import ProjectileOracle
from hpm_ai_v2.utils.oracle.base import CountingOracle

# -------------------------------------------------------------------
# Experiment: SP98 (Preferential Structure Selection) - STRONG FORM
# -------------------------------------------------------------------

def generate_preference_data(g=9.8, distortion=0.0, noise_sigma=0.01, n_points=10):
    inputs = []
    outputs = []
    for _ in range(n_points):
        t = random.uniform(0.1, 5.0)
        y_clean = 0.5 * g * (t ** 2)
        z1 = y_clean + distortion * np.sin(t)
        y_noisy = y_clean + np.random.normal(0, noise_sigma)
        # Input: [theta, v0, t, g, z1, ...] + padding
        inp = np.array([0.0, 0.0, t, g, z1] + [0.0]*15)
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

class PreferentialAgent(L2MacroMixin, BaseHFNAgent):
    pass

def run_seed(seed: int, parity: bool = True):
    print(f"\n--- [Seed {seed}] {'(Accuracy Parity)' if parity else ''} ---")
    random.seed(seed)
    np.random.seed(seed)

    config = ProjectileDomainConfig(s_dim=20)
    renderer = ProjectileRenderer(config)
    oracle = ProjectileOracle(config)
    
    cold_dir = Path(tempfile.mkdtemp(prefix=f"sp98_{seed}_"))
    forest = TieredForest(D=config.m_dim, cold_dir=cold_dir / "forest")
    
    agent = PreferentialAgent(
        config=config, renderer=renderer, forest=forest,
        n_workers=8, tolerance=0.15
    )
    agent.oracle = oracle
    agent.counting_oracle = CountingOracle(oracle)
    
    # Register strategies
    agent.add_strategy("bfs", agent._try_bfs)
    agent.add_strategy("greedy", agent._try_greedy_chain)

    # [PHASE 0] Pre-training: Register a Physical Prior
    print(f"  [Phase 0] Pre-training: Learning Full Vertical Drop macro")
    physics_concepts = ["VAR_T", "VAR_G", "OP_CONST_05", "OP_MUL", "OP_SQUARE"]
    agent._candidate_ops = create_primitives(agent, config, physics_concepts)
    
    t0_in, t0_out = generate_preference_data(distortion=0.0, noise_sigma=0.01)
    success0, code0, _ = agent.solve(t0_in, t0_out, goal_type="map", task_id="motif_vertical_drop", beam_width=1000, max_depth=6)
    
    if success0:
        print(f"    Macro vertical_drop registered and institutionalised.")
        motif_node = agent.patterns["motif_vertical_drop"]
        # Ensure observer state is initialized
        agent.observer.register(motif_node, protected=True)
        # mu layout in observer state: [0]=weight
        agent.observer._set_state_field(motif_node.id, 0, 100.0)
        
        all_concepts = ["VAR_T", "VAR_G", "VAR_Z1", "OP_CONST_05", "OP_MUL", "OP_SQUARE"]
        agent._candidate_ops = create_primitives(agent, config, all_concepts)
        agent._candidate_ops.append(motif_node)
    else:
        print("    FAILED motif pre-training.")
        return None

    # [PHASE 1] Task 1 Discovery
    print(f"  [Phase 1] Discovery on Ambiguous Data")
    if parity:
        t1_in, t1_out = generate_preference_data(distortion=0.0, noise_sigma=0.1)
        for inp in t1_in:
            # Z1 is index 4
            y_clean = inp[4] 
            inp[4] = y_clean + np.random.normal(0, 0.1)
    else:
        t1_in, t1_out = generate_preference_data(distortion=0.25, noise_sigma=0.05)
    
    path1 = agent._try_bfs(t1_in, t1_out, beam_width=2000, max_depth=6)
    
    if path1:
        path_ids = [n.id for n in path1]
        uses_z1 = any("VAR_Z1" in pid for pid in path_ids)
        uses_invariant = any("motif_vertical_drop" in pid or "OP_SQUARE" in pid for pid in path_ids)
        print(f"    Solution: uses Z1: {uses_z1}, uses Invariant: {uses_invariant}")
        
        macro_node = agent._compose_sequence(path1)
        macro_node.id = "macro_task1"
        agent.observer.register(macro_node, protected=True)
        # Reinforce successful discovery
        agent.observer._set_state_field(macro_node.id, 0, 50.0)
        agent._candidate_ops.append(macro_node)
        
        print(f"  [Phase 3] Task 2: Scaled Physics (y = 2.0 * g * t²)")
        t2_in, t2_out = generate_preference_data(g=19.6, distortion=0.25, noise_sigma=0.05)
        start_calls = agent.counting_oracle.call_count
        path2 = agent._try_bfs(t2_in, t2_out, beam_width=2000, max_depth=6)
        calls = agent.counting_oracle.call_count - start_calls
        
        reused = False
        if path2:
            reused = any("macro_task1" in n.id for n in path2)
            print(f"    Task 2 solved in {calls} oracle calls. Reused Task 1: {reused}")
            
        return {"chosen": "invariant" if uses_invariant and not uses_z1 else "shortcut", "reused": reused}
    else:
        print("    FAILED task 1 discovery.")
        return None

def main():
    print("=" * 80)
    print("SP98: Preferential Structure Selection (Strong Form: Accuracy Parity)")
    print("=" * 80)

    results = []
    for s in range(42, 47): # 5 seeds
        res = run_seed(s, parity=True)
        if res:
            results.append(res)
            
    n_invariant = sum(1 for r in results if r["chosen"] == "invariant")
    n_reused = sum(1 for r in results if r["reused"])
    
    print("\n" + "=" * 80)
    print(f"Summary: Invariant Selected in {n_invariant}/{len(results)} seeds")
    print(f"Summary: Transfer Reused in {n_reused}/{len(results)} seeds")
    
    if n_invariant >= 3:
        print("[VERDICT] SP98 VALIDATED: HPM prefers invariant under Accuracy Parity.")
    else:
        print("[VERDICT] SP98 FAILED: Agent stuck on shortcuts.")
    print("=" * 80)

if __name__ == "__main__":
    main()
