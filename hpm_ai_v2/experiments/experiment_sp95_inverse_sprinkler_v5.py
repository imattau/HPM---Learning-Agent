#!/usr/bin/env python3
"""
SP95 v5: Retrieval-Driven Cognition (Inverse Sprinkler).

The Definitive Proof of HPM:
1. Discover Invariant (Q -> Q^2).
2. Contextual Activation (Background observation).
3. Retrieval-Driven Execution (Greedy walk with ZERO branching).
4. Cross-Domain Cognition (Kinetic Energy).
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.fluid_domain import FluidDomainConfig
from hpm_ai_v2.domains.fluid_renderer import FluidRenderer
from hpm_ai_v2.utils.oracle.fluid_oracle import FluidOracle
from hpm_ai_v2.utils.oracle.base import CountingOracle


# ----------------------------------------------------------------------
# 1. Data Generation
# ----------------------------------------------------------------------
def get_task1_data():
    """Task 1: Discover Square (Q -> Q^2)"""
    inputs = [
        np.array([0.0, 0.0, 0.0, -1.0, 1.0] + [0.0]*15),
        np.array([0.0, 0.0, 0.0,  2.0, 1.0] + [0.0]*15),
        np.array([0.0, 0.0, 0.0, -3.0, 1.0] + [0.0]*15),
    ]
    outputs = [1.0, 4.0, 9.0]
    return inputs, outputs

def get_task2_data():
    """Task 2: Inverse Sprinkler (sign(sin(theta) * Q^2))"""
    train_in = [
        np.array([0.2, 0.5, 0.5, -1.0, 1.0] + [0.0]*15), # theta=0.5, Q=-1, Output=+1
        np.array([0.2, 0.5, 1.5, -1.0, 1.0] + [0.0]*15), # theta=1.5, Q=-1, Output=-1
        np.array([0.2, 0.5, 0.5, -2.0, 1.0] + [0.0]*15), # theta=0.5, Q=-2, Output=1
        np.array([0.2, 0.5, 0.5,  0.0, 1.0] + [0.0]*15), # theta=0.5, Q=0,  Output=0
    ]
    train_out = [1.0, -1.0, 1.0, 0.0]
    return (train_in, train_out)

def get_task3_data():
    """Task 3: Kinetic Energy Proxy (rho * Q^2)"""
    inputs = [
        np.array([0.0, 0.0, 0.0, -1.0, 2.0] + [0.0]*15), # Q=-1, rho=2 -> Output=2.0
        np.array([0.0, 0.0, 0.0,  2.0, 0.5] + [0.0]*15), # Q=2,  rho=0.5 -> Output=2.0
    ]
    outputs = [2.0, 2.0]
    return inputs, outputs


# ----------------------------------------------------------------------
# 2. Primitive Creation
# ----------------------------------------------------------------------
def create_primitives(agent, config, concepts):
    """Register primitive nodes."""
    primitives = []
    for concept in concepts:
        mu = np.zeros(config.m_dim)
        mu[config.S_DIM + config.concept_idx[concept]] = 1.0
        node = HFN(mu=mu, sigma=np.ones(config.m_dim), id=f"prior_rule_{concept}", use_diag=True)
        agent.observer.register(node, protected=True)
        primitives.append(node)
    return primitives


# ----------------------------------------------------------------------
# 3. Main Experiment
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("SP95 v5: Retrieval-Driven Cognition (Inverse Sprinkler)")
    print("=" * 80)

    config = FluidDomainConfig(s_dim=20)
    renderer = FluidRenderer(config)
    oracle = FluidOracle(config)

    cold_dir = Path(tempfile.mkdtemp(prefix="sp95_v5_"))
    forest = TieredForest(D=config.m_dim, cold_dir=cold_dir / "forest")

    agent = BaseHFNAgent(
        config=config,
        renderer=renderer,
        forest=forest,
        retriever_type="contextual",
        use_hfn_forward_model=False,
        use_hfn_meta_controller=False,
    )
    agent.oracle = oracle
    agent.counting_oracle = CountingOracle(oracle)
    
    # Register strategies: Task 1 uses BFS, Task 2/3 use GREEDY
    agent.add_strategy("bfs", agent._try_bfs, position=0)
    agent.add_strategy("greedy", agent._try_greedy_chain, position=1)

    # ------------------------------------------------------------------
    # Step 1: Task 1 - Learn Invariant
    # ------------------------------------------------------------------
    print("\n[Step 1] Task 1: Discover Square (BFS)")
    t1_in, t1_out = get_task1_data()
    agent._candidate_ops = create_primitives(agent, config, ["VAR_Q", "OP_MUL_Q"])
    
    success, t1_code, _ = agent.solve(t1_in, t1_out, goal_type="map", task_id="discover_square")
    if not success: return
    macro_square = agent.patterns["discover_square"]
    print(f"  Invariant registered: {macro_square.id}")

    # ------------------------------------------------------------------
    # Step 2: Contextual Activation
    # ------------------------------------------------------------------
    print("\n[Step 2] Background Observation (Contextual Activation)")
    (t2_in, t2_out) = get_task2_data()
    for inp in t2_in:
        agent.observe_example(inp)
    
    # ------------------------------------------------------------------
    # Step 3: Task 2 - Inverse Sprinkler (GREEDY ONLY)
    # ------------------------------------------------------------------
    print("\n[Step 3] Task 2: Inverse Sprinkler (GREEDY Strategy ONLY)")
    print("  Zero branching (beam_width=1). Success requires top-ranked retrieval.")
    
    # Primitives for Task 2
    t2_primitives = create_primitives(agent, config, ["VAR_THETA", "OP_SIN", "OP_SIGN", "OP_MUL"])
    agent._candidate_ops = t2_primitives + [macro_square]
    
    # [FIX] Properly isolate greedy strategy
    agent._strategies = {"greedy": agent._try_greedy_chain}
    agent._strategy_order = ["greedy"] 
    
    success2, t2_code, strategy2 = agent.solve(t2_in, t2_out, goal_type="map", task_id="inverse_sprinkler_v5")
    
    if success2:
        print(f"  Task 2 SUCCESS via '{strategy2}'! Code Audit: {'macro_discover_square' in t2_code}")
    else:
        print("  Task 2 FAILED greedy walk. Search was likely still required.")
    
    # [Step 4] Task 3 - Kinetic Energy
    t3_in, t3_out = get_task3_data()
    agent._candidate_ops = create_primitives(agent, config, ["VAR_RHO", "OP_MUL"]) + [macro_square]
    
    success3, t3_code, strategy3 = agent.solve(t3_in, t3_out, goal_type="map", task_id="kinetic_energy_v5")
    
    if success3:
        print(f"  Task 3 SUCCESS via '{strategy3}'!")
        print(f"  Code Audit: {'macro_discover_square' in t3_code}")
    
    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    if success2 and success3 and ('macro_discover_square' in t2_code) and ('macro_discover_square' in t3_code):
        print("[VERDICT] HPM FULLY VALIDATED: Cognition is driven by hierarchy, not search.")
    else:
        print("[VERDICT] Greedy walk failed. Hierarchy exists but is not yet driving cognition.")
    print("=" * 80)


if __name__ == "__main__":
    main()
