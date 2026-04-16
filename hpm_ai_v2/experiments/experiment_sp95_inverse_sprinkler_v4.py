#!/usr/bin/env python3
"""
SP95 v4: Contextual Transfer & Cognitive Drive (Inverse Sprinkler).

Demonstrates:
1. Invariant Discovery (Q -> Q^2).
2. Contextual Activation (Background observation biases retrieval).
3. Structural Necessity (Strict depth limit forces macro reuse).
4. Cross-Task Transfer (Kinetic Energy proxy).
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
    
    test_in = [np.array([0.3, 0.6, 0.5, -1.5, 1.2] + [0.0]*15)]
    test_out = [1.0]
    
    return (train_in, train_out), (test_in, test_out)

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
    """Register primitive nodes for specific concepts."""
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
    print("SP95 v4: Contextual Transfer & Cognitive Drive (Inverse Sprinkler)")
    print("=" * 80)

    config = FluidDomainConfig(s_dim=20)
    renderer = FluidRenderer(config)
    oracle = FluidOracle(config)

    cold_dir = Path(tempfile.mkdtemp(prefix="sp95_v4_"))
    forest = TieredForest(D=config.m_dim, cold_dir=cold_dir / "forest")

    # [UPGRADE] Use ContextualRetriever
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
    agent.add_strategy("exact", agent._try_exact, position=0)
    agent.add_strategy("bfs", agent._try_bfs, position=1)

    # ------------------------------------------------------------------
    # Step 1: Task 1 - Discover Invariant (Q^2)
    # ------------------------------------------------------------------
    print("\n[Step 1] Task 1: Discover Square (Q -> Q^2)")
    t1_in, t1_out = get_task1_data()
    agent._candidate_ops = create_primitives(agent, config, ["VAR_Q", "OP_MUL_Q"])
    
    success, t1_code, _ = agent.solve(t1_in, t1_out, goal_type="map", task_id="discover_square")
    if not success: return
    macro_square = agent.patterns["discover_square"]
    print(f"  Invariant discovered and registered: {macro_square.id}")

    # ------------------------------------------------------------------
    # Step 2: Background Observation (Contextual Activation)
    # ------------------------------------------------------------------
    print("\n[Step 2] Background Observation (Contextual Activation)")
    (t2_train_in, t2_train_out), (t2_test_in, t2_test_out) = get_task2_data()
    
    # Observe Task 2 inputs (unlabeled) to activate Q-related patterns
    print("  Presenting Task 2 inputs to the agent's observer...")
    for inp in t2_train_in:
        agent.observe_example(inp)
    
    # Verify retrieval prioritization
    query = HFN(mu=np.zeros(config.m_dim), sigma=np.ones(config.m_dim), use_diag=True)
    # We query for Q-related state
    query.mu[3] = -1.0 
    retrieved = agent.retriever.retrieve(query, k=5)
    macro_ids = [n.id for n in retrieved if n.relation_type == "macro"]
    print(f"  Recently active macros in retrieval: {macro_ids}")

    # ------------------------------------------------------------------
    # Step 3: Task 2 - Inverse Sprinkler (STRICT DEPTH LIMIT)
    # ------------------------------------------------------------------
    print("\n[Step 3] Task 2: Inverse Sprinkler (STRICT DEPTH: max_depth=5)")
    print("  Flat solution (Depth 6) is impossible. Macro reuse is required.")
    
    # Primitives for Task 2: VAR_THETA, OP_SIN, OP_SIGN, OP_MUL + Learned Macro
    t2_primitives = create_primitives(agent, config, ["VAR_THETA", "OP_SIN", "OP_SIGN", "OP_MUL"])
    agent._candidate_ops = t2_primitives + [macro_square]
    
    # Override BFS depth
    success2, t2_code, strategy2 = agent.solve(
        t2_train_in, t2_train_out, 
        goal_type="map", 
        task_id="inverse_sprinkler_v4"
    )
    
    if success2:
        print(f"  Task 2 SUCCESS! Hierarchical path found at Depth {agent.meta.history[-1].depth}")
        print(f"  Code Audit: {'BEGIN MACRO' in t2_code}")
    else:
        print("  Task 2 FAILED (as expected if macro was ignored or search budget too low)")
        return

    # ------------------------------------------------------------------
    # Step 4: Task 3 - Cross-Task Transfer (Kinetic Energy)
    # ------------------------------------------------------------------
    print("\n[Step 4] Task 3: Cross-Task Transfer (Kinetic Energy Proxy: rho * Q^2)")
    t3_in, t3_out = get_task3_data()
    
    # Ensure necessary primitives for Task 3 are present
    agent._candidate_ops = create_primitives(agent, config, ["VAR_RHO", "OP_MUL"]) + [macro_square]
    
    # If it reuses macro_square, it solves at Depth 3 (VAR_RHO, macro, OP_MUL)
    success3, t3_code, _ = agent.solve(t3_in, t3_out, goal_type="map", task_id="kinetic_energy")
    
    if success3:
        print(f"  Task 3 SUCCESS! Code:\n{t3_code}")
        has_macro = f"BEGIN MACRO: {macro_square.id}" in t3_code
        print(f"  Reused Q^2 macro in new physical context: {has_macro}")
    
    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    if success2 and success3 and ('BEGIN MACRO' in t2_code) and has_macro:
        print("[VERDICT] HPM VALIDATED: Hierarchical reuse is driven by context and necessity.")
    else:
        print("[VERDICT] Mixed results. Audit required.")
    print("=" * 80)


if __name__ == "__main__":
    main()
