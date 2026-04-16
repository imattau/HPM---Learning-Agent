#!/usr/bin/env python3
"""
SP95 v3: Hierarchical Transfer (Inverse Sprinkler).

Demonstrates HPM's ability to:
1. Discover an invariant (Q -> Q^2) as a reusable macro.
2. Transfer that macro to solve a more complex physical law (Inverse Sprinkler).
3. Validate reuse via structure audit (no re-searching of Q*Q).
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
    # Vector: [N, L, theta, Q, rho, 0...]
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
    
    val_in = [
        np.array([0.2, 0.5, 0.5, 1.0, 1.0] + [0.0]*15),  # theta=0.5, Q=1, Output=+1
    ]
    val_out = [1.0]
    
    test_in = [np.array([0.3, 0.6, 0.5, -1.5, 1.2] + [0.0]*15)]
    test_out = [1.0]
    
    return (train_in, train_out), (val_in, val_out), (test_in, test_out)


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
    print("SP95 v3: Hierarchical Transfer (Inverse Sprinkler)")
    print("=" * 80)

    config = FluidDomainConfig(s_dim=20)
    renderer = FluidRenderer(config)
    oracle = FluidOracle(config)

    cold_dir = Path(tempfile.mkdtemp(prefix="sp95_v3_"))
    forest = TieredForest(D=config.m_dim, cold_dir=cold_dir / "forest")

    agent = BaseHFNAgent(
        config=config,
        renderer=renderer,
        forest=forest,
        retriever_type="geometric",
        use_hfn_forward_model=False,
        use_hfn_meta_controller=False,
    )
    agent.oracle = oracle
    agent.counting_oracle = CountingOracle(oracle)
    agent.add_strategy("exact", agent._try_exact, position=0)
    agent.add_strategy("bfs", agent._try_bfs, position=1)

    # ------------------------------------------------------------------
    # Task 1: Discover Square
    # ------------------------------------------------------------------
    print("\n[Step 1] Task 1: Discover Square (Q -> Q^2)")
    t1_in, t1_out = get_task1_data()
    # Primitives: VAR_Q, OP_MUL_Q
    agent._candidate_ops = create_primitives(agent, config, ["VAR_Q", "OP_MUL_Q"])
    
    success, t1_code, _ = agent.solve(t1_in, t1_out, goal_type="map", task_id="discover_square")
    if not success:
        print("  Failed Task 1. Exiting.")
        return
    
    # Retrieve the macro from the patterns dict
    macro_square = agent.patterns["discover_square"]
    print(f"  Task 1 SUCCESS! Learned macro: {macro_square.id}")
    print(f"  Code:\n{t1_code}")

    # ------------------------------------------------------------------
    # Step 2: Macro Promotion
    # ------------------------------------------------------------------
    print("\n[Step 2] Promoting 'macro_discover_square' to primitives.")
    # Primitives for Task 2: VAR_THETA, OP_SIN, OP_SIGN, OP_MUL + Learned Macro
    # We KEEP OP_MUL because it is needed to combine the sin(theta) and Q^2 terms.
    # But since macro_square already contains Q*Q, it's a shorter path.
    t2_primitives = create_primitives(agent, config, ["VAR_THETA", "OP_SIN", "OP_SIGN", "OP_MUL"])
    agent._candidate_ops = t2_primitives + [macro_square]

    # ------------------------------------------------------------------
    # Task 2: Inverse Sprinkler (Hierarchical Reuse)
    # ------------------------------------------------------------------
    print("\n[Step 3] Task 2: Inverse Sprinkler (Hierarchical Reuse)")
    (t2_train_in, t2_train_out), (t2_val_in, t2_val_out), (t2_test_in, t2_test_out) = get_task2_data()
    
    # We solve Task 2. HPM should find sign(sin(theta) * macro_square)
    success2, t2_code, strategy2 = agent.solve(t2_train_in, t2_train_out, goal_type="map", task_id="inverse_sprinkler_v3")
    if not success2:
        print("  Failed Task 2. Exiting.")
        return
    
    print(f"  Task 2 SUCCESS via '{strategy2}'! Code:\n{t2_code}")

    # ------------------------------------------------------------------
    # Phase 3: Zero-Shot Validation (Frozen Hierarchical Macro)
    # ------------------------------------------------------------------
    print("\n[Step 4] Zero-Shot Validation (Outflow & Novel Geometry)")
    
    # Validate Outflow
    results_v, _ = agent.executor.run_batch(t2_code, t2_val_in)
    if results_v[0] == t2_val_out[0]:
        print(f"  [SUCCESS] Outflow Validated! Output: {results_v[0]}")
    else:
        print(f"  [FAIL] Outflow Failed. Output: {results_v[0]}")

    # Zero-shot Test
    results_t, _ = agent.executor.run_batch(t2_code, t2_test_in)
    if results_t[0] == t2_test_out[0]:
        print(f"  [SUCCESS] Zero-Shot Test Validated! Output: {results_t[0]}")
    else:
        print(f"  [FAIL] Zero-Shot Test Failed. Output: {results_t[0]}")

    # ------------------------------------------------------------------
    # Phase 4: Structure Audit (Proof of Hierarchy)
    # ------------------------------------------------------------------
    print("\n[Step 5] Structure Audit (Proof of Hierarchy)")
    # We check if the rendered code explicitly contains the macro ID.
    macro_id_in_code = f"BEGIN MACRO: {macro_square.id}"
    has_macro = macro_id_in_code in t2_code
    
    print(f"  Macro '{macro_square.id}' found in Task 2 code: {has_macro}")
    
    rec = agent.meta.history[-1]
    print(f"  Task 2 Search Depth: {rec.depth}")
    
    if has_macro:
        print("\n  [VERDICT] HPM successfully demonstrated HIERARCHICAL TRANSFER.")
        print("  The agent reused the square invariant as a high-level primitive.")
    else:
        print("\n  [VERDICT] FAIL: The solution was found but did not use the macro.")

    print("\n" + "=" * 80)
    print("[FINISH] SP95 v3 Experiment Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
