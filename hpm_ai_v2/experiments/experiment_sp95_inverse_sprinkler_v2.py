#!/usr/bin/env python3
"""
SP95 v2: True Invariance Discovery (Inverse Sprinkler).

Tests HPM's ability to discover physical invariants (Q -> Q^2) from data.
Learns on Inflow, validates on Outflow without re-solving.
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
# 1. Data Generation (V2: Signed Q)
# ----------------------------------------------------------------------
def get_data():
    """
    Returns training, validation, and test examples.
    Vector: [N, L, theta, Q, rho, 0...]
    """
    # Training: Force discovery of sign(sin(theta) * Q**2)
    train_in = [
        np.array([0.2, 0.5, 0.5, -1.0, 1.0] + [0.0]*15), # theta=0.5, Q=-1, Output=+1
        np.array([0.2, 0.5, 1.5, -1.0, 1.0] + [0.0]*15), # theta=1.5, Q=-1, Output=-1
        np.array([0.2, 0.5, 0.5,  0.0, 1.0] + [0.0]*15), # theta=0.5, Q=0,  Output=0
        np.array([0.2, 0.5, 0.5, -2.0, 1.0] + [0.0]*15), # theta=0.5, Q=-2, Output=1
        np.array([0.2, 0.5, 0.5, -3.0, 1.0] + [0.0]*15), # theta=0.5, Q=-3, Output=1
    ]
    train_out = [1.0, -1.0, 0.0, 1.0, 1.0]
    
    # Validation: Outflow (Q = +1.0).
    val_in = [
        np.array([0.2, 0.5, 0.5, 1.0, 1.0] + [0.0]*15),  # theta=0.5, Q=1, Output=+1
        np.array([0.2, 0.5, 1.5, 1.0, 1.0] + [0.0]*15),  # theta=1.5, Q=1, Output=-1
    ]
    val_out = [1.0, -1.0]
    
    # Test: Novel magnitude and geometry
    test_in = [np.array([0.3, 0.6, 0.5, -1.5, 1.2] + [0.0]*15)]
    test_out = [1.0]
    
    return (train_in, train_out), (val_in, val_out), (test_in, test_out)


# ----------------------------------------------------------------------
# 2. Primitive Creation
# ----------------------------------------------------------------------
def create_primitive_nodes(agent, config):
    """Register a primitive HFN node for each math concept."""
    primitives = []
    for i, concept in enumerate(config.concepts):
        mu = np.zeros(config.m_dim)
        mu[config.S_DIM + i] = 1.0
        node = HFN(mu=mu, sigma=np.ones(config.m_dim), id=f"prior_rule_{concept}", use_diag=True)
        agent.observer.register(node, protected=True)
        primitives.append(node)
    return primitives


# ----------------------------------------------------------------------
# 3. Main Experiment
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("SP95 v2: True Invariance Discovery (Inverse Sprinkler)")
    print("=" * 80)

    config = FluidDomainConfig(s_dim=20)
    renderer = FluidRenderer(config)
    oracle = FluidOracle(config)

    cold_dir = Path(tempfile.mkdtemp(prefix="sp95_v2_"))
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

    # Register strategies
    agent.add_strategy("exact", agent._try_exact, position=0)
    agent.add_strategy("bfs", agent._try_bfs, position=1)

    # Inject primitives (Math ops, not physics blocks)
    agent._candidate_ops = create_primitive_nodes(agent, config)

    # Data
    (train_in, train_out), (val_in, val_out), (test_in, test_out) = get_data()

    # ------------------------------------------------------------------
    # Phase 1: Discovery (Inflow)
    # ------------------------------------------------------------------
    print("\n[Phase 1] Discovery (Learning from single Inflow case: Q=-1.0)")
    # We allow BFS to find the minimal mapping
    success, code, strategy = agent.solve(train_in, train_out, goal_type="map", task_id="invariance_discovery")
    if not success:
        print("  Failed to find macro in Phase 1. BFS depth might be too shallow.")
        return
    print(f"  Macro discovered via '{strategy}'. Code:\n{code}")

    # ------------------------------------------------------------------
    # Phase 2: Strict Validation (Outflow - FROZEN MACRO)
    # ------------------------------------------------------------------
    print("\n[Phase 2] Strict Validation (Testing on Outflow: Q=+1.0)")
    print("  Note: No re-solving allowed. Running Phase 1 macro directly.")
    
    results, errors = agent.executor.run_batch(code, val_in)
    res = results[0]
    if res == val_out[0]:
        print(f"  [SUCCESS] Invariance Validated! Output: {res}")
    else:
        print(f"  [FAIL] Invariance Failed. Output: {res}, Expected: {val_out[0]}")
        print("  The discovered rule was likely sign-dependent.")

    # ------------------------------------------------------------------
    # Phase 3: True Zero-Shot (Novel Geometry/Magnitude)
    # ------------------------------------------------------------------
    print("\n[Phase 3] True Zero-Shot (Novel Geometry & Flow magnitude)")
    results, _ = agent.executor.run_batch(code, test_in)
    res = results[0]
    if res == test_out[0]:
        print(f"  [SUCCESS] Zero-Shot Generalization! Output: {res}")
    else:
        print(f"  [FAIL] Zero-Shot Failed. Output: {res}, Expected: {test_out[0]}")

    # ------------------------------------------------------------------
    # Phase 4: Structure Audit
    # ------------------------------------------------------------------
    print("\n[Phase 4] Structure Audit")
    has_square = ('res**2' in code) or (code.count('* Q') >= 2)
    has_sin = 'np.sin' in code
    has_sign = 'np.sign' in code
    print(f"  Discovered Q**2 invariance: {has_square}")
    print(f"  Discovered Sign abstraction: {has_sign}")

    if has_square and has_sin:
        print("\n  [VERDICT] HPM successfully discovered the hidden physical invariant.")
    else:
        print("\n  [VERDICT] HPM found a valid but less general mapping.")

    print("\n" + "=" * 80)
    print("[FINISH] SP95 v2 Experiment Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
