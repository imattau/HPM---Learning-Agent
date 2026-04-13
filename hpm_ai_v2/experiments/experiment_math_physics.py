"""
experiment_math_physics.py — Collaborative Math-Physics Problem Solving

Demonstrates social collaboration between specialized agents (Math and Physics).
Validates:
- Social exchange of specialized macros (HPM §9.5)
- Sequential composition of cross-domain macros
- Pluggable probabilistic models (GMM validation)
"""
from __future__ import annotations

import sys
import time
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure we can import from project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.probabilistic_models import FlatGaussianModel, GaussianMixtureModel
from hpm_ai_v2.agents.agents import SocialAnalogicalAgent
from hpm_ai_v2.agents.mixins.social import SocialForest
from hpm_ai_v2.domains.math_physics_domain import MathPhysicsDomainConfig

# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

# Math tasks
# solve_linear(a, b) -> -b/a
MATH_TASKS = [
    ("math_solve_linear", "scalar", [(2, 4), (3, -6), (5, 10)], [-2.0, 2.0, -2.0]),
]

# Physics tasks
# force_to_accel(F, m, t) -> (a, t) where a = F/m
# distance_from_accel(a, t) -> 0.5 * a * t**2
PHYSICS_TASKS = [
    ("phys_force_to_accel", "tuple", [(10, 2, 3), (20, 4, 2)], [(5.0, 3.0), (5.0, 2.0)]),
    ("phys_distance_from_accel", "scalar", [(5.0, 2.0), (3.0, 4.0)], [10.0, 24.0]),
]

# Compound task
# distance_from_force_mass_time(F, m, t) -> 0.5 * (F/m) * t**2
COMPOUND_TASK = ("comp_dist_from_force", "scalar", [(10, 2, 3), (20, 4, 2)], [22.5, 10.0])


# ---------------------------------------------------------------------------
# Macro Seeding (Simulation of L2 learning)
# ---------------------------------------------------------------------------

def seed_math_macros(agent: SocialAnalogicalAgent):
    # solve_linear(a, b) -> -b/a
    code = textwrap.dedent("""
        a, b = inp
        return -b / a
    """).strip()
    node = HFN(mu=np.zeros(agent.m_dim), sigma=np.ones(agent.m_dim), id="macro_solve_linear")
    node._code = code
    node.relation_type = "macro"
    agent.patterns["solve_linear"] = node
    agent.observer.register(node, protected=False, initial_weight=1.0)

def seed_physics_macros(agent: SocialAnalogicalAgent):
    # force_to_accel(F, m, t) -> (F/m, t)
    code_f2a = textwrap.dedent("""
        F, m, t = inp
        return (F / m, t)
    """).strip()
    node_f2a = HFN(mu=np.zeros(agent.m_dim), sigma=np.ones(agent.m_dim), id="macro_force_to_accel")
    node_f2a._code = code_f2a
    node_f2a.relation_type = "macro"
    agent.patterns["force_to_accel"] = node_f2a
    agent.observer.register(node_f2a, protected=False, initial_weight=1.0)

    # distance_from_accel(a, t) -> 0.5 * a * t**2
    code_d = textwrap.dedent("""
        a, t = inp
        return 0.5 * a * t**2
    """).strip()
    node_d = HFN(mu=np.zeros(agent.m_dim), sigma=np.ones(agent.m_dim), id="macro_distance_from_accel")
    node_d._code = code_d
    node_d.relation_type = "macro"
    agent.patterns["distance_from_accel"] = node_d
    agent.observer.register(node_d, protected=False, initial_weight=1.0)


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("Collaborative Math-Physics Problem Solving")
    print("HPM §9.5 & §9.7 + Pluggable Probabilistic Models")
    print("=" * 70 + "\n")

    shared_dir = Path(tempfile.mkdtemp(prefix="math_phys_shared_"))
    config = MathPhysicsDomainConfig()
    shared_forest = SocialForest(D=config.m_dim, cold_dir=shared_dir)
    
    math_agent = SocialAnalogicalAgent(agent_id="MathAgent", config=config, social_forest=shared_forest)
    phys_agent = SocialAnalogicalAgent(agent_id="PhysicsAgent", config=config, social_forest=shared_forest)
    
    successes = []

    # -----------------------------------------------------------------------
    # Phase 1: Individual Specialisation
    # -----------------------------------------------------------------------
    print("PHASE 1: Individual Specialisation...")
    seed_math_macros(math_agent)
    seed_physics_macros(phys_agent)
    print("  MathAgent acquired 'solve_linear'")
    print("  PhysicsAgent acquired 'force_to_accel' and 'distance_from_accel'")

    # Verify accuracy
    for task_id, _, inputs, expected in MATH_TASKS:
        code = math_agent.patterns["solve_linear"]._code
        results, _ = math_agent.executor.run_batch(code, inputs)
        if results == expected:
            print(f"  [OK] MathAgent verified '{task_id}'")
        else:
            print(f"  [FAIL] MathAgent failed '{task_id}': got {results}, expected {expected}")

    # -----------------------------------------------------------------------
    # Phase 2: Social Exchange
    # -----------------------------------------------------------------------
    print("\nPHASE 2: Social Exchange...")
    math_agent.exchange_patterns()
    phys_agent.exchange_patterns()
    
    print(f"  MathAgent patterns: {list(math_agent.patterns.keys())}")
    print(f"  MathAgent social memory: {list(math_agent._social_memory.keys())}")
    print(f"  PhysicsAgent patterns: {list(phys_agent.patterns.keys())}")
    print(f"  PhysicsAgent social memory: {list(phys_agent._social_memory.keys())}")
    
    if "macro_force_to_accel" in math_agent._social_memory and "macro_solve_linear" in phys_agent._social_memory:
        print("  [SUCCESS] Social exchange complete")
        successes.append("Phase 2: Social exchange")
    else:
        print("  [FAIL] Social exchange incomplete")

    # -----------------------------------------------------------------------
    # Phase 3: Collaborative Problem Solving
    # -----------------------------------------------------------------------
    print("\nPHASE 3: Collaborative Problem Solving (Compound Task)...")
    task_id, goal_type, inputs, expected = COMPOUND_TASK
    print(f"  Task: {task_id} (F, m, t) -> distance")
    
    # Physics agent tries to solve it. It has f2a and d_from_a.
    success, code, strategy = phys_agent.solve(inputs, expected, goal_type=goal_type, task_id=task_id)
    if success:
        print(f"  [SUCCESS] PhysicsAgent solved '{task_id}' using strategy: {strategy}")
        # print(f"  Code:\n{textwrap.indent(code, '    ')}")
        successes.append("Phase 3: Collaborative solution")
    else:
        print(f"  [FAIL] PhysicsAgent could not solve '{task_id}'")

    # -----------------------------------------------------------------------
    # PHASE 4: Parameter Learning Validation (GMM)...
    # -----------------------------------------------------------------------
    print("\nPHASE 4: Parameter Learning Validation (GMM)...")
    # quadratic_roots(1, -3, 2) -> [1, 2]
    # We want a single node to learn both roots as modes.
    # Initialize components closer to the data to avoid one component capturing both.
    comp1 = FlatGaussianModel(mu=np.array([0.5]), sigma=np.array([0.2]), use_diag=True)
    comp2 = FlatGaussianModel(mu=np.array([2.5]), sigma=np.array([0.2]), use_diag=True)
    gmm = GaussianMixtureModel(components=[comp1, comp2])
    root_node = HFN(mu=np.array([1.5]), sigma=np.array([1.0]), prob_model=gmm, id="quad_roots", use_diag=True)

    # Train on both roots
    observations = [np.array([1.0]), np.array([2.0])] * 20
    for obs in observations:
        root_node.update(obs, weight=1.0, learning_rate=0.1)

    m1, m2 = gmm.components[0].mu[0], gmm.components[1].mu[0]
    print(f"  Trained GMM means: {m1:.2f}, {m2:.2f}")
    if (abs(m1 - 1.0) < 0.2 and abs(m2 - 2.0) < 0.2) or (abs(m1 - 2.0) < 0.2 and abs(m2 - 1.0) < 0.2):
        print("  [SUCCESS] GMM converged to both roots")
        successes.append("Phase 4: GMM parameter learning")
    else:
        print("  [FAIL] GMM did not converge correctly")


    # -----------------------------------------------------------------------
    # Final Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {len(successes)}/3 success criteria met.")
    if len(successes) == 3:
        print("[SUCCESS] Collaborative Math-Physics experiment passed!")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
