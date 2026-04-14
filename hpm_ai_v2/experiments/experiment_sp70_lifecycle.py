import numpy as np
import time
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from hpm_ai_v2.domains.math_physics_domain import MathPhysicsDomainConfig
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hfn.hfn import HFN

def run_lifecycle_experiment():
    print("="*70)
    print("SP70: Agent Lifecycle and Persistence Validation")
    print("="*70)

    # 1. Setup mock domain
    config = MathPhysicsDomainConfig()
    
    # Create a temporary directory for the experiment
    temp_dir = Path(tempfile.mkdtemp(prefix="sp70_"))
    cold_dir = temp_dir / "agent"

    # ------------------------------------------------------------------
    # Phase 1: Auto-Observation Loop
    # ------------------------------------------------------------------
    print("\n[Phase 1] Auto-Observation Loop")
    # auto_observe_frequency=2 means every 2nd call to _maybe_auto_observe triggers observer.observe()
    agent = BaseHFNAgent(
        config=config,
        cold_dir=str(cold_dir),
        auto_observe_frequency=2,
        replay_buffer_size=5
    )
    
    initial_observations = agent.observer._observation_step
    
    # Observe 3 examples manually via observe_example()
    # Each call to observe_example calls observer.observe(x) AND _maybe_auto_observe()
    # 1st: observe(x1), counter=1
    # 2nd: observe(x2), counter=2 -> observe(sample from [x1, x2]), counter=0
    # 3rd: observe(x3), counter=1
    # Expected total observations = 3 manual + 1 auto = 4
    
    for i in range(3):
        x = np.random.randn(config.m_dim)
        agent.observe_example(x)
        
    print(f"  [OK] Total observations: {agent.observer._observation_step} (expected {initial_observations + 4})")
    assert agent.observer._observation_step == initial_observations + 4
    assert len(agent._replay_buffer) == 3

    # ------------------------------------------------------------------
    # Phase 2: Node Persistence (Auto-Save)
    # ------------------------------------------------------------------
    print("\n[Phase 2] Node Persistence (Auto-Save)")
    agent.auto_save_frequency = 2
    agent._solve_counter = 0
    
    # Mock a successful strategy
    def mock_strat(inputs, outputs):
        # Return a simple HFN path
        return [HFN(mu=np.zeros(config.m_dim), sigma=np.ones(config.m_dim), id="mock_node")]
        
    agent.add_strategy("mock", mock_strat)
    
    # Ensure agent_state.pkl doesn't exist yet
    save_file = cold_dir / "agent_state.pkl"
    if save_file.exists():
        save_file.unlink()

    # We need to make sure solve() succeeds. 
    # Let's mock _check_outputs to always return True for this test
    agent._check_outputs = lambda r, e: True
    # And mock executor.run_batch to return dummy results
    agent.executor.run_batch = lambda c, i: ([1]*len(i), [None]*len(i))
    # And mock oracle.compute_state to return high success
    agent.oracle.compute_state = lambda r, e, c=None: (1.0, 0.0, 0.0)

    # 1st solve (success)
    print("  Triggering 1st successful solve...")
    agent.solve([1], [1], task_id="t1")
    # _solve_counter = 1
    assert not save_file.exists(), "Save file should not exist after 1st solve"
    
    # 2nd solve (success)
    print("  Triggering 2nd successful solve...")
    agent.solve([1], [1], task_id="t2")
    # _solve_counter = 2 -> _maybe_save_state calls save_state(), _solve_counter resets to 0
    assert save_file.exists(), "Save file should exist after 2nd solve"
    print("  [OK] Auto-save triggered after 2 successful solves.")

    # ------------------------------------------------------------------
    # Phase 3: Auto-Observation in Solve Loop
    # ------------------------------------------------------------------
    print("\n[Phase 3] Auto-Observation in Solve Loop")
    agent.auto_observe_frequency = 1 # observe on EVERY call
    agent._observe_counter = 0
    obs_before = agent.observer._observation_step
    
    print("  Triggering solve with auto-observation enabled...")
    agent.solve([1], [1], task_id="t3")
    # solve calls _maybe_auto_observe(flat_input)
    # Since freq=1, it should trigger observer.observe() immediately.
    assert agent.observer._observation_step == obs_before + 1
    print(f"  [OK] Observation triggered during solve loop. Total: {agent.observer._observation_step}")

    print("\n" + "="*70)
    print("SUMMARY: 3/3 phases passed")
    print("[SUCCESS] SP70 – Agent lifecycle and persistence validated!")
    print("="*70)
    
    # Cleanup
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_lifecycle_experiment()
