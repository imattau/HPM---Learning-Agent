"""
SP100: High-Dimensional Structure vs Correlation (Scaling Test)
"""
import numpy as np
import time
from typing import List, Any, Tuple, Optional
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l2_macro import L2MacroMixin
from hpm_ai_v2.domains.highdim_domain import HighDimDomainConfig
from hpm_ai_v2.domains.highdim_renderer import HighDimRenderer
from hpm_ai_v2.utils.oracle.highdim_oracle import HighDimOracle

def generate_data(D: int, n_samples: int, a: float, b: float, noise: float = 0.05, is_transfer: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate y = a * t**2 + b * sin(theta) + noise.
    Input vector is D-dimensional with various spurious features.
    
    If is_transfer is True, the shortcut (X[:, 2]) is broken (uncorrelated).
    """
    t = np.random.uniform(0.1, 2.0, n_samples)
    theta = np.random.uniform(0, np.pi, n_samples)
    y = a * t**2 + b * np.sin(theta)
    y_std = np.std(y)
    y_noisy = y + np.random.normal(0, noise * y_std, n_samples)
    
    X = np.zeros((n_samples, D))
    X[:, 0] = t
    X[:, 1] = theta
    
    # Spurious shortcut
    if is_transfer:
        # In transfer, the shortcut is pure noise (Broken)
        X[:, 2] = np.random.uniform(0, 1, n_samples)
    else:
        # In training, the shortcut is highly correlated with y
        X[:, 2] = y_noisy + np.random.normal(0, 0.01, n_samples)
        
    # Correlated with t
    X[:, 3] = t + np.random.normal(0, 0.1, n_samples)
    # Non-linear distractors
    X[:, 4] = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    X[:, 5] = t**2 + np.random.normal(0, 0.1, n_samples)
    # Pure noise
    if D > 6:
        X[:, 6:] = np.random.uniform(0, 1, (n_samples, D - 6))
        
    return X, y_noisy

class HighDimAgent(L2MacroMixin, BaseHFNAgent):
    """
    Specialized agent for SP100.
    """
    def __init__(self, D: int):
        config = HighDimDomainConfig(D=D)
        renderer = HighDimRenderer(config)
        super().__init__(config=config, renderer=renderer)
        # Manually override oracle for state computation
        self.oracle = HighDimOracle(config)
        self.counting_oracle.oracle = self.oracle
        
        # Add strategies
        self.add_strategy("exact", self._try_exact)
        self.add_strategy("bfs", self._try_bfs)

        # Prime SELECT_i priors for correlation guidance
        for i in range(D):
            node_id = f"prior_rule_SELECT_{i}"
            node = self.forest.get(node_id)
            if node:
                # Prime mu to match correlation index in scientific part
                node.mu[20 + i] = 10.0
        
        # Candidate ops: SELECT(0..D-1) + base ops
        self._candidate_ops = []
        for c in config.concepts:
            node = self.forest.get(f"prior_rule_{c}")
            if node:
                self._candidate_ops.append(node)

def run_scaling_test(D: int, seed: int = 42) -> bool:
    """Run a full SP100 test for a given D."""
    np.random.seed(seed)
    print(f"\n--- [D={D} Scaling Test] ---")
    
    # 1. Setup Agent
    agent = HighDimAgent(D=D)
    agent.tolerance = 0.1
    agent.n_workers = 1
    
    # 2. Phase 1: Discovery (a=1.0, b=1.0)
    a_train, b_train = 1.0, 1.0
    X_train, y_train = generate_data(D, 50, a_train, b_train, is_transfer=False)
    inputs = [X_train[i] for i in range(50)]
    outputs = [y_train[i] for i in range(50)]
    
    start_time = time.time()
    # BFS: Beam width 50, max depth 6
    # We use a smaller beam for speed, guided by correlations
    success, code, strat, path = agent.solve(inputs, outputs, goal_type="map", beam_width=50, max_depth=6)
    duration = time.time() - start_time
    
    if not success or not path:
        print(f"  [Discovery] FAILED after {duration:.2f}s")
        return False
    
    print(f"  [Discovery] SUCCESS in {duration:.2f}s using {strat}. Path: {[n.id for n in path]}")
    
    # 3. Phase 2: Audit for structure recovery
    spurious_used = any(f"inp[{i}]" in code for i in range(2, D))
    true_used = "inp[0]" in code and "inp[1]" in code
    recovery = true_used and not spurious_used
    print(f"  [Audit] Recovery: {recovery} (True: {true_used}, Spurious: {spurious_used})")
    
    # Register the found macro for Phase 3
    macro_id = f"macro_structure_D{D}"
    composed = agent._compose_sequence(path)
    if composed:
        composed.id = macro_id
        agent.patterns[macro_id] = composed
        agent.observer.register(composed)
        agent._candidate_ops.append(composed)
    
    # 4. Phase 3: Transfer (a=2.0, b=0.5)
    print(f"  [Transfer] Test with new coefficients (a=2.0, b=0.5) and BROKEN shortcut...")
    a_test, b_test = 2.0, 0.5
    X_test, y_test = generate_data(D, 20, a_test, b_test, is_transfer=True)
    inputs_t = [X_test[i] for i in range(20)]
    outputs_t = [y_test[i] for i in range(20)]
    
    # Clear discovery strategies to force macro reuse or re-discovery
    # Actually, keep them but prioritize exact
    start_t = time.time()
    success_t, code_t, strat_t, path_t = agent.solve(inputs_t, outputs_t, goal_type="map", beam_width=50)
    duration_t = time.time() - start_t
    
    if success_t:
        is_reused = any(node.id == macro_id for node in path_t) or (len(path_t) == 1 and path_t[0].id == macro_id)
        print(f"  [Transfer] SUCCESS in {duration_t:.2f}s via {strat_t}. Path length: {len(path_t)}. Macro Reused: {is_reused}")
    else:
        print(f"  [Transfer] FAILED.")
        
    return recovery

def main():
    print("================================================================================")
    print("SP100: High-Dimensional Structure vs Correlation")
    print("================================================================================")
    
    # Test D=10, 50, 100
    dims = [10, 50, 100]
    results = {}
    for D in dims:
        recovered = run_scaling_test(D)
        results[D] = recovered
        
    print("\n================================================================================")
    print("Summary:")
    total_recovered = sum(1 for v in results.values() if v)
    print(f"  Total Recovery: {total_recovered}/{len(dims)}")
    for D, rec in results.items():
        print(f"  D={D:3}: Recovery={rec}")
    print("================================================================================")

if __name__ == "__main__":
    main()
