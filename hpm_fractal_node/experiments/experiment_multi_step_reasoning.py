"""
SP34: Experiment 10 — Multi-Step Internal Reasoning (Chain-of-Thought)

Validates sustained, stateful internal reasoning: A → B → C → D.
Setup:
1. Forest populated with 'Atomic' rules (increment individual dimensions).
2. Goal: Reach a target state requiring multiple sequential steps.
3. Mechanism: Iterative goal-conditioned retrieval and state update.

Metrics:
- Convergence: Distance to target over steps.
- Stability: Detection of cycles/oscillations.
- Step Count: Efficiency of the internal reasoning chain.
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever
from hfn.evaluator import Evaluator
from hfn.decoder import Decoder

def generate_atomic_forest(dim=10):
    """Creates a forest where each rule increments a single dimension."""
    forest = Forest()
    rules = []
    
    # Manifold: [Input(dim), Delta(dim)]
    # We use 2*dim dimensions total
    MANIFOLD_DIM = 2 * dim
    
    for i in range(dim):
        mu = np.zeros(MANIFOLD_DIM)
        # Context (Input): Vague (sigma=10), so it applies everywhere
        # Delta: Precise (sigma=0.0001), increments dimension i
        mu[dim + i] = 1.0 
        
        sig = np.ones(MANIFOLD_DIM) * 10.0 # Vague context
        sig[dim:] = 0.0001 # Precise transformation
        
        rule = HFN(mu=mu, sigma=sig, id=f"inc_dim_{i}", use_diag=True)
        forest.register(rule)
        rules.append(rule)
        
    return forest, rules

def run_experiment():
    print("--- SP34: Experiment 10 — Multi-Step Internal Reasoning ---\n")
    
    DIM = 10
    M_DIM = 2 * DIM
    forest, rules = generate_atomic_forest(DIM)
    
    # 1. Start State A and Target State D
    A = np.zeros(DIM)
    D = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # This requires 4 steps: inc_dim_0 twice, inc_dim_1 twice.
    
    # 2. Setup Goal-Conditioned Agent
    # Target slice is the Delta (DIM:2*DIM)
    retriever = GoalConditionedRetriever(forest, target_slice=slice(DIM, M_DIM), target_weight=100.0)
    
    # 3. Multi-Step Reasoning Loop
    current_state = A.copy()
    max_steps = 10
    history = [current_state.copy()]
    
    print(f"Initial State: {A}")
    print(f"Target State:  {D}\n")
    print(f"{'Step':<6} | {'Rule Selected':<15} | {'Dist to Target':<15} | {'State'}")
    print("-" * 70)
    
    success = False
    for step in range(1, max_steps + 1):
        target_delta = D - current_state
        dist = np.linalg.norm(target_delta)
        
        if dist < 0.1:
            success = True
            break
            
        # QUERY: [Current_Input, Desired_Delta]
        query_mu = np.zeros(M_DIM)
        query_mu[:DIM] = current_state
        query_mu[DIM:] = target_delta
        query_node = HFN(mu=query_mu, sigma=np.ones(M_DIM), id=f"query_{step}", use_diag=True)
        
        # RETRIEVE
        candidates = retriever.retrieve(query_node, k=1)
        if not candidates:
            print(f"  [ERROR] Step {step}: No rule found.")
            break
            
        best_rule = candidates[0]
        
        # EXECUTE (Update internal state)
        current_state += best_rule.mu[DIM:]
        history.append(current_state.copy())
        
        print(f"{step:<6} | {best_rule.id:<15} | {dist:<15.4f} | {current_state}")
        
        # Stability Check: Detect Oscillation
        # (Compare current state to history excluding last step)
        for i, prev in enumerate(history[:-1]):
            if np.linalg.norm(current_state - prev) < 0.01:
                print(f"\n[FAIL] Oscillation detected! Current state matches Step {i}.")
                return

    # 4. Final Report
    print("-" * 70)
    if success:
        print(f"\n[SUCCESS] Goal reached in {step-1} steps.")
        print("Internal stateful reasoning sustained: the HFN acted as a robust, composable operator.")
    else:
        print(f"\n[FAIL] Failed to reach goal within budget ({max_steps} steps).")
        print(f"Final Dist: {np.linalg.norm(D - current_state):.4f}")

if __name__ == "__main__":
    run_experiment()
