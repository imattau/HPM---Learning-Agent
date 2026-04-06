"""
SP38: Experiment 14 — World Model Simulation (Imagination Test)

Validates if HFN can act as a generative world model by simulating future states.
Uses RELATIONAL ENCODING [State, Delta] to enable extrapolation beyond training data.
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.retriever import GoalConditionedRetriever
from hfn.evaluator import Evaluator

def generate_trajectory_data(dim=10, n_steps=20):
    """
    Creates linear movement data.
    Encoded as [Current_Pos, Velocity_Delta].
    """
    velocity = np.array([1.0, 0.5, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current_pos = np.zeros(dim)
    data = []
    
    for _ in range(n_steps):
        # Relational Encoding: [State, Delta]
        vec = np.zeros(2 * dim)
        vec[:dim] = current_pos
        vec[dim:] = velocity # The Delta
        data.append(vec)
        current_pos = current_pos + velocity
        
    return data, velocity

def run_experiment():
    print("--- SP38: Experiment 14 — World Model Simulation (Relational) ---\n")
    
    DIM = 10
    M_DIM = 2 * DIM
    TRAIN_STEPS = 15
    SIM_STEPS = 10 
    
    train_data, true_velocity = generate_trajectory_data(DIM, TRAIN_STEPS + SIM_STEPS)
    
    # 1. Setup World Model
    forest = Forest()
    observer = Observer(
        forest=forest,
        tau=1.0,
        residual_surprise_threshold=1.5,
        adaptive_compression=True,
        compression_cooccurrence_threshold=2,
        node_use_diag=True
    )
    
    print(f"PHASE 1: Training on {TRAIN_STEPS} transitions...")
    for x in train_data[:TRAIN_STEPS]:
        observer.observe(x)
        
    # 2. Setup Imagination Engine
    # We want a transition node that matches ANY position (context) 
    # but produces the correct RELATIONAL DELTA.
    # Our training data had varying positions but CONSTANT delta.
    # The observer should have synthesized a node with HIGH variance in [0:DIM] 
    # and LOW variance in [DIM:M_DIM].
    
    decoder = Decoder(target_forest=forest, sigma_threshold=1.5)
    
    print("\nPHASE 2: Imagination (Forward Simulation)...")
    
    # Start state: the state after the last training step
    last_train_vec = train_data[TRAIN_STEPS-1]
    current_state = last_train_vec[:DIM] + last_train_vec[DIM:] 
    
    imagined_trajectory = []
    
    print(f"{'Step':<6} | {'Dist to Truth':<15} | {'State (First 3 Dims)'}")
    print("-" * 65)
    
    for k in range(1, SIM_STEPS + 1):
        # GOAL: find transition for current_state
        query_mu = np.zeros(M_DIM)
        query_mu[:DIM] = current_state
        
        goal_node = HFN(mu=query_mu, sigma=np.ones(M_DIM)*10.0, id=f"dream_{k}", use_diag=True)
        goal_node.sigma[:DIM] = 0.5 # Match position context loosely
        
        dec_res = decoder.decode(goal_node)
        
        if isinstance(dec_res, list) and dec_res:
            transition_node = dec_res[0]
            # Output is the RELATIONAL DELTA
            predicted_delta = transition_node.mu[DIM:]
        else:
            print(f"  [ERROR] Step {k}: Simulation collapsed.")
            break
            
        # APPLY RULE: state_t+1 = state_t + delta
        current_state = current_state + predicted_delta
        imagined_trajectory.append(current_state.copy())
        
        # Ground Truth comparison
        # Truth at step k is the start state + k*velocity
        truth_state = (last_train_vec[:DIM] + last_train_vec[DIM:]) + (k * true_velocity)
        dist = np.linalg.norm(current_state - truth_state)
        
        print(f"{k:<6} | {dist:<15.4f} | {current_state[:3]}")
        
    # 3. Results Analysis
    print("\n--- RESULTS ANALYSIS ---")
    if len(imagined_trajectory) == SIM_STEPS:
        avg_dist = np.mean([np.linalg.norm(imagined_trajectory[i] - ((last_train_vec[:DIM] + last_train_vec[DIM:]) + (i+1)*true_velocity)) for i in range(SIM_STEPS)])
        
        print(f"Simulation Horizon: {SIM_STEPS} steps reached.")
        print(f"Average Drift:      {avg_dist:.4f}")
        
        if avg_dist < 0.1:
            print("\n[SUCCESS] HFN acted as a coherent generative world model via Relational Extrapolation.")
        else:
            print("\n[FAIL] Significant drift detected.")
    else:
        print("\n[FAIL] Simulation collapsed.")

if __name__ == "__main__":
    run_experiment()
