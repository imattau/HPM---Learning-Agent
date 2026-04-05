"""
SP31: Experiment 7 — Closed-Loop Learning (World Model Refinement)

Validates the fundamental HPM cycle:
observe → explain → fail → create → re-observe

We track average residual surprise and forest size over multiple epochs
of structured observations.
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.evaluator import Evaluator

def generate_synthetic_curriculum(dim=100, n_patterns=5):
    """Generates a sequence of distinct, structured Gaussian patterns."""
    patterns = []
    rng = np.random.RandomState(42)
    
    for i in range(n_patterns):
        # Base noise
        mu = rng.normal(0, 0.01, size=dim)
        # SHARP FEATURE: Strong signal makes log-prob discrimination easy
        start = i * 20
        mu[start:start+10] += 10.0 
        patterns.append(mu)
    
    return patterns

def run_experiment():
    print("--- SP31: Experiment 7 — Closed-Loop Learning ---\n")
    
    DIM = 100
    N_PATTERNS = 5
    EPOCHS = 10
    
    # 1. Setup
    forest = Forest()
    evaluator = Evaluator()
    
    # CALIBRATION:
    # Perfect match surprise in 1D is ~0.918 (log(sqrt(2pi e))).
    # We set tau > perfect match so it can 'explain' known patterns.
    observer = Observer(
        forest=forest,
        evaluator=evaluator,
        tau=1.0, 
        residual_surprise_threshold=1.5,
        adaptive_compression=True,
        compression_cooccurrence_threshold=2,
        node_use_diag=True # Fast O(D)
    )
    
    curriculum = generate_synthetic_curriculum(dim=DIM, n_patterns=N_PATTERNS)
    
    # 2. Loop
    stats = []
    
    print(f"{'Epoch':<8} | {'Avg Surprise':<15} | {'Forest Size':<12} | {'New Nodes'}")
    print("-" * 60)
    
    prev_size = len(forest)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_surprises = []
        
        # Consistent order within epoch for learning stability
        for x in curriculum:
            # THE CORE CYCLE: observe -> explain -> fail -> create
            res = observer.observe(x)
            
            # Normalize surprise by dimension
            norm_surprise = res.residual_surprise
            epoch_surprises.append(norm_surprise)
            
        avg_surprise = np.mean(epoch_surprises)
        current_size = len(forest)
        new_nodes = current_size - prev_size
        prev_size = current_size
        
        print(f"{epoch:<8} | {avg_surprise:<15.6f} | {current_size:<12} | {new_nodes}")
        
        stats.append({
            "epoch": epoch,
            "surprise": avg_surprise,
            "size": current_size
        })
        
    # 3. Validation
    print("\n--- RESULTS ANALYSIS ---")
    start_surprise = stats[0]["surprise"]
    end_surprise = stats[-1]["surprise"]
    surprise_reduction = (start_surprise - end_surprise) / start_surprise if start_surprise > 0 else 0
    
    print(f"Total Surprise Reduction: {surprise_reduction*100:.1f}%")
    print(f"Final Forest Structure:   {len(forest)} nodes")
    
    # Expected: Epoch 1 creates nodes. Epoch 2 uses them.
    if surprise_reduction > 0.3:
        print("\n[SUCCESS] The system successfully improved its world model (Surprise ↓).")
    else:
        print("\n[FAIL] Surprise did not decrease significantly.")

if __name__ == "__main__":
    run_experiment()
