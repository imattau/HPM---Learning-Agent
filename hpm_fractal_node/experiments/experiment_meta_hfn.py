"""
SP32: Experiment 8 — Meta-HFN Utilisation (Adaptation & Substrate Efficiency)

Tests if self-representation (meta_forest) improves adaptation UNDER PRESSURE.
Setup:
1. Limited Expansion Budget (budget=2).
2. Phase A patterns -> Phase B Shift.
3. Meta-Active agent PRUNES obsolete nodes.

Metrics:
- Structural Efficiency: Forest Size.
- Signal-to-Noise: Avg Surprise.
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.evaluator import Evaluator

def generate_curriculum(dim=100, n_patterns=3, phase="A"):
    patterns = []
    seed = 42 if phase == "A" else 99
    rng = np.random.RandomState(seed)
    offset = 0 if phase == "A" else 50
    for i in range(n_patterns):
        mu = rng.normal(0, 0.01, size=dim)
        start = offset + (i * 10)
        mu[start:start+10] += 10.0 
        patterns.append(mu)
    return patterns

def run_agent(name, observer, curriculum_a, curriculum_b, epochs_per_phase=10, use_pruning=False):
    print(f"\nRunning Agent: {name}...")
    stats = []
    
    # PHASE A
    for epoch in range(1, epochs_per_phase + 1):
        surprises = []
        for x in curriculum_a:
            res = observer.observe(x)
            surprises.append(res.residual_surprise)
        
        if use_pruning:
            pruned = observer.prune(min_weight=0.05)
            # if pruned > 0: print(f"      [META] Pruned {pruned} obsolete nodes.")
            
        stats.append({"epoch": epoch, "surprise": np.mean(surprises), "size": len(observer.forest)})

    # PHASE B (The Shift)
    for epoch in range(1, epochs_per_phase + 1):
        surprises = []
        for x in curriculum_b:
            res = observer.observe(x)
            surprises.append(res.residual_surprise)
            
        if use_pruning:
            observer.prune(min_weight=0.05)
            
        stats.append({"epoch": epochs_per_phase + epoch, "surprise": np.mean(surprises), "size": len(observer.forest)})
        
    return stats

def run_experiment():
    print("--- SP32: Experiment 8 — Meta-HFN Utilisation (Resource Pressure) ---\n")
    
    DIM = 100
    N_PATTERNS = 3
    EPOCHS = 10
    
    curr_a = generate_curriculum(DIM, N_PATTERNS, "A")
    curr_b = generate_curriculum(DIM, N_PATTERNS, "B")
    
    # 1. Agent 1: Meta-Active (Resource Aware)
    forest_1 = Forest()
    observer_1 = Observer(
        forest=forest_1,
        tau=1.0,
        budget=2, # PRESSURE: Very limited expansion budget
        residual_surprise_threshold=1.5,
        absorption_miss_threshold=3, 
        alpha_gain=0.1,
        beta_loss=0.05,
        weight_decay_rate=0.01, # Constant pressure on weights
        node_use_diag=True
    )
    
    # 2. Agent 2: Ablated (Dumb Accumulator)
    forest_2 = Forest()
    observer_2 = Observer(
        forest=forest_2,
        tau=1.0,
        budget=2, # Same pressure
        residual_surprise_threshold=1.5,
        absorption_miss_threshold=999, 
        alpha_gain=0.0,                
        beta_loss=0.0,
        node_use_diag=True
    )
    
    res_1 = run_agent("META-ACTIVE", observer_1, curr_a, curr_b, EPOCHS, use_pruning=True)
    res_2 = run_agent("ABLATED (ACCUMULATOR)", observer_2, curr_a, curr_b, EPOCHS, use_pruning=False)
    
    print("\n" + "="*80)
    print(f"{'Epoch':<6} | {'Meta-Active Size':<20} | {'Ablated Size':<20} | {'Surprise Diff'}")
    print("-" * 80)
    
    for i in range(len(res_1)):
        ep = res_1[i]["epoch"]
        s1 = res_1[i]["surprise"]
        s2 = res_2[i]["surprise"]
        sz1 = res_1[i]["size"]
        sz2 = res_2[i]["size"]
        print(f"{ep:<6} | {sz1:<20} | {sz2:<20} | {s1-s2:<15.4f}")

    # Analysis
    # Structural Efficiency = (Total Surprise * Forest Size) - lower is better
    efficiency_1 = np.mean([r["surprise"] for r in res_1]) * res_1[-1]["size"]
    efficiency_2 = np.mean([r["surprise"] for r in res_2]) * res_2[-1]["size"]
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Meta-Active Complexity-Error Product: {efficiency_1:.2f}")
    print(f"Ablated Complexity-Error Product:     {efficiency_2:.2f}")
    
    if efficiency_1 < (efficiency_2 * 0.7):
        print("\n[SUCCESS] Meta-HFN improved learning efficiency by > 30%.")
    else:
        print("\n[FAIL] No significant efficiency benefit detected.")

if __name__ == "__main__":
    run_experiment()
