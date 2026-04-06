"""
SP36: Experiment 12 — Self-Curiosity (Autonomous Demand-Driven Learning)

Tests if the HFN system can drive its own learning trajectory via 'Self-Play'.
Setup:
1. Initialize with a few seed priors (Input + Delta rules).
2. Autonomous Loop: Generate (Dream) -> Perceive -> Evaluate -> Expand.
3. Observe: Does it discover new combinations/rules without external training data?

Metrics:
- Structural Growth: Number of discovered nodes over time.
- Surprise Trajectory: Cyclic spikes (discovery) followed by integration (collapse).
- Diversity: Variance of self-generated observations.
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.evaluator import Evaluator

def initialize_seed_priors(dim=20):
    """Seed with 2 orthogonal atomic rules."""
    forest = Forest()
    
    # Rule 1: Increment Dim 0
    mu1 = np.zeros(dim)
    mu1[dim//2 + 0] = 1.0 # Delta at index 10
    rule1 = HFN(mu=mu1, sigma=np.ones(dim)*0.001, id="seed_inc_0", use_diag=True)
    forest.register(rule1)
    
    # Rule 2: Increment Dim 1
    mu2 = np.zeros(dim)
    mu2[dim//2 + 1] = 1.0 # Delta at index 11
    rule2 = HFN(mu=mu2, sigma=np.ones(dim)*0.001, id="seed_inc_1", use_diag=True)
    forest.register(rule2)
    
    return forest

def run_experiment():
    print("--- SP36: Experiment 12 — Self-Curiosity (True Autonomy) ---\n")
    
    DIM = 20
    DREAMS = 50
    forest = initialize_seed_priors(DIM)
    decoder = Decoder(target_forest=forest, sigma_threshold=0.1)
    observer = Observer(
        forest=forest,
        tau=1.0,
        residual_surprise_threshold=1.2,
        adaptive_compression=True,
        compression_cooccurrence_threshold=3,
        node_use_diag=True
    )
    
    stats = []
    print(f"{'Dream':<6} | {'Source Node':<15} | {'Surprise':<10} | {'Forest Size'}")
    print("-" * 60)
    
    for d in range(1, DREAMS + 1):
        # 1. GENERATE (Dream): Randomly sample an existing node and add noise
        active_nodes = list(forest.active_nodes())
        source = np.random.choice(active_nodes)
        
        # Add exploration noise to the mean
        dream_mu = source.mu + np.random.normal(0, 0.5, size=DIM)
        
        # Synthesis: Use decoder to collapse dream into a concrete perception
        dream_goal = HFN(mu=dream_mu, sigma=np.ones(DIM)*5.0, id=f"dream_goal_{d}", use_diag=True)
        dec_res = decoder.decode(dream_goal)
        
        # PERCEIVE: Feed generated observation back into the Observer
        if isinstance(dec_res, list) and dec_res:
            observation = dec_res[0].mu
        else:
            # Fallback to noisy mu if decoder couldn't collapse
            observation = dream_mu
            
        # EVALUATE & EXPAND
        res = observer.observe(observation)
        
        # Track statistics
        stats.append({
            "dream": d,
            "surprise": res.residual_surprise,
            "size": len(list(forest.active_nodes()))
        })
        
        if d % 5 == 0 or res.residual_surprise > 1.2:
            print(f"{d:<6} | {source.id[:15]:<15} | {res.residual_surprise:<10.4f} | {stats[-1]['size']}")

    # Analysis
    print("\n--- RESULTS ANALYSIS ---")
    initial_size = 2 # seed priors
    final_size = stats[-1]["size"]
    growth = final_size - initial_size
    
    surprises = [s["surprise"] for s in stats]
    avg_surprise = np.mean(surprises)
    max_surprise = np.max(surprises)
    
    print(f"Total Structural Growth: {growth} new concepts discovered.")
    print(f"Average Self-Surprise:   {avg_surprise:.4f}")
    print(f"Peak Discovery Surprise: {max_surprise:.4f}")
    
    if growth > 5 and avg_surprise < max_surprise:
        print("\n[SUCCESS] The system successfully drove its own learning trajectory.")
        print("It explored the manifold, encountered internal gaps, and autonomously expanded its forest.")
    else:
        print("\n[FAIL] No significant autonomous learning detected.")

if __name__ == "__main__":
    run_experiment()
