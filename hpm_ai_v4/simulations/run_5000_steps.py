import os
import time
import numpy as np
from hpm_ai_v4.simulations.wikipedia_sim import run_simulation

CORPUS_PATH = "hpm_ai_v4/simulations/data/wiki_sample.txt"
STEPS = 2000
LOG_EVERY = 100
WORKERS = 4

if __name__ == "__main__":
    print(f"Starting {STEPS}-step simulation on {CORPUS_PATH}...")
    start_time = time.perf_counter()
    
    results = run_simulation(
        corpus_path=CORPUS_PATH,
        total_steps=STEPS,
        log_every=LOG_EVERY,
        num_workers=WORKERS
    )
    
    duration = time.perf_counter() - start_time
    print(f"\nSimulation finished in {duration:.2f}s ({STEPS/duration:.1f} steps/s)")
    
    # Analysis
    print("\n--- Analysis ---")
    for level_name, patterns in [("L1", results['L1_patterns']), 
                                 ("L2", results['L2_patterns']), 
                                 ("L3", results['L3_patterns'])]:
        print(f"\n{level_name} Level:")
        print(f"  Total patterns: {len(patterns)}")
        
        # Sort by weight
        sorted_p = sorted(patterns, key=lambda p: p.weight, reverse=True)
        top_p = sorted_p[:3]
        
        for i, p in enumerate(top_p):
            print(f"  Top {i+1} Pattern (ID {p.id}):")
            print(f"    Weight: {p.weight:.4f}")
            print(f"    Running Loss: {p.running_loss:.4f}")
            if p.complexity >= 2:
                print(f"    Latent Dim (K): {p.latent_dim}")
    
    print("\nFinal Buffers Lengths:")
    print(f"  L1: {len(results['buf1'])}")
    print(f"  L2: {len(results['buf2'])}")
    print(f"  L3: {len(results['buf3'])}")
