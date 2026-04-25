import time
import numpy as np
import torch # Use torch for timing if available, or just time.perf_counter
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern

def benchmark_agent_throughput(num_steps=1000, num_patterns=10, latent_dim=4, obs_dim=10, num_workers=1):
    agent = HPMAgent(num_initial_patterns=num_patterns, obs_dim=obs_dim, num_workers=num_workers)
    # Set latent dim for all hierarchical patterns
    for p in agent.patterns:
        if isinstance(p, HierarchicalPattern) and not isinstance(p, FlatPattern):
            # Re-init with target latent_dim for benchmark
            p.__init__(p.id, latent_dim=latent_dim, obs_dim=obs_dim)
            
    obs_sequence = np.random.randint(0, obs_dim, size=num_steps)
    
    start_time = time.perf_counter()
    for obs in obs_sequence:
        agent.perceive_and_learn(int(obs))
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    sps = num_steps / total_time
    
    if num_workers > 1:
        agent._pool.close()
        
    return sps, total_time

def benchmark_core_baum_welch(num_iterations=100, latent_dim=4, obs_dim=10, window_size=30):
    p = HierarchicalPattern(0, latent_dim=latent_dim, obs_dim=obs_dim)
    obs_seq = list(np.random.randint(0, obs_dim, size=window_size))
    
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        p.update_parameters_online(obs_seq, window_size=window_size)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    updates_per_sec = num_iterations / total_time
    return updates_per_sec, total_time

def run_benchmarks():
    print("=== HPM v4 Core Baum-Welch (100 iterations) ===")
    print(f"{'K':<5} | {'Updates/sec':>12} | {'Total Time':>10}")
    print("-" * 35)
    for K in [2, 4, 8, 16]:
        ups, t = benchmark_core_baum_welch(num_iterations=1000, latent_dim=K)
        print(f"{K:<5} | {ups:12.2f} | {t:9.3f}s")
    print("\n")

    print("=== HPM v4 Agent Throughput (500 steps) ===")
    # ... rest of run_benchmarks ...
    print(f"{'Config':<40} | {'SPS':>10} | {'Total Time':>10}")
    print("-" * 65)
    
    configs = [
        {"num_patterns": 5,  "latent_dim": 2, "workers": 1},
        {"num_patterns": 10, "latent_dim": 2, "workers": 1},
        {"num_patterns": 10, "latent_dim": 4, "workers": 1},
        {"num_patterns": 10, "latent_dim": 8, "workers": 1},
        {"num_patterns": 20, "latent_dim": 4, "workers": 1},
        {"num_patterns": 20, "latent_dim": 4, "workers": 2},
        {"num_patterns": 20, "latent_dim": 4, "workers": 4},
    ]
    
    for cfg in configs:
        label = f"P={cfg['num_patterns']}, K={cfg['latent_dim']}, W={cfg['workers']}"
        try:
            sps, t = benchmark_agent_throughput(
                num_steps=500, 
                num_patterns=cfg['num_patterns'], 
                latent_dim=cfg['latent_dim'],
                num_workers=cfg['workers']
            )
            print(f"{label:<40} | {sps:10.2f} | {t:9.3f}s")
        except Exception as e:
            print(f"{label:<40} | FAILED: {e}")

if __name__ == "__main__":
    run_benchmarks()
