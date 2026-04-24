# hpm_ai_v4/simulations/benchmark_parallel.py
"""
Benchmark steps/second for HPMAgent with varying num_workers.

Usage:
    python3 -m hpm_ai_v4.simulations.benchmark_parallel
"""
import time
import numpy as np
from hpm_ai_v4.agents.agent import HPMAgent

STEPS = 100 # Reduced from 500 for faster verification during implementation
NUM_PATTERNS = 16
OBS_DIM = 4
CONFIGS = [1, 2, 4] # Reduced configs for faster verification


def run_benchmark(num_workers: int) -> float:
    np.random.seed(42)
    agent = HPMAgent(num_initial_patterns=NUM_PATTERNS,
                     obs_dim=OBS_DIM,
                     num_workers=num_workers)
    obs_seq = list(np.random.randint(0, OBS_DIM, size=STEPS))

    t0 = time.perf_counter()
    for obs in obs_seq:
        agent.perceive_and_learn(int(obs))
    elapsed = time.perf_counter() - t0

    agent._pool.close()
    return STEPS / elapsed


def main():
    print(f"\nBenchmark: {STEPS} steps, {NUM_PATTERNS} patterns, obs_dim={OBS_DIM}")
    print(f"{'num_workers':>12} {'steps/s':>10} {'speedup':>10}")
    print("-" * 36)

    baseline = None
    for nw in CONFIGS:
        sps = run_benchmark(nw)
        if baseline is None:
            baseline = sps
        speedup = sps / baseline
        print(f"{nw:>12} {sps:>10.1f} {speedup:>10.2f}x")


if __name__ == '__main__':
    main()
