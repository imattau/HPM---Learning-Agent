"""
Benchmark: fast online learning vs baseline.

Usage:
    PYTHONPATH=. python3 hpm_ai_v4/simulations/benchmark_fast_learning.py
"""
import time
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern

N_STEPS = 200
N_PATTERNS = 5
OBS_DIM = 5
K = 2
CHUNK = 100

rng = np.random.default_rng(42)
obs_stream = rng.integers(0, OBS_DIM, size=N_STEPS).tolist()

# --- Baseline: adapt() every step ---
patterns_baseline = [HierarchicalPattern(i, latent_dim=K, obs_dim=OBS_DIM)
                     for i in range(N_PATTERNS)]
buffer = []
t0 = time.perf_counter()
for step, obs in enumerate(obs_stream):
    buffer.append(obs)
    if len(buffer) > 100:
        buffer = buffer[-100:]
    for p in patterns_baseline:
        if len(buffer) >= 2:
            p.adapt(buffer)
        p.update_running_loss(buffer)
baseline_time = time.perf_counter() - t0

# --- Fast path: maybe_update() every step ---
patterns_fast = [HierarchicalPattern(i, latent_dim=K, obs_dim=OBS_DIM)
                 for i in range(N_PATTERNS)]
buffer2 = []
t0 = time.perf_counter()
for step, obs in enumerate(obs_stream):
    buffer2.append(obs)
    if len(buffer2) > 100:
        buffer2 = buffer2[-100:]
    for p in patterns_fast:
        p.maybe_update(obs, buffer2)
fast_time = time.perf_counter() - t0

print(f"Baseline ({N_STEPS} steps, {N_PATTERNS} patterns): {baseline_time:.3f}s")
print(f"Fast path ({N_STEPS} steps, {N_PATTERNS} patterns): {fast_time:.3f}s")
if fast_time > 0:
    print(f"Speedup: {baseline_time / fast_time:.1f}x")
else:
    print("Fast path too fast to measure precisely")
