"""Run the Regime Shift Adaptation (RSA) Benchmark."""

from __future__ import annotations

from hpm_ai_v5.planning.rsa import RSABenchmark
from hpm_ai_v5.planning.cartpole import CartpoleEnvConfig


def run_rsa_benchmark():
    benchmark = RSABenchmark()
    
    # Define Phases
    phases = [
        {
            "name": "Normal",
            "count": 100,
            "config": CartpoleEnvConfig(mass_cart=0.1, length=0.5)
        },
        {
            "name": "Heavy",
            "count": 50, # 101-150
            "config": CartpoleEnvConfig(mass_cart=1.0, length=0.5)
        },
        {
            "name": "Normal Return",
            "count": 50, # 151-200
            "config": CartpoleEnvConfig(mass_cart=0.1, length=0.5)
        }
    ]
    
    result = benchmark.run(phases)
    
    print("\n\nRSA Benchmark Results:")
    print(f"Total Episodes: {len(result.scores)}")
    print(f"Detection Points (Episode Index): {sorted(list(set(result.detection_points)))}")
    print(f"Regime Shift Points: {result.regime_shifts}")
    
    # Calculate performance metrics
    # Normal Baseline (Last 10 of Phase 1)
    p1_avg = sum(result.scores[90:100]) / 10
    print(f"Phase 1 (Normal) Baseline: {p1_avg:.2f}")
    
    # Heavy Adaptation (Phase 2)
    p2_avg = sum(result.scores[140:150]) / 10
    print(f"Phase 2 (Heavy) Adapted: {p2_avg:.2f}")
    
    # Return (Phase 3)
    p3_avg = sum(result.scores[190:200]) / 10
    print(f"Phase 3 (Return) Recovered: {p3_avg:.2f}")
    
    # Check success criteria
    detection_ok = any(100 <= dp <= 105 for dp in result.detection_points)
    print(f"Detection Success (Shift 1): {'PASS' if detection_ok else 'FAIL'}")
    
    recovery_ok = p3_avg >= (0.8 * p1_avg)
    print(f"Recovery Success (Phase 3): {'PASS' if recovery_ok else 'FAIL'} ({p3_avg/p1_avg:.2%})")


if __name__ == "__main__":
    run_rsa_benchmark()
