"""
Benchmark 1: Elegance Recovery
================================
Tests whether the Recombination Operator produces compact patterns that
structurally match a hidden mathematical law.

Hidden function: y = x^2 / (1 + x),  x ~ Uniform[-2, 2]

Run:
    python benchmarks/elegance_recovery.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hpm.substrate.base import hash_vectorise
from hpm.patterns.gaussian import GaussianPattern
from benchmarks.common import make_agent, print_results_table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIM = 16
N_STEPS = 1500
SRR_THRESHOLD = 0.7
RNG_SEED = 42

# Atomic building blocks (algebraic seeds)
ATOMIC_EXPRESSIONS = ["x", "1", "x**2", "x**3", "1+x"]
TARGET_EXPRESSION = "x**2 / (1 + x)"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def seed_atomic_patterns(agent) -> None:
    """Pre-seed agent store with atomic algebraic building blocks."""
    n = len(ATOMIC_EXPRESSIONS)
    init_weight = 1.0 / n

    # Clear the auto-seeded random pattern first
    records = agent.store.query(agent.agent_id)
    for p, _ in records:
        agent.store.delete(p.id)

    for expr in ATOMIC_EXPRESSIONS:
        mu = hash_vectorise(expr, FEATURE_DIM)
        sigma = np.eye(FEATURE_DIM) * agent.config.init_sigma
        # level=4 placeholder (overridden by min_recomb_level=1 in config)
        # freeze_mu=True: symbolic algebraic patterns must not converge toward observations
        pattern = GaussianPattern(mu=mu, sigma=sigma, level=4, freeze_mu=True)
        agent.store.save(pattern, init_weight, agent.agent_id)


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    agent = make_agent(feature_dim=FEATURE_DIM, agent_id="elegance_bench", min_recomb_level=1, init_sigma=0.1)

    # Pre-seed with atomic patterns
    seed_atomic_patterns(agent)

    recomb_count = 0

    for _ in range(N_STEPS):
        x_val = rng.uniform(-2.0, 2.0)
        y_val = x_val ** 2 / (1.0 + x_val)
        obs_str = f"{x_val:.4f} {y_val:.4f}"
        obs_vec = hash_vectorise(obs_str, FEATURE_DIM)

        result = agent.step(obs_vec)
        if result.get("recombination_accepted"):
            recomb_count += 1

    # Evaluate: find top-weighted pattern
    records = agent.store.query(agent.agent_id)
    if not records:
        return {"srr": 0.0, "top_weight": 0.0, "recomb_count": recomb_count}

    top_pattern, top_weight = max(records, key=lambda pw: pw[1])

    target_vec = hash_vectorise(TARGET_EXPRESSION, FEATURE_DIM)
    srr = cosine_similarity(top_pattern.mu, target_vec)

    # Sanity check
    assert 0.0 <= srr <= 1.0 + 1e-9, f"SRR out of range: {srr}"

    return {
        "srr": srr,
        "top_weight": top_weight,
        "recomb_count": recomb_count,
    }


def main():
    metrics = run()

    srr = metrics["srr"]
    result_label = "RECOVERED" if srr > SRR_THRESHOLD else "NOT RECOVERED"

    print_results_table(
        title="Elegance Recovery",
        cols=["Steps", "Top Weight", "Recomb Count", "SRR", "Result"],
        rows=[{
            "Steps": str(N_STEPS),
            "Top Weight": f"{metrics['top_weight']:.2f}",
            "Recomb Count": str(metrics["recomb_count"]),
            "SRR": f"{srr:.2f}",
            "Result": f"{'✓' if srr > SRR_THRESHOLD else '✗'} {result_label}",
        }],
    )


if __name__ == "__main__":
    main()
