"""
Benchmark 1: Elegance Recovery
================================
Tests whether the agent recovers the specific structure of a hidden mathematical
law, not just any smooth mapping.

Metric — Generalization Gap (NLL-based):
  Train on y = x² / (1 + x).
  After training, evaluate the top-weighted pattern on:
    - Test set A (true law):   200 samples from y = x² / (1 + x)
    - Test set B (distractor): 200 samples from y = x²
  Gap = mean_NLL(distractor) - mean_NLL(true)
  Positive gap → agent fits the specific training law better than the distractor.

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
N_TEST = 200
GAP_THRESHOLD = 0.0   # gap > 0 means agent fits true law better than distractor
RNG_SEED = 42

ATOMIC_EXPRESSIONS = ["x", "1", "x**2", "x**3", "1+x"]


def _sample_pairs(rng: np.random.Generator, n: int, fn) -> list[np.ndarray]:
    """Sample n (x, y) observation vectors from fn(x), skipping near-singularities."""
    vecs = []
    while len(vecs) < n:
        x = rng.uniform(-2.0, 2.0)
        if abs(1.0 + x) < 0.1:   # avoid x ≈ -1 singularity in true law
            continue
        y = fn(x)
        vecs.append(hash_vectorise(f"{x:.4f} {y:.4f}", FEATURE_DIM))
    return vecs


def _true_law(x: float) -> float:
    return x ** 2 / (1.0 + x)


def _distractor(x: float) -> float:
    return x ** 2


def seed_atomic_patterns(agent) -> None:
    """Pre-seed agent store with atomic algebraic building blocks."""
    records = agent.store.query(agent.agent_id)
    for p, _ in records:
        agent.store.delete(p.id)

    init_weight = 1.0 / len(ATOMIC_EXPRESSIONS)
    for expr in ATOMIC_EXPRESSIONS:
        mu = hash_vectorise(expr, FEATURE_DIM)
        sigma = np.eye(FEATURE_DIM) * agent.config.init_sigma
        pattern = GaussianPattern(mu=mu, sigma=sigma, level=4, freeze_mu=True)
        agent.store.save(pattern, init_weight, agent.agent_id)


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    agent = make_agent(
        feature_dim=FEATURE_DIM,
        agent_id="elegance_bench",
        min_recomb_level=1,
        init_sigma=0.1,
    )
    seed_atomic_patterns(agent)

    recomb_count = 0
    for _ in range(N_STEPS):
        x_val = rng.uniform(-2.0, 2.0)
        if abs(1.0 + x_val) < 0.1:
            continue
        obs_vec = hash_vectorise(f"{x_val:.4f} {_true_law(x_val):.4f}", FEATURE_DIM)
        result = agent.step(obs_vec)
        if result.get("recombination_accepted"):
            recomb_count += 1

    records = agent.store.query(agent.agent_id)
    if not records:
        return {"gap": 0.0, "nll_true": 0.0, "nll_distractor": 0.0,
                "top_weight": 0.0, "recomb_count": recomb_count}

    top_pattern, top_weight = max(records, key=lambda pw: pw[1])

    # Generalization gap: evaluate top pattern on true law vs distractor
    true_vecs = _sample_pairs(rng, N_TEST, _true_law)
    dist_vecs = _sample_pairs(rng, N_TEST, _distractor)

    nll_true = float(np.mean([-top_pattern.log_prob(v) for v in true_vecs]))
    nll_dist = float(np.mean([-top_pattern.log_prob(v) for v in dist_vecs]))
    gap = nll_dist - nll_true   # positive = agent fits true law better

    return {
        "gap": gap,
        "nll_true": nll_true,
        "nll_distractor": nll_dist,
        "top_weight": top_weight,
        "recomb_count": recomb_count,
    }


def main():
    metrics = run()
    gap = metrics["gap"]
    recovered = gap > GAP_THRESHOLD
    result_label = "RECOVERED" if recovered else "NOT RECOVERED"

    print_results_table(
        title="Elegance Recovery",
        cols=["Steps", "Top Weight", "Recomb Count", "NLL True", "NLL Distractor", "Gap", "Result"],
        rows=[{
            "Steps": str(N_STEPS),
            "Top Weight": f"{metrics['top_weight']:.2f}",
            "Recomb Count": str(metrics["recomb_count"]),
            "NLL True": f"{metrics['nll_true']:.2f}",
            "NLL Distractor": f"{metrics['nll_distractor']:.2f}",
            "Gap": f"{gap:+.2f}",
            "Result": f"{'✓' if recovered else '✗'} {result_label}",
        }],
    )


if __name__ == "__main__":
    main()
