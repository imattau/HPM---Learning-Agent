"""
Multi-Agent Benchmark 1: Elegance Recovery
==========================================
Same law-recovery task as elegance_recovery.py, but with two agents sharing
a PatternField. Each agent is seeded with a different subset of atomic
algebraic building blocks:

  agent_a: ["x", "x**2", "1+x"]      ← has the key recombination term
  agent_b: ["1", "x**3", "1+x"]      ← has x**3 as alternative hypothesis

"1+x" is shared — it's the term that appears in the true law y = x²/(1+x),
so cross-agent communication should reinforce it.

The benchmark tests whether the agents collectively recover the law faster
and more reliably than if each agent worked from only its own subset.

Run:
    python benchmarks/multi_agent_elegance_recovery.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hpm.substrate.base import hash_vectorise
from hpm.patterns.gaussian import GaussianPattern
from benchmarks.multi_agent_common import (
    make_orchestrator, compute_redundancy, print_results_table,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIM = 16
N_STEPS = 1500
N_TEST = 200
GAP_THRESHOLD = 0.0
RNG_SEED = 42

# Each agent starts with a different subset of atomic expressions.
# "1+x" appears in both — it's the key term in y = x²/(1+x) and will be
# reinforced when agents communicate via the shared PatternField.
AGENT_EXPRESSIONS = {
    "elegance_a": ["x", "x**2", "1+x"],
    "elegance_b": ["1", "x**3", "1+x"],
}


def _sample_pairs(rng, n, fn):
    vecs = []
    while len(vecs) < n:
        x = rng.uniform(-2.0, 2.0)
        if abs(1.0 + x) < 0.1:
            continue
        y = fn(x)
        vecs.append(hash_vectorise(f"{x:.4f} {y:.4f}", FEATURE_DIM))
    return vecs


def _true_law(x):
    return x ** 2 / (1.0 + x)


def _distractor(x):
    return x ** 2


# Map expression names to numpy functions for semantic seed encoding
_EXPR_FNS = {
    "x":    lambda x: x,
    "x**2": lambda x: x ** 2,
    "1+x":  lambda x: 1.0 + x,
    "1":    lambda x: np.ones_like(x),
    "x**3": lambda x: x ** 3,
}


def _expr_mu(expr: str, n_samples: int = 400) -> np.ndarray:
    """
    Compute seed mu for an expression by evaluating it at many x values and
    taking the mean hash vector of the resulting (x, y) pairs.

    Puts expression seeds in the same encoding space as observations so that
    pattern distances are mathematically meaningful rather than arbitrary
    string-hash distances.
    """
    fn = _EXPR_FNS[expr]
    rng = np.random.default_rng(0)
    vecs = []
    while len(vecs) < n_samples:
        x = rng.uniform(-2.0, 2.0)
        if abs(1.0 + x) < 0.1:
            continue
        y = float(fn(x))
        if not np.isfinite(y) or abs(y) > 1e6:
            continue
        vecs.append(hash_vectorise(f"{x:.4f} {y:.4f}", FEATURE_DIM))
    return np.mean(vecs, axis=0)


def seed_agent_expressions(agent, expressions: list[str]) -> None:
    """Pre-seed agent store with its assigned subset of atomic expressions.

    Seeds are placed at the centroid of hash-encoded (x, y) observations for
    each expression, so patterns compete on mathematical similarity to the true
    law rather than on arbitrary string-hash geometry.
    """
    records = agent.store.query(agent.agent_id)
    for p, _ in records:
        agent.store.delete(p.id)
    init_weight = 1.0 / len(expressions)
    for expr in expressions:
        mu = _expr_mu(expr)
        sigma = np.eye(FEATURE_DIM) * agent.config.init_sigma
        pattern = GaussianPattern(mu=mu, sigma=sigma, level=4, freeze_mu=False)
        agent.store.save(pattern, init_weight, agent.agent_id)


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)

    orch, agents, store = make_orchestrator(
        n_agents=2,
        feature_dim=FEATURE_DIM,
        agent_ids=["elegance_a", "elegance_b"],
        with_monitor=True,
        T_monitor=50,
        min_recomb_level=1,
        init_sigma=0.1,
    )

    # Seed each agent with its assigned subset of atomic expressions
    for agent in agents:
        seed_agent_expressions(agent, AGENT_EXPRESSIONS[agent.agent_id])

    recomb_counts = {a.agent_id: 0 for a in agents}

    for _ in range(N_STEPS):
        x_val = rng.uniform(-2.0, 2.0)
        if abs(1.0 + x_val) < 0.1:
            continue
        obs_vec = hash_vectorise(f"{x_val:.4f} {_true_law(x_val):.4f}", FEATURE_DIM)
        observations = {a.agent_id: obs_vec for a in agents}
        result = orch.step(observations)
        for agent in agents:
            if result[agent.agent_id].get("recombination_accepted"):
                recomb_counts[agent.agent_id] += 1

    # Evaluate per-agent gap
    true_vecs = _sample_pairs(rng, N_TEST, _true_law)
    dist_vecs = _sample_pairs(rng, N_TEST, _distractor)

    agent_results = []
    for agent in agents:
        records = agent.store.query(agent.agent_id)
        if not records:
            agent_results.append({
                "agent_id": agent.agent_id,
                "expressions": ", ".join(AGENT_EXPRESSIONS[agent.agent_id]),
                "gap": 0.0, "nll_true": 0.0, "nll_distractor": 0.0,
                "top_weight": 0.0, "recomb_count": recomb_counts[agent.agent_id],
            })
            continue

        top_pattern, top_weight = max(records, key=lambda pw: pw[1])
        nll_true = float(np.mean([-top_pattern.log_prob(v) for v in true_vecs]))
        nll_dist = float(np.mean([-top_pattern.log_prob(v) for v in dist_vecs]))
        agent_results.append({
            "agent_id": agent.agent_id,
            "expressions": ", ".join(AGENT_EXPRESSIONS[agent.agent_id]),
            "gap": nll_dist - nll_true,
            "nll_true": nll_true,
            "nll_distractor": nll_dist,
            "top_weight": top_weight,
            "recomb_count": recomb_counts[agent.agent_id],
        })

    level4plus = sum(1 for p, _, _ in store.query_all() if getattr(p, "level", 1) >= 4)

    return {
        "agent_results": agent_results,
        "final_redundancy": compute_redundancy(orch),
        "final_level4plus": level4plus,
    }


def main():
    result = run()
    agent_results = result["agent_results"]
    redundancy = result["final_redundancy"]
    level4plus = result["final_level4plus"]

    rows = []
    for ar in agent_results:
        gap = ar["gap"]
        recovered = gap > GAP_THRESHOLD
        rows.append({
            "Agent": ar["agent_id"],
            "Seed Expressions": ar["expressions"],
            "Recomb": str(ar["recomb_count"]),
            "NLL True": f"{ar['nll_true']:.2f}",
            "NLL Distractor": f"{ar['nll_distractor']:.2f}",
            "Gap": f"{gap:+.2f}",
            "Result": f"{'✓' if recovered else '✗'} {'RECOVERED' if recovered else 'NOT RECOVERED'}",
        })

    print_results_table(
        title="Multi-Agent Elegance Recovery (2 agents, partitioned atomic seeds, shared field)",
        cols=["Agent", "Seed Expressions", "Recomb", "NLL True", "NLL Distractor", "Gap", "Result"],
        rows=rows,
    )

    redundancy_str = f"{redundancy:.3f}" if redundancy is not None else "N/A"
    print_results_table(
        title="Shared Field Quality",
        cols=["Metric", "Value"],
        rows=[
            {"Metric": "Field Redundancy", "Value": redundancy_str},
            {"Metric": "Level 4+ Patterns", "Value": str(level4plus)},
        ],
    )


if __name__ == "__main__":
    main()
