"""
Multi-Agent Benchmark: Reber Grammar — CategoricalPattern
==========================================================
Three categorical agents share a PatternField and learn the Reber Grammar.
All agents see all training sequences via orch.step(), exercising the full
orchestrator: PatternField, StructuralLawMonitor, and RecombinationStrategist.

Protocol:
  - Generate N_TRAIN valid sequences; step all 3 agents on each via orch.step()
  - StructuralLawMonitor fires every T_monitor steps; RecombinationStrategist
    triggers recombination bursts to prevent convergence
  - Generate N_TEST valid + N_TEST invalid sequences
  - Evaluate ensemble NLL (weighted sum across agents) for each test sequence
  - Report AUROC and NLL separation vs single-agent baseline (same data)

Run:
    python benchmarks/multi_agent_reber_grammar.py

Expected: ensemble AUROC comparable to or better than single-agent;
benchmark primarily validates that all orchestrator machinery works with
CategoricalPattern (integer observations, MC KL, entropy-based metrics).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.common import print_results_table
from benchmarks.multi_agent_common import make_orchestrator
from benchmarks.multi_agent_arc import ensemble_score
from hpm.patterns.categorical import CategoricalPattern

# ---------------------------------------------------------------------------
# Grammar (shared with reber_grammar.py)
# ---------------------------------------------------------------------------
SYMBOLS = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
SYM = {s: i for i, s in enumerate(SYMBOLS)}
K_GRAMMAR = len(SYMBOLS)
K = K_GRAMMAR + 1   # 8 (including PAD)
PAD = K_GRAMMAR

TRANSITIONS = {
    0: [('B', 1)],
    1: [('T', 2), ('P', 3)],
    2: [('S', 2), ('X', 3)],
    3: [('X', 2), ('S', 4)],
    4: [('P', 1), ('V', 5)],
    5: [('V', 5), ('E', -1)],
}

MAX_LEN = 20
D = MAX_LEN


def generate_reber(rng: np.random.Generator) -> list[int]:
    state = 0
    seq = []
    for _ in range(100):
        choices = TRANSITIONS[state]
        sym_str, next_state = choices[rng.integers(len(choices))]
        seq.append(SYM[sym_str])
        if next_state == -1:
            break
        state = next_state
    return seq


def encode(seq: list[int]) -> np.ndarray:
    arr = np.full(D, PAD, dtype=np.intp)
    n = min(len(seq), D)
    arr[:n] = seq[:n]
    return arr


def generate_invalid(rng: np.random.Generator) -> list[int]:
    length = rng.integers(4, 13)
    return list(rng.integers(0, K_GRAMMAR, size=length))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
N_TRAIN = 2000
N_TEST = 500
N_AGENTS = 3
RNG_SEED = 42
TRAIN_REPS = 3   # step() calls per sequence (mirrors ARC train_reps pattern)


def auroc(valid_nlls: np.ndarray, invalid_nlls: np.ndarray) -> float:
    """Fraction of (invalid, valid) pairs where invalid NLL > valid NLL."""
    return float((invalid_nlls[:, None] > valid_nlls[None, :]).mean())


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)

    # Generate all sequences upfront
    train_seqs = [generate_reber(rng) for _ in range(N_TRAIN)]
    test_valid = [generate_reber(rng) for _ in range(N_TEST)]
    test_invalid = [generate_invalid(rng) for _ in range(N_TEST)]

    # -----------------------------------------------------------------------
    # Single-agent baseline: 1 agent, 1/N_AGENTS of training data
    # -----------------------------------------------------------------------
    subset = train_seqs[::N_AGENTS]   # every Nth sequence
    solo_pattern = CategoricalPattern(np.ones((D, K)) / K, K=K)
    for seq in subset:
        x = encode(seq)
        for _ in range(TRAIN_REPS):
            solo_pattern = solo_pattern.update(x)

    solo_valid = np.array([solo_pattern.log_prob(encode(s)) for s in test_valid])
    solo_invalid = np.array([solo_pattern.log_prob(encode(s)) for s in test_invalid])
    solo_auroc = auroc(solo_valid, solo_invalid)
    solo_sep = float(solo_invalid.mean() - solo_valid.mean())

    # -----------------------------------------------------------------------
    # Multi-agent: 3 agents, all see all training sequences via orch.step().
    # Monitor (every T_monitor steps) + RecombinationStrategist fire normally.
    # Symmetry is broken by agent-level stochastic dynamics and recombination.
    # -----------------------------------------------------------------------
    agent_ids = [f"reber_{chr(ord('a') + i)}" for i in range(N_AGENTS)]
    orch, agents, store = make_orchestrator(
        n_agents=N_AGENTS,
        feature_dim=D,
        agent_ids=agent_ids,
        pattern_types=["categorical"] * N_AGENTS,
        alphabet_size=K,
        with_monitor=True,
        T_monitor=200,
        with_forecaster=True,
        with_actor=True,
        n_actions=K,  # one action per grammar symbol
    )

    for seq in train_seqs:
        x = encode(seq)
        obs = {aid: x for aid in agent_ids}
        for _ in range(TRAIN_REPS):
            orch.step(obs)

    # Ensemble score: sum of per-agent ensemble_score (weighted NLL)
    multi_valid = np.array([ensemble_score(agents, encode(s)) for s in test_valid])
    multi_invalid = np.array([ensemble_score(agents, encode(s)) for s in test_invalid])
    multi_auroc = auroc(multi_valid, multi_invalid)
    multi_sep = float(multi_invalid.mean() - multi_valid.mean())

    n_patterns = len(store.query_all())

    return {
        "solo_auroc": solo_auroc,
        "solo_sep": solo_sep,
        "solo_valid_nll": float(solo_valid.mean()),
        "solo_invalid_nll": float(solo_invalid.mean()),
        "multi_auroc": multi_auroc,
        "multi_sep": multi_sep,
        "multi_valid_nll": float(multi_valid.mean()),
        "multi_invalid_nll": float(multi_invalid.mean()),
        "n_patterns": n_patterns,
        "n_agents": N_AGENTS,
        "subset_size": len(subset),
    }


def main():
    print(f"Running Multi-Agent Reber Grammar benchmark ({N_AGENTS} agents)...")
    m = run()

    improvement = m["multi_auroc"] - m["solo_auroc"]
    passed = m["multi_auroc"] >= m["solo_auroc"] - 0.02   # allow 2pp tolerance
    result_label = "✓ PASS" if passed else "✗ FAIL"

    print_results_table(
        title=f"Reber Grammar — Single vs Multi-Agent (D={D}, K={K}, train={N_TRAIN})",
        cols=["Setup", "Valid NLL", "Invalid NLL", "Separation", "AUROC"],
        rows=[
            {
                "Setup": f"Single agent ({m['subset_size']} seqs, same data)",
                "Valid NLL": f"{m['solo_valid_nll']:.2f}",
                "Invalid NLL": f"{m['solo_invalid_nll']:.2f}",
                "Separation": f"{m['solo_sep']:.2f}",
                "AUROC": f"{m['solo_auroc']:.3f}",
            },
            {
                "Setup": f"3-agent ensemble ({N_TRAIN} seqs, partitioned)",
                "Valid NLL": f"{m['multi_valid_nll']:.2f}",
                "Invalid NLL": f"{m['multi_invalid_nll']:.2f}",
                "Separation": f"{m['multi_sep']:.2f}",
                "AUROC": f"{m['multi_auroc']:.3f}",
            },
        ],
    )

    print_results_table(
        title="Result",
        cols=["Criterion", "Value", "Status"],
        rows=[
            {
                "Criterion": "AUROC improvement (multi vs solo-subset)",
                "Value": f"{improvement:+.3f}",
                "Status": "✓" if improvement >= 0 else "~",
            },
            {
                "Criterion": "Shared patterns in store",
                "Value": str(m["n_patterns"]),
                "Status": "✓" if m["n_patterns"] > 0 else "✗",
            },
            {
                "Criterion": "Overall",
                "Value": "",
                "Status": result_label,
            },
        ],
    )


if __name__ == "__main__":
    main()
