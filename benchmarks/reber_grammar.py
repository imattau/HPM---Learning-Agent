"""
Benchmark: Reber Grammar — CategoricalPattern Discrete Learning
===============================================================
Classic cognitive-science benchmark for evaluating discrete sequence learning.

The Reber Grammar is a finite-state automaton over a 7-symbol alphabet
{B, T, P, S, X, V, E}. Valid sequences always start with B and end with E.
Internal transitions constrain which symbols can appear at each position.

Protocol:
  - Generate N_TRAIN valid sequences, train CategoricalPattern via update()
  - Generate N_TEST valid + N_TEST invalid (random) sequences
  - Measure NLL separation: invalid sequences should score higher NLL than valid ones
  - Report AUROC and mean NLL for each class

The CategoricalPattern learns positional distributions: probs[d, k] = P(symbol k
at position d). Valid sequences share a common positional structure imposed by the
grammar's transition rules; random sequences do not.

Run:
    python benchmarks/reber_grammar.py

Expected result: AUROC > 0.80, NLL separation > 5.0 (on 2000 training sequences)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
from benchmarks.common import print_results_table
from hpm.patterns.categorical import CategoricalPattern
from hpm.patterns.poisson import PoissonPattern

# ---------------------------------------------------------------------------
# Grammar definition
# ---------------------------------------------------------------------------
# Symbols: B=0, T=1, P=2, S=3, X=4, V=5, E=6, PAD=7
SYMBOLS = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
SYM = {s: i for i, s in enumerate(SYMBOLS)}
K_GRAMMAR = len(SYMBOLS)   # 7 grammar symbols
K = K_GRAMMAR + 1          # 8 including PAD
PAD = K_GRAMMAR            # 7

# FSA transitions: state -> list of (symbol, next_state)
# next_state = -1 means terminal (E emitted, then done)
TRANSITIONS = {
    0: [('B', 1)],
    1: [('T', 2), ('P', 3)],
    2: [('S', 2), ('X', 3)],
    3: [('X', 2), ('S', 4)],
    4: [('P', 1), ('V', 5)],
    5: [('V', 5), ('E', -1)],
}

MAX_LEN = 20   # cap sequence generation length; pad shorter sequences
D = MAX_LEN    # feature dimension for CategoricalPattern


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def generate_reber(rng: np.random.Generator, max_steps: int = 100) -> list[int]:
    """Generate one valid Reber Grammar sequence as a list of symbol indices."""
    state = 0
    seq = []
    for _ in range(max_steps):
        choices = TRANSITIONS[state]
        sym_str, next_state = choices[rng.integers(len(choices))]
        seq.append(SYM[sym_str])
        if next_state == -1:
            break
        state = next_state
    return seq


def encode(seq: list[int], max_len: int = MAX_LEN) -> np.ndarray:
    """Pad/truncate sequence to max_len and return as integer array."""
    arr = np.full(max_len, PAD, dtype=np.intp)
    n = min(len(seq), max_len)
    arr[:n] = seq[:n]
    return arr


def generate_invalid(rng: np.random.Generator, length_rng_min: int = 4,
                     length_rng_max: int = 12) -> list[int]:
    """Generate a random sequence (not grammar-constrained) of similar length."""
    length = rng.integers(length_rng_min, length_rng_max + 1)
    return list(rng.integers(0, K_GRAMMAR, size=length))


def encode_counts(seq: list[int]) -> np.ndarray:
    """Count occurrences of each grammar symbol. Returns shape (K_GRAMMAR,) int array.

    Used by the Poisson mode: Poisson(lambda_k) models the rate of symbol k in valid strings.
    e.g. B and E always appear exactly once; S and X appear more due to grammar loops.
    """
    arr = np.zeros(K_GRAMMAR, dtype=np.intp)
    for s in seq:
        if 0 <= s < K_GRAMMAR:
            arr[s] += 1
    return arr


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

N_TRAIN = 2000
N_TEST = 500
RNG_SEED = 42


def run(use_poisson: bool = False) -> dict:
    rng = np.random.default_rng(RNG_SEED)

    if use_poisson:
        enc = encode_counts
        pattern = PoissonPattern(np.ones(K_GRAMMAR))
        dim_label = f"D={K_GRAMMAR} (symbol counts)"
    else:
        enc = encode
        pattern = CategoricalPattern(np.ones((D, K)) / K, K=K)
        dim_label = f"D={D}, K={K} (positional)"

    # --- Train ---
    train_nlls = []
    for _ in range(N_TRAIN):
        seq = generate_reber(rng)
        x = enc(seq)
        nll = pattern.log_prob(x)
        train_nlls.append(nll)
        pattern = pattern.update(x)

    # --- Test: valid sequences ---
    valid_nlls = []
    for _ in range(N_TEST):
        seq = generate_reber(rng)
        x = enc(seq)
        valid_nlls.append(pattern.log_prob(x))

    # --- Test: invalid sequences ---
    invalid_nlls = []
    for _ in range(N_TEST):
        seq = generate_invalid(rng)
        x = enc(seq)
        invalid_nlls.append(pattern.log_prob(x))

    valid_arr = np.array(valid_nlls)
    invalid_arr = np.array(invalid_nlls)

    pairs = invalid_arr[:, None] > valid_arr[None, :]
    auroc = float(pairs.mean())
    separation = float(invalid_arr.mean() - valid_arr.mean())
    mean_train_nll_start = float(train_nlls[0]) if train_nlls else float('nan')
    mean_train_nll_end = float(np.mean(train_nlls[-50:])) if len(train_nlls) >= 50 else float('nan')

    return {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "dim_label": dim_label,
        "valid_nll_mean": float(valid_arr.mean()),
        "valid_nll_std": float(valid_arr.std()),
        "invalid_nll_mean": float(invalid_arr.mean()),
        "invalid_nll_std": float(invalid_arr.std()),
        "separation": separation,
        "auroc": auroc,
        "train_nll_start": mean_train_nll_start,
        "train_nll_end": mean_train_nll_end,
        "n_obs_final": pattern._n_obs,
        "pattern_type": "poisson" if use_poisson else "categorical",
    }


def main():
    parser = argparse.ArgumentParser(description="Reber Grammar benchmark")
    parser.add_argument("--poisson", action="store_true",
                        help="Use PoissonPattern with symbol-count encoding instead of CategoricalPattern")
    args = parser.parse_args()

    label = "PoissonPattern (symbol counts)" if args.poisson else "CategoricalPattern (positional)"
    print(f"Running Reber Grammar benchmark — {label}...")
    m = run(use_poisson=args.poisson)

    passed_sep = m["separation"] > 5.0
    passed_auroc = m["auroc"] > 0.80
    result_label = "✓ PASS" if (passed_sep and passed_auroc) else "✗ FAIL"

    print_results_table(
        title=f"Reber Grammar — {m['pattern_type'].capitalize()}Pattern ({m['dim_label']}, train={m['n_train']})",
        cols=["Metric", "Valid sequences", "Invalid sequences"],
        rows=[
            {"Metric": "Mean NLL",
             "Valid sequences": f"{m['valid_nll_mean']:.2f} ± {m['valid_nll_std']:.2f}",
             "Invalid sequences": f"{m['invalid_nll_mean']:.2f} ± {m['invalid_nll_std']:.2f}"},
            {"Metric": "NLL Separation (invalid − valid)",
             "Valid sequences": f"{m['separation']:.2f}",
             "Invalid sequences": ""},
            {"Metric": "AUROC",
             "Valid sequences": f"{m['auroc']:.3f}",
             "Invalid sequences": ""},
        ],
    )

    print_results_table(
        title="Training convergence",
        cols=["Stage", "Mean NLL (last 50)"],
        rows=[
            {"Stage": "First observation", "Mean NLL (last 50)": f"{m['train_nll_start']:.2f}"},
            {"Stage": "Last 50 train obs",  "Mean NLL (last 50)": f"{m['train_nll_end']:.2f}"},
        ],
    )

    print_results_table(
        title="Result",
        cols=["Criterion", "Threshold", "Actual", "Status"],
        rows=[
            {"Criterion": "NLL Separation", "Threshold": "> 5.0",
             "Actual": f"{m['separation']:.2f}", "Status": "✓" if passed_sep else "✗"},
            {"Criterion": "AUROC", "Threshold": "> 0.80",
             "Actual": f"{m['auroc']:.3f}", "Status": "✓" if passed_auroc else "✗"},
            {"Criterion": "Overall", "Threshold": "both pass",
             "Actual": "", "Status": result_label},
        ],
    )


if __name__ == "__main__":
    main()
