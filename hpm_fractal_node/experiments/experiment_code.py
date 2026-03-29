"""
Experiment: Python Code Token Prediction with Gap-Driven Stdlib Queries.

Runs the HPM Observer on synthetic Python code snippet observations,
with QueryStdlib + ConverterCode for gap-driven knowledge injection.

Reports:
- Category purity for syntax/type/builtin/pattern categories
- Query HFN count
- Knowledge Gap HFN count
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from hfn import Observer, calibrate_tau
from hpm_fractal_node.code.code_world_model import build_code_world_model
from hpm_fractal_node.code.code_loader import (
    D, VOCAB, generate_code_snippets,
)
from hpm_fractal_node.code.query_stdlib import QueryStdlib
from hpm_fractal_node.code.converter_code import ConverterCode


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SAMPLES = 2000
N_PASSES = 3
SEED = 42


# ---------------------------------------------------------------------------
# Category purity helpers
# ---------------------------------------------------------------------------

_CATEGORY_TOKENS: dict[str, set[str]] = {
    "syntax": {
        "if", "elif", "else", "for", "while", "break", "continue", "pass",
        "def", "class", "return", "lambda", "yield",
        "import", "from", "as", "with", "try", "except",
        "and", "or", "not", "in", "is", "assert", "raise", "del", "global",
        "=", "==", "!=", "<", ">", "<=", ">=",
        "+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "~",
        "(", ")", ":", ",", ".", "[", "]", "{", "}",
    },
    "type": {"int", "str", "float", "bool", "list"},
    "builtin": {
        "print", "len", "range", "type", "input", "open",
        "map", "filter", "zip", "enumerate",
    },
    "pattern": {
        "for", "if", "def", "=",  # overlap is OK for experiment
    },
}


def token_to_category(token: str) -> str:
    if token in _CATEGORY_TOKENS["builtin"]:
        return "builtin"
    if token in _CATEGORY_TOKENS["type"]:
        return "type"
    if token in _CATEGORY_TOKENS["pattern"]:
        return "pattern"
    if token in _CATEGORY_TOKENS["syntax"]:
        return "syntax"
    return "unknown"


def compute_purity(
    obs_list: list[tuple[np.ndarray, str, str]],
    observer: Observer,
) -> dict[str, float]:
    """
    For each category, compute purity = fraction of observations where the
    top-weight explaining node's true category matches the observation category.
    """
    category_correct: dict[str, int] = defaultdict(int)
    category_total: dict[str, int] = defaultdict(int)

    for vec, true_token, obs_category in obs_list:
        result = observer._expand(vec)
        if not result.explanation_tree:
            continue

        # Best explaining node by weight
        best_node = max(
            result.explanation_tree,
            key=lambda n: observer.get_weight(n.id),
        )
        predicted_category = token_to_category(
            best_node.id.replace("word_", "")
        )

        category_total[obs_category] += 1
        if predicted_category == obs_category:
            category_correct[obs_category] += 1

    purity: dict[str, float] = {}
    for cat in ["syntax", "type", "builtin", "pattern", "control_flow", "functions", "data", "builtins"]:
        total = category_total.get(cat, 0)
        correct = category_correct.get(cat, 0)
        if total > 0:
            purity[cat] = correct / total
    return purity


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> None:
    print("=" * 60)
    print("Code Token Prediction Experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print("=" * 60)

    # Build world model
    forest, prior_nodes = build_code_world_model()
    prior_ids = {n.id for n in prior_nodes}
    print(f"World model: {len(forest)} nodes ({len(prior_ids)} priors)")

    # Calibrate tau
    tau = calibrate_tau(D, sigma_scale=1.0, margin=1.0)

    # Create Observer with Query/Converter
    observer = Observer(
        forest=forest,
        tau=tau,
        budget=10,
        protected_ids=prior_ids,
        query=QueryStdlib(max_results=10),
        converter=ConverterCode(),
        gap_query_threshold=0.7,
        max_expand_depth=2,
        vocab=VOCAB,
    )
    print(f"Observer created (tau={tau:.3f})")

    # Generate observations
    obs_list = generate_code_snippets(seed=SEED)
    # Trim or pad to N_SAMPLES
    if len(obs_list) < N_SAMPLES:
        rng = np.random.default_rng(SEED)
        extra_idx = rng.integers(0, len(obs_list), N_SAMPLES - len(obs_list))
        extra = [obs_list[i] for i in extra_idx]
        obs_list = obs_list + extra
    obs_list = obs_list[:N_SAMPLES]
    print(f"Observations: {len(obs_list)}")

    # 3-pass observation loop
    for pass_num in range(1, N_PASSES + 1):
        rng = np.random.default_rng(SEED + pass_num)
        perm = rng.permutation(len(obs_list))
        for i, idx in enumerate(perm):
            vec, true_token, category = obs_list[idx]
            observer.observe(vec)

        n_query = sum(1 for nid in forest if nid.startswith("query_"))
        n_gap = sum(1 for nid in forest if nid.startswith("gap_"))
        print(f"Pass {pass_num}: forest={len(forest)}, query_nodes={n_query}, gap_nodes={n_gap}")

    # Category purity
    print("\nCategory purity:")
    purity = compute_purity(obs_list[:200], observer)
    for cat, p in sorted(purity.items()):
        print(f"  {cat:12s}: {p:.3f}")

    # Count query/gap HFNs
    query_count = sum(1 for nid in forest if nid.startswith("query_"))
    gap_count = sum(1 for nid in forest if nid.startswith("gap_"))
    sig_count = sum(1 for nid in forest if nid.startswith("sig_"))
    print(f"\nQuery HFN count:        {query_count}")
    print(f"Signature HFN count:    {sig_count}")
    print(f"Knowledge Gap HFN count:{gap_count}")
    print(f"Total forest size:      {len(forest)}")
    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    run_experiment()
