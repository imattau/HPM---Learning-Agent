
"""Reber comparison benchmark using the existing multi-agent benchmark as a thin adapter."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import benchmarks.multi_agent_reber_grammar as reber_bench
from benchmarks.completion_adapter import make_orchestrator_factory, patch_attr

REBER_OVERRIDES = dict(
    evaluator_arbitration_mode="adaptive",
    meta_evaluator_learning_rate=0.2,
    lifecycle_decay_rate=0.08,
    lifecycle_consolidation_window=2,
    lifecycle_absence_window=2,
    lifecycle_stable_weight_threshold=0.2,
    lifecycle_retire_weight_threshold=0.04,
)


def _run_condition(use_poisson: bool, condition: str, n_train: int | None = None, n_test: int | None = None):
    base_factory = reber_bench.make_orchestrator
    if condition == "completion":
        base_factory = make_orchestrator_factory(base_factory, seed=None, overrides=REBER_OVERRIDES)
    with patch_attr(reber_bench, "make_orchestrator", base_factory):
        with patch_attr(reber_bench, "N_TRAIN", n_train if n_train is not None else reber_bench.N_TRAIN):
            with patch_attr(reber_bench, "N_TEST", n_test if n_test is not None else reber_bench.N_TEST):
                return reber_bench.run(use_poisson=use_poisson)


def run(use_poisson: bool = False, n_train: int | None = None, n_test: int | None = None) -> dict[str, dict[str, float]]:
    baseline = _run_condition(use_poisson, "baseline", n_train=n_train, n_test=n_test)
    completion = _run_condition(use_poisson, "completion", n_train=n_train, n_test=n_test)
    return {
        "baseline": baseline,
        "completion": completion,
        "delta": {
            "solo_auroc": completion["solo_auroc"] - baseline["solo_auroc"],
            "multi_auroc": completion["multi_auroc"] - baseline["multi_auroc"],
            "solo_sep": completion["solo_sep"] - baseline["solo_sep"],
            "multi_sep": completion["multi_sep"] - baseline["multi_sep"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Reber comparison benchmark for baseline vs completion-aware agents")
    parser.add_argument("--poisson", action="store_true")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    args = parser.parse_args()

    print("Running Reber completion comparison...", flush=True)
    result = run(use_poisson=args.poisson, n_train=args.n_train, n_test=args.n_test)
    base = result["baseline"]
    comp = result["completion"]

    print(f"Baseline:   solo_auroc={base['solo_auroc']:.3f} multi_auroc={base['multi_auroc']:.3f} solo_sep={base['solo_sep']:.2f} multi_sep={base['multi_sep']:.2f}")
    print(f"Completion: solo_auroc={comp['solo_auroc']:.3f} multi_auroc={comp['multi_auroc']:.3f} solo_sep={comp['solo_sep']:.2f} multi_sep={comp['multi_sep']:.2f}")
    print(f"Delta:      solo_auroc={result['delta']['solo_auroc']:+.3f} multi_auroc={result['delta']['multi_auroc']:+.3f} solo_sep={result['delta']['solo_sep']:+.2f} multi_sep={result['delta']['multi_sep']:+.2f}")


if __name__ == "__main__":
    main()
