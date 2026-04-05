"""
Experiment 3: Competing Explanations (critical).

Goal:
- Option A: reuse a known node (cheap, slightly wrong)
- Option B: build a new node (expensive, more correct)

The experiment keeps decision logic local to this harness. Core HFN/Observer
modules are not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.evaluator import Evaluator
from hfn.forest import Forest
from hfn.hfn import HFN


@dataclass
class ExperimentConfig:
    dim: int = 2
    seed: int = 7
    samples: int = 220
    early_window: int = 40
    noise_std: float = 0.08
    lambda_complexity: float = 0.03
    structural_penalty: float = 0.05
    mismatch_threshold: float = 0.22
    pressure_increment: float = 1.0
    pressure_decay: float = 0.97
    pressure_gain: float = 0.010
    cheap_pressure_penalty_gain: float = 0.050
    base_creation_cost: float = 0.55
    min_creation_cost: float = 0.10
    max_new_nodes: int = 2


@dataclass
class StepRecord:
    step: int
    decision: str
    cheap_error: float
    chosen_error: float
    pressure_evidence: float
    create_cost_effective: float


@dataclass
class ExperimentSummary:
    samples: int
    cheap_reuse_count: int
    new_reuse_count: int
    creation_step: int | None
    created_nodes: int
    early_reuse_rate: float
    pre_creation_error_mean: float
    post_creation_error_mean: float
    stagnation: bool
    explosion: bool
    records: list[StepRecord]


def _make_node(node_id: str, mu: np.ndarray, sigma_diag: float, d: int) -> HFN:
    return HFN(mu=mu.copy(), sigma=np.ones(d) * sigma_diag, id=node_id, use_diag=True)


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)
    evaluator = Evaluator()

    forest = Forest(D=cfg.dim, forest_id="competing_explanations")
    cheap_node = _make_node("cheap_known", np.array([0.0, 0.0]), sigma_diag=0.32, d=cfg.dim)
    forest.register(cheap_node)

    # Ground truth cluster lives away from cheap_known: reusable but wrong early.
    true_center = np.array([0.45, 0.0])

    records: list[StepRecord] = []
    cheap_reuse_count = 0
    new_reuse_count = 0
    created_nodes = 0
    creation_step: int | None = None
    pressure_evidence = 0.0
    precise_node: HFN | None = None

    for step in range(cfg.samples):
        x = true_center + rng.normal(0.0, cfg.noise_std, size=cfg.dim)
        cheap_error = float(np.linalg.norm(x - cheap_node.mu))
        cheap_fit = evaluator.accuracy(x, cheap_node)
        cheap_utility = cheap_fit - (cfg.lambda_complexity * evaluator.description_length(cheap_node))

        if cheap_error > cfg.mismatch_threshold:
            pressure_evidence += cfg.pressure_increment
        pressure_evidence *= cfg.pressure_decay

        create_cost_effective = max(
            cfg.min_creation_cost,
            cfg.base_creation_cost - (cfg.pressure_gain * pressure_evidence),
        )
        cheap_utility -= (
            cfg.cheap_pressure_penalty_gain
            * pressure_evidence
            * max(0.0, cheap_error - cfg.mismatch_threshold)
        )

        if precise_node is None:
            candidate_precise = _make_node("candidate_precise", x, sigma_diag=0.035, d=cfg.dim)
            build_fit = evaluator.accuracy(x, candidate_precise)
            build_utility = (
                build_fit
                - (cfg.lambda_complexity * evaluator.description_length(candidate_precise))
                - cfg.structural_penalty
                - create_cost_effective
            )
            if build_utility > cheap_utility and created_nodes < cfg.max_new_nodes:
                created_nodes += 1
                creation_step = step
                precise_node = _make_node(f"precise_{created_nodes}", x, sigma_diag=0.035, d=cfg.dim)
                forest.register(precise_node)
                decision = "build_new"
                chosen_error = float(np.linalg.norm(x - precise_node.mu))
                new_reuse_count += 1
            else:
                decision = "reuse_cheap"
                chosen_error = cheap_error
                cheap_reuse_count += 1
        else:
            precise_fit = evaluator.accuracy(x, precise_node)
            precise_utility = precise_fit - (
                cfg.lambda_complexity * evaluator.description_length(precise_node)
            )
            if precise_utility >= cheap_utility:
                decision = "reuse_new"
                chosen_error = float(np.linalg.norm(x - precise_node.mu))
                new_reuse_count += 1
            else:
                decision = "reuse_cheap"
                chosen_error = cheap_error
                cheap_reuse_count += 1

        records.append(
            StepRecord(
                step=step,
                decision=decision,
                cheap_error=cheap_error,
                chosen_error=chosen_error,
                pressure_evidence=pressure_evidence,
                create_cost_effective=create_cost_effective,
            )
        )

    early_n = min(cfg.early_window, len(records))
    early_reuse_rate = (
        sum(1 for r in records[:early_n] if r.decision == "reuse_cheap") / float(early_n)
        if early_n > 0
        else 0.0
    )

    if creation_step is None:
        pre_creation_errors = [r.chosen_error for r in records]
        post_creation_errors: list[float] = []
    else:
        pre_creation_errors = [r.chosen_error for r in records[:creation_step]]
        post_creation_errors = [r.chosen_error for r in records[creation_step + 1 :]]

    pre_creation_error_mean = (
        float(np.mean(pre_creation_errors)) if pre_creation_errors else 0.0
    )
    post_creation_error_mean = (
        float(np.mean(post_creation_errors)) if post_creation_errors else pre_creation_error_mean
    )

    stagnation = creation_step is None
    explosion = created_nodes > 1

    return ExperimentSummary(
        samples=cfg.samples,
        cheap_reuse_count=cheap_reuse_count,
        new_reuse_count=new_reuse_count,
        creation_step=creation_step,
        created_nodes=created_nodes,
        early_reuse_rate=early_reuse_rate,
        pre_creation_error_mean=pre_creation_error_mean,
        post_creation_error_mean=post_creation_error_mean,
        stagnation=stagnation,
        explosion=explosion,
        records=records,
    )


def _print_summary(summary: ExperimentSummary) -> None:
    print("Experiment 3: Competing Explanations")
    print(f"  samples:                {summary.samples}")
    print(f"  cheap reuse count:      {summary.cheap_reuse_count}")
    print(f"  new-structure reuse:    {summary.new_reuse_count}")
    print(f"  creation step:          {summary.creation_step}")
    print(f"  created nodes:          {summary.created_nodes}")
    print(f"  early cheap reuse rate: {summary.early_reuse_rate:.3f}")
    print(f"  error mean (pre):       {summary.pre_creation_error_mean:.4f}")
    print(f"  error mean (post):      {summary.post_creation_error_mean:.4f}")
    print(f"  stagnation:             {summary.stagnation}")
    print(f"  explosion:              {summary.explosion}")

    if summary.stagnation:
        print("  verdict: STAGNATION (always picked cheap reuse)")
    elif summary.explosion:
        print("  verdict: EXPLOSION (kept creating new structure)")
    else:
        print("  verdict: BALANCED (cheap early, new structure under pressure)")


def main() -> int:
    summary = run_experiment()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
