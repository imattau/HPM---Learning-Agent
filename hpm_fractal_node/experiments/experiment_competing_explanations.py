"""
Experiment 4: Competing Explanations (critical).

Goal:
- Option A: reuse a simple but incorrect pattern (cheap)
- Option B: choose a complex but correct structure (accurate)

The experiment keeps decision logic local to this harness while using
Evaluator and DecisionPolicy primitives for score/weight evolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.evaluator import Evaluator
from hfn.hfn import HFN
from hfn.policy import DecisionPolicy, LearningPolicyConfig


@dataclass
class ExperimentConfig:
    dim: int = 4
    seed: int = 7
    samples: int = 220
    early_window: int = 40
    late_window: int = 40
    transition_window: int = 12
    transition_threshold: float = 0.7
    premature_window: int = 12
    premature_threshold: float = 0.7
    noise_std: float = 0.08
    lambda_complexity: float = 0.08
    weight_gain: float = 0.80
    alpha_gain: float = 0.10
    beta_loss: float = 0.04
    explainer_relative_accuracy_floor: float = 0.50
    initial_cheap_weight: float = 0.95
    initial_complex_weight: float = 0.15
    cheap_sigma: float = 0.80
    complex_sigma: float = 0.35
    complex_offdiag: float = 0.0
    evidence_gain: float = 0.06
    evidence_decay: float = 0.96
    evidence_scale: float = 0.5


@dataclass
class StepRecord:
    step: int
    decision: str
    cheap_score: float
    complex_score: float
    cheap_weight: float
    complex_weight: float
    cheap_accuracy: float
    complex_accuracy: float


@dataclass
class ExperimentSummary:
    samples: int
    cheap_choice_count: int
    complex_choice_count: int
    transition_step: int | None
    early_cheap_rate: float
    late_complex_rate: float
    cheap_weight_trace: list[float]
    complex_weight_trace: list[float]
    cheap_score_trace: list[float]
    complex_score_trace: list[float]
    stagnation: bool
    premature_complexity: bool
    records: list[StepRecord]


def _make_node(node_id: str, mu: np.ndarray, sigma_diag: float, d: int) -> HFN:
    return HFN(mu=mu.copy(), sigma=np.ones(d) * sigma_diag, id=node_id, use_diag=True)


def _make_complex(node_id: str, mu: np.ndarray, sigma_diag: float, offdiag: float, children: list[HFN]) -> HFN:
    D = mu.shape[0]
    if offdiag <= 0.0:
        node = HFN(mu=mu.copy(), sigma=np.ones(D) * sigma_diag, id=node_id, use_diag=True)
    else:
        sigma = np.eye(D) * sigma_diag
        sigma += (np.ones((D, D)) - np.eye(D)) * offdiag
        node = HFN(mu=mu.copy(), sigma=sigma, id=node_id, use_diag=False)
    for child in children:
        node.add_child(child)
    return node


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)
    evaluator = Evaluator()
    policy = DecisionPolicy(
        learning=LearningPolicyConfig(
            alpha_gain=cfg.alpha_gain,
            beta_loss=cfg.beta_loss,
            lambda_complexity=cfg.lambda_complexity,
            explainer_relative_accuracy_floor=cfg.explainer_relative_accuracy_floor,
        )
    )

    true_center = np.array([0.42, 0.08, 0.08, 0.08])
    cheap_mu = np.array([0.30, 0.0, 0.0, 0.0])
    cheap_node = _make_node("cheap_wrong", cheap_mu, sigma_diag=cfg.cheap_sigma, d=cfg.dim)
    c1 = _make_node("c1", true_center, sigma_diag=cfg.complex_sigma, d=cfg.dim)
    c2 = _make_node("c2", true_center, sigma_diag=cfg.complex_sigma, d=cfg.dim)
    complex_node = _make_complex(
        "complex_correct",
        true_center,
        cfg.complex_sigma,
        cfg.complex_offdiag,
        [c1, c2],
    )

    weights = {
        cheap_node.id: cfg.initial_cheap_weight,
        complex_node.id: cfg.initial_complex_weight,
    }
    records: list[StepRecord] = []
    cheap_choice_count = 0
    complex_choice_count = 0
    cheap_weight_trace: list[float] = []
    complex_weight_trace: list[float] = []
    cheap_score_trace: list[float] = []
    complex_score_trace: list[float] = []
    decisions: list[str] = []
    transition_step: int | None = None

    evidence = 0.0

    for step in range(cfg.samples):
        x = true_center + rng.normal(0.0, cfg.noise_std, size=cfg.dim)
        cheap_accuracy = evaluator.accuracy(x, cheap_node)
        complex_accuracy = evaluator.accuracy(x, complex_node)
        cheap_complexity = evaluator.description_length(cheap_node)
        complex_complexity = evaluator.description_length(complex_node)

        evidence = (evidence * cfg.evidence_decay) + cfg.evidence_gain * max(
            0.0, complex_accuracy - cheap_accuracy
        )

        cheap_score = (
            cheap_accuracy
            + cfg.weight_gain * weights[cheap_node.id]
            - cfg.lambda_complexity * cheap_complexity
        )
        complex_score = (
            complex_accuracy
            + cfg.weight_gain * weights[complex_node.id]
            - cfg.lambda_complexity * complex_complexity
            + cfg.evidence_scale * evidence
        )

        if complex_score > cheap_score:
            decision = "complex"
            complex_choice_count += 1
        else:
            decision = "cheap"
            cheap_choice_count += 1

        decisions.append(decision)

        # Update weights using observer-style "explaining" gating.
        accuracy_scores = {
            cheap_node.id: cheap_accuracy,
            complex_node.id: complex_accuracy,
        }
        explaining_ids = policy.active_explaining_ids(accuracy_scores)
        for node in (cheap_node, complex_node):
            if node.id in explaining_ids:
                weights[node.id] = policy.weight_update(
                    current_weight=weights[node.id],
                    explaining=True,
                    accuracy=accuracy_scores[node.id],
                    overlap_sum=0.0,
                )
            else:
                overlap_sum = 0.0
                for eid in explaining_ids:
                    other_node = cheap_node if eid == cheap_node.id else complex_node
                    overlap_sum += node.overlap(other_node)
                weights[node.id] = policy.weight_update(
                    current_weight=weights[node.id],
                    explaining=False,
                    accuracy=0.0,
                    overlap_sum=overlap_sum,
                )

        records.append(
            StepRecord(
                step=step,
                decision=decision,
                cheap_score=cheap_score,
                complex_score=complex_score,
                cheap_weight=weights[cheap_node.id],
                complex_weight=weights[complex_node.id],
                cheap_accuracy=cheap_accuracy,
                complex_accuracy=complex_accuracy,
            )
        )

        cheap_weight_trace.append(weights[cheap_node.id])
        complex_weight_trace.append(weights[complex_node.id])
        cheap_score_trace.append(cheap_score)
        complex_score_trace.append(complex_score)

        if transition_step is None and len(decisions) >= cfg.transition_window:
            window = decisions[-cfg.transition_window:]
            complex_rate = sum(1 for d in window if d == "complex") / len(window)
            if complex_rate >= cfg.transition_threshold:
                transition_step = step

    early_n = min(cfg.early_window, len(decisions))
    early_cheap_rate = (
        sum(1 for d in decisions[:early_n] if d == "cheap") / float(early_n)
        if early_n > 0
        else 0.0
    )
    late_n = min(cfg.late_window, len(decisions))
    late_complex_rate = (
        sum(1 for d in decisions[-late_n:] if d == "complex") / float(late_n)
        if late_n > 0
        else 0.0
    )
    stagnation = transition_step is None
    premature_window = min(cfg.premature_window, len(decisions))
    premature_rate = (
        sum(1 for d in decisions[:premature_window] if d == "complex") / float(premature_window)
        if premature_window > 0
        else 0.0
    )
    premature_complexity = premature_rate >= cfg.premature_threshold

    return ExperimentSummary(
        samples=cfg.samples,
        cheap_choice_count=cheap_choice_count,
        complex_choice_count=complex_choice_count,
        transition_step=transition_step,
        early_cheap_rate=early_cheap_rate,
        late_complex_rate=late_complex_rate,
        cheap_weight_trace=cheap_weight_trace,
        complex_weight_trace=complex_weight_trace,
        cheap_score_trace=cheap_score_trace,
        complex_score_trace=complex_score_trace,
        stagnation=stagnation,
        premature_complexity=premature_complexity,
        records=records,
    )


def _print_summary(summary: ExperimentSummary) -> None:
    print("Experiment 4: Competing Explanations")
    print(f"  samples:                {summary.samples}")
    print(f"  cheap choice count:     {summary.cheap_choice_count}")
    print(f"  complex choice count:   {summary.complex_choice_count}")
    print(f"  transition step:        {summary.transition_step}")
    print(f"  early cheap rate:       {summary.early_cheap_rate:.3f}")
    print(f"  late complex rate:      {summary.late_complex_rate:.3f}")
    print(f"  stagnation:             {summary.stagnation}")
    print(f"  premature complexity:   {summary.premature_complexity}")

    if summary.stagnation:
        print("  verdict: STAGNATION (never shifted to complex explanation)")
    elif summary.premature_complexity:
        print("  verdict: PREMATURE_COMPLEXITY (skipped cheap early phase)")
    else:
        print("  verdict: BALANCED (cheap early, complex after experience)")


def main() -> int:
    summary = run_experiment()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
