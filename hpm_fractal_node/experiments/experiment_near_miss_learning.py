"""
Experiment 3: Near-Miss Learning (predictive completion).

Train on ABC, then probe AB?. The system should either predict C via completion
or create new structure when the near-miss is too surprising.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.forest import Forest
from hfn.hfn import HFN
from hfn.observer import Observer
from hfn.decoder import Decoder


@dataclass
class ExperimentConfig:
    dim: int = 4
    seed: int = 5
    train_steps: int = 30
    probe_steps: int = 40
    warmup_noise: float = 0.02
    probe_noise: float = 0.03
    tau: float = 2.2
    residual_surprise_threshold: float = 1.6
    compression_cooccurrence_threshold: int = 999
    sigma_leaf: float = 0.6
    sigma_composite: float = 1.2
    sigma_threshold: float = 0.7


@dataclass
class ProbeRecord:
    step: int
    predicted_c: bool
    used_abc: bool
    residual_surprise: float
    created_new_node: bool
    node_count: int


@dataclass
class ExperimentSummary:
    probe_steps: int
    predicted_c_count: int
    prediction_accuracy: float
    first_correct_prediction_step: int | None
    abc_reuse_count: int
    creation_count: int
    reuse_rate: float
    creation_rate: float
    residual_surprise_mean: float
    residual_surprise_by_step: list[float]
    initial_node_count: int
    final_node_count: int
    created_node_ids: list[str]
    records: list[ProbeRecord]


def _make_leaf(node_id: str, mu: np.ndarray, sigma_diag: float) -> HFN:
    return HFN(mu=mu, sigma=np.ones(mu.shape[0]) * sigma_diag, id=node_id, use_diag=True)


def _make_composite(node_id: str, children: list[HFN], sigma_diag: float) -> HFN:
    mu = np.mean([c.mu for c in children], axis=0)
    node = HFN(mu=mu, sigma=np.ones(mu.shape[0]) * sigma_diag, id=node_id, use_diag=True)
    for child in children:
        node.add_child(child)
    return node


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)

    forest = Forest(D=cfg.dim, forest_id="near_miss_learning")
    mu_a = np.array([1.0, 0.0, 0.0, 0.0])
    mu_b = np.array([0.0, 1.0, 0.0, 0.0])
    mu_c = np.array([0.0, 0.0, 1.0, 0.0])

    a = _make_leaf("A", mu_a, cfg.sigma_leaf)
    b = _make_leaf("B", mu_b, cfg.sigma_leaf)
    c = _make_leaf("C", mu_c, cfg.sigma_leaf)
    for n in [a, b, c]:
        forest.register(n)

    abc = _make_composite("ABC", [a, b, c], sigma_diag=cfg.sigma_composite)
    forest.register(abc)

    decoder = Decoder(target_forest=forest, sigma_threshold=cfg.sigma_threshold)
    obs = Observer(
        forest=forest,
        tau=cfg.tau,
        residual_surprise_threshold=cfg.residual_surprise_threshold,
        compression_cooccurrence_threshold=cfg.compression_cooccurrence_threshold,
        node_use_diag=True,
        decoder=decoder,
    )

    initial_node_count = len(forest)

    # Train on ABC
    for _ in range(cfg.train_steps):
        x = abc.mu + rng.normal(0.0, cfg.warmup_noise, size=cfg.dim)
        obs.observe(x)

    # Probe on AB?
    x_partial = (mu_a + mu_b) / 2.0
    predicted_c_count = 0
    abc_reuse_count = 0
    creation_count = 0
    residuals: list[float] = []
    created_node_ids: list[str] = []
    records: list[ProbeRecord] = []
    first_correct: int | None = None

    for step in range(cfg.probe_steps):
        x = x_partial + rng.normal(0.0, cfg.probe_noise, size=cfg.dim)

        expand_res = obs._expand(x)  # type: ignore[attr-defined]
        pred_mu = obs.predict(expand_res)
        predicted_c = False
        if pred_mu is not None:
            predicted_c = np.linalg.norm(pred_mu - mu_c) < 0.25
        if predicted_c:
            predicted_c_count += 1
            if first_correct is None:
                first_correct = step

        used_abc = any(n.id == "ABC" for n in expand_res.explanation_tree)
        if used_abc:
            abc_reuse_count += 1

        before_ids = {n.id for n in forest.active_nodes()}
        learn_res = obs.observe(x)
        residuals.append(float(learn_res.residual_surprise))
        after_ids = {n.id for n in forest.active_nodes()}
        new_ids = sorted(after_ids - before_ids)
        created_new = False
        for nid in new_ids:
            if nid.startswith(obs.node_prefix):
                created_new = True
                creation_count += 1
                created_node_ids.append(nid)

        records.append(
            ProbeRecord(
                step=step,
                predicted_c=predicted_c,
                used_abc=used_abc,
                residual_surprise=float(learn_res.residual_surprise),
                created_new_node=created_new,
                node_count=len(forest),
            )
        )

    prediction_accuracy = predicted_c_count / max(1, cfg.probe_steps)
    reuse_rate = abc_reuse_count / max(1, cfg.probe_steps)
    creation_rate = creation_count / max(1, cfg.probe_steps)
    residual_mean = float(np.mean(residuals)) if residuals else 0.0

    return ExperimentSummary(
        probe_steps=cfg.probe_steps,
        predicted_c_count=predicted_c_count,
        prediction_accuracy=prediction_accuracy,
        first_correct_prediction_step=first_correct,
        abc_reuse_count=abc_reuse_count,
        creation_count=creation_count,
        reuse_rate=reuse_rate,
        creation_rate=creation_rate,
        residual_surprise_mean=residual_mean,
        residual_surprise_by_step=residuals,
        initial_node_count=initial_node_count,
        final_node_count=len(forest),
        created_node_ids=created_node_ids,
        records=records,
    )


def _print_summary(summary: ExperimentSummary) -> None:
    print("Experiment 3: Near-Miss Learning")
    print(f"  probe steps:             {summary.probe_steps}")
    print(f"  predicted C count:       {summary.predicted_c_count}")
    print(f"  prediction accuracy:     {summary.prediction_accuracy:.3f}")
    print(f"  first correct step:      {summary.first_correct_prediction_step}")
    print(f"  ABC reuse count:         {summary.abc_reuse_count}")
    print(f"  reuse rate:              {summary.reuse_rate:.3f}")
    print(f"  creation count:          {summary.creation_count}")
    print(f"  creation rate:           {summary.creation_rate:.3f}")
    print(f"  residual mean:           {summary.residual_surprise_mean:.3f}")
    print(f"  initial nodes:           {summary.initial_node_count}")
    print(f"  final nodes:             {summary.final_node_count}")

    if summary.predicted_c_count == 0 and summary.creation_count == 0:
        print("  verdict: BROKEN_STAGNATION")
    elif summary.predicted_c_count == 0:
        print("  verdict: BROKEN_NO_COMPLETION")
    elif summary.creation_rate > 0.6:
        print("  verdict: BROKEN_CREATION_DOMINATED")
    else:
        print("  verdict: WORKING_PREDICTIVE_COMPLETION")


def main() -> int:
    summary = run_experiment()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
