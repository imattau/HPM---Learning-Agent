"""
Experiment 4: Local Density Stress Test

Setup:
- Create a dense cluster of similar nodes.
- Create a sparse region elsewhere.
- Introduce new inputs inside the dense region.

What it tests:
- Whether lacunarity-guided creation suppresses learning globally instead
  of differentiating locally within dense regions.
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


@dataclass
class ExperimentConfig:
    dim: int = 2
    seed: int = 11
    dense_count: int = 16
    dense_spread: float = 0.06
    dense_center: tuple[float, float] = (0.0, 0.0)
    sparse_count: int = 4
    sparse_center: tuple[float, float] = (3.5, 3.5)
    sparse_spread: float = 0.10
    dense_inputs: int = 6
    sparse_inputs: int = 2
    dense_input_offset: tuple[float, float] = (0.20, 0.0)
    dense_input_noise: float = 0.02
    sparse_input_noise: float = 0.06
    residual_surprise_threshold: float = 1.8
    lacunarity_guided_creation: bool = True
    lacunarity_creation_radius: float = 0.12
    lacunarity_creation_factor: float = 2.4
    max_new_nodes: int = 1
    density_sample_size: int = 24


@dataclass
class PhaseStats:
    name: str
    inputs: int
    creations: int
    mean_ratio: float
    max_ratio: float


@dataclass
class ExperimentSummary:
    dense_phase: PhaseStats
    sparse_phase: PhaseStats
    created_nodes: int
    residual_surprise_mean: float
    stagnation: bool
    explosion: bool
    suppression_active: bool


def _make_leaf(node_id: str, mu: np.ndarray, sigma_diag: float) -> HFN:
    return HFN(mu=mu.copy(), sigma=np.ones(mu.shape[0]) * sigma_diag, id=node_id, use_diag=True)


def _seed_cluster(
    forest: Forest,
    rng: np.random.Generator,
    center: np.ndarray,
    count: int,
    spread: float,
    prefix: str,
) -> None:
    for i in range(count):
        mu = center + rng.normal(0.0, spread, size=center.shape[0])
        forest.register(_make_leaf(f"{prefix}_{i}", mu, sigma_diag=0.02))


def _run_phase(
    obs: Observer,
    rng: np.random.Generator,
    inputs: int,
    base: np.ndarray,
    noise: float,
    max_new_nodes: int,
    ratios_out: list[float],
    residuals_out: list[float],
) -> int:
    creations = 0
    for _ in range(inputs):
        x = base + rng.normal(0.0, noise, size=base.shape[0])
        if creations >= max_new_nodes:
            result = obs._expand(x)  # type: ignore[attr-defined]
        else:
            result = obs.observe(x)
        # Density ratio is expensive; use a bounded random sample to keep runtime tight.
        nodes = list(obs.forest.active_nodes())
        if len(nodes) > obs._density_sample_size:
            sample_idx = rng.choice(len(nodes), size=obs._density_sample_size, replace=False)
            nodes = [nodes[i] for i in sample_idx]
        ratio = obs.evaluator.density_ratio(x, nodes, obs.lacunarity_creation_radius)
        ratios_out.append(float(ratio))
        residuals_out.append(float(result.residual_surprise))
        if len(obs.forest) > obs._creation_count:
            creations += 1
            obs._creation_count = len(obs.forest)
            if creations >= max_new_nodes:
                break
    return creations


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)

    forest = Forest(D=cfg.dim, forest_id="local_density_stress")
    dense_center = np.array(cfg.dense_center, dtype=float)
    sparse_center = np.array(cfg.sparse_center, dtype=float)

    _seed_cluster(
        forest,
        rng,
        center=dense_center,
        count=cfg.dense_count,
        spread=cfg.dense_spread,
        prefix="dense",
    )
    _seed_cluster(
        forest,
        rng,
        center=sparse_center,
        count=cfg.sparse_count,
        spread=cfg.sparse_spread,
        prefix="sparse",
    )

    obs = Observer(
        forest=forest,
        tau=1.0,
        residual_surprise_threshold=cfg.residual_surprise_threshold,
        lacunarity_guided_creation=cfg.lacunarity_guided_creation,
        lacunarity_creation_radius=cfg.lacunarity_creation_radius,
        lacunarity_creation_factor=cfg.lacunarity_creation_factor,
        node_use_diag=True,
    )
    obs._creation_count = len(forest)  # type: ignore[attr-defined]
    obs._density_sample_size = cfg.density_sample_size  # type: ignore[attr-defined]

    dense_ratios: list[float] = []
    sparse_ratios: list[float] = []
    residuals: list[float] = []

    dense_base = dense_center + np.array(cfg.dense_input_offset, dtype=float)
    dense_creations = _run_phase(
        obs,
        rng,
        inputs=cfg.dense_inputs,
        base=dense_base,
        noise=cfg.dense_input_noise,
        max_new_nodes=cfg.max_new_nodes,
        ratios_out=dense_ratios,
        residuals_out=residuals,
    )
    # Dense phase is the signal; keep run bounded.
    if dense_creations >= cfg.max_new_nodes:
        sparse_inputs = 0
    else:
        sparse_inputs = cfg.sparse_inputs

    sparse_creations = _run_phase(
        obs,
        rng,
        inputs=sparse_inputs,
        base=sparse_center,
        noise=cfg.sparse_input_noise,
        max_new_nodes=cfg.max_new_nodes,
        ratios_out=sparse_ratios,
        residuals_out=residuals,
    )

    dense_stats = PhaseStats(
        name="dense",
        inputs=cfg.dense_inputs,
        creations=dense_creations,
        mean_ratio=float(np.mean(dense_ratios)) if dense_ratios else 0.0,
        max_ratio=float(np.max(dense_ratios)) if dense_ratios else 0.0,
    )
    sparse_stats = PhaseStats(
        name="sparse",
        inputs=cfg.sparse_inputs,
        creations=sparse_creations,
        mean_ratio=float(np.mean(sparse_ratios)) if sparse_ratios else 0.0,
        max_ratio=float(np.max(sparse_ratios)) if sparse_ratios else 0.0,
    )

    created_nodes = dense_stats.creations + sparse_stats.creations
    residual_mean = float(np.mean(residuals)) if residuals else 0.0
    suppression_active = cfg.lacunarity_guided_creation and cfg.lacunarity_creation_factor <= 0.5
    stagnation = dense_stats.creations == 0 or suppression_active
    explosion = dense_stats.creations > max(1, cfg.max_new_nodes)

    return ExperimentSummary(
        dense_phase=dense_stats,
        sparse_phase=sparse_stats,
        created_nodes=created_nodes,
        residual_surprise_mean=residual_mean,
        stagnation=stagnation,
        explosion=explosion,
        suppression_active=suppression_active,
    )


def _print_summary(summary: ExperimentSummary) -> None:
    print("Experiment 4: Local Density Stress Test")
    print(f"  dense creations:        {summary.dense_phase.creations}")
    print(f"  sparse creations:       {summary.sparse_phase.creations}")
    print(f"  dense mean ratio:       {summary.dense_phase.mean_ratio:.3f}")
    print(f"  sparse mean ratio:      {summary.sparse_phase.mean_ratio:.3f}")
    print(f"  residual mean:          {summary.residual_surprise_mean:.3f}")
    print(f"  created nodes:          {summary.created_nodes}")
    print(f"  stagnation:             {summary.stagnation}")
    print(f"  explosion:              {summary.explosion}")

    if summary.stagnation:
        print("  verdict: BROKEN_STAGNATION (dense region never differentiates)")
    elif summary.explosion:
        print("  verdict: BROKEN_EXPLOSION (dense region creates too many nodes)")
    else:
        print("  verdict: WORKING (local differentiation in dense region)")


def main() -> int:
    summary = run_experiment()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
