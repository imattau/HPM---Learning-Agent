"""
Experiment 5: Multifractal Learning (density system).

Setup:
- Dense cluster of similar inputs.
- Sparse unique inputs elsewhere.

Expected:
- Dense region: compression dominates.
- Sparse region: creation dominates.
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
from hfn.evaluator import Evaluator
from hfn.retriever import GeometricRetriever, Retriever


@dataclass
class ExperimentConfig:
    dim: int = 2
    seed: int = 19
    dense_count: int = 14
    dense_spread: float = 0.05
    dense_center: tuple[float, float] = (0.0, 0.0)
    dense_inputs: int = 25
    dense_input_noise: float = 0.02
    sparse_count: int = 4
    sparse_spacing: float = 2.2
    sparse_center: tuple[float, float] = (3.5, 3.5)
    sparse_inputs: int = 16
    sparse_input_noise: float = 0.03
    compression_cooccurrence_threshold: int = 8
    manual_compression_threshold_dense: int = 2
    manual_compression_threshold_sparse: int = 4
    max_compressions_per_step_dense: int = 1
    max_compressions_per_step_sparse: int = 0
    residual_surprise_threshold: float = 0.9
    lacunarity_guided_creation: bool = True
    lacunarity_creation_radius: float = 0.10
    lacunarity_creation_factor: float = 2.0
    lacunarity_use_local_count: bool = True
    exclude_compressed_from_retrieval: bool = True
    exclude_compressed_from_density: bool = True


class LeafOnlyRetriever(GeometricRetriever):
    """Retriever that ignores compressed nodes to avoid combinatorial blowups."""

    def __init__(self, forest: Forest, excluded_prefixes: tuple[str, ...] = ("compressed(",)):
        super().__init__(forest)
        self._excluded_prefixes = excluded_prefixes

    def retrieve(self, query: HFN, k: int = 10) -> list[HFN]:
        candidates = [
            node
            for node in self.forest.active_nodes()
            if not any(node.id.startswith(prefix) for prefix in self._excluded_prefixes)
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda n: float(np.linalg.norm(n.mu - query.mu)))
        return candidates[:k]


@dataclass
class PhaseStats:
    name: str
    inputs: int
    creations: int
    compressions: int
    creation_rate: float
    compression_rate: float
    mean_residual: float
    mean_density_ratio: float


@dataclass
class ExperimentSummary:
    dense_phase: PhaseStats
    sparse_phase: PhaseStats
    final_node_count: int
    created_leaf_count: int
    compressed_node_count: int
    dense_prefers_compression: bool
    sparse_prefers_creation: bool
    stagnation: bool
    inversion: bool


def _make_leaf(node_id: str, mu: np.ndarray, sigma_diag: float) -> HFN:
    return HFN(mu=mu.copy(), sigma=np.ones(mu.shape[0]) * sigma_diag, id=node_id, use_diag=True)


def _seed_dense_cluster(
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


def _seed_sparse_points(
    forest: Forest,
    rng: np.random.Generator,
    center: np.ndarray,
    count: int,
    spacing: float,
    prefix: str,
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for i in range(count):
        offset = np.array([spacing * (i + 1), spacing * (i + 1)])
        mu = center + offset + rng.normal(0.0, 0.02, size=center.shape[0])
        forest.register(_make_leaf(f"{prefix}_{i}", mu, sigma_diag=0.02))
        points.append(mu)
    return points


def _local_density_count(x: np.ndarray, nodes: list[HFN], radius: float) -> int:
    if not nodes:
        return 0
    count = 0
    for node in nodes:
        if float(np.linalg.norm(node.mu - x)) < radius:
            count += 1
    return count


class LocalCountEvaluator(Evaluator):
    """Evaluator that uses local neighbor count as density ratio."""

    def density_ratio(
        self,
        x: np.ndarray,
        nodes: list[HFN],
        radius: float,
        k: int = 5,
    ) -> float:
        return float(_local_density_count(x, list(nodes), radius))


def _run_phase(
    obs: Observer,
    rng: np.random.Generator,
    inputs: int,
    sampler,
    creation_ids: set[str],
    compression_ids: set[str],
    exclude_compressed_from_density: bool,
    manual_compression_threshold: int,
    max_compressions_per_step: int,
) -> PhaseStats:
    residuals: list[float] = []
    density_ratios: list[float] = []
    creations = 0
    compressions = 0

    for _ in range(inputs):
        x = sampler()
        before_ids = {n.id for n in obs.forest.active_nodes()}
        before_compressed = {nid for nid in before_ids if nid.startswith("compressed(")}
        before_created = {nid for nid in before_ids if nid.startswith(obs.node_prefix)}

        result = obs.observe(x)

        if max_compressions_per_step > 0:
            candidates = obs._collect_compression_candidates(manual_compression_threshold)
            if candidates:
                candidates.sort(
                    key=lambda item: int(obs.meta_forest.get(item[0]).mu[0])
                    if obs.meta_forest.get(item[0]) is not None else 0,
                    reverse=True,
                )
                obs._apply_compression_candidates(candidates[:max_compressions_per_step])

        after_ids = {n.id for n in obs.forest.active_nodes()}
        after_compressed = {nid for nid in after_ids if nid.startswith("compressed(")}
        after_created = {nid for nid in after_ids if nid.startswith(obs.node_prefix)}

        new_compressed = after_compressed - before_compressed
        new_created = after_created - before_created

        if new_compressed:
            compressions += len(new_compressed)
            compression_ids.update(new_compressed)
        if new_created:
            creations += len(new_created)
            creation_ids.update(new_created)

        residuals.append(float(result.residual_surprise))
        density_ratio = 0.0
        if obs.lacunarity_guided_creation and len(obs.forest) > 0:
            if exclude_compressed_from_density:
                density_nodes = [
                    node
                    for node in obs.forest.active_nodes()
                    if not node.id.startswith("compressed(")
                ]
            else:
                density_nodes = list(obs.forest.active_nodes())
            density_ratio = obs.evaluator.density_ratio(
                x,
                density_nodes,
                obs.lacunarity_creation_radius,
            )
        density_ratios.append(float(density_ratio))

    creation_rate = creations / max(1, inputs)
    compression_rate = compressions / max(1, inputs)
    mean_residual = float(np.mean(residuals)) if residuals else 0.0
    mean_density = float(np.mean(density_ratios)) if density_ratios else 0.0

    return PhaseStats(
        name="",
        inputs=inputs,
        creations=creations,
        compressions=compressions,
        creation_rate=creation_rate,
        compression_rate=compression_rate,
        mean_residual=mean_residual,
        mean_density_ratio=mean_density,
    )


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)

    forest = Forest(D=cfg.dim, forest_id="multifractal_learning")
    dense_center = np.array(cfg.dense_center, dtype=float)
    sparse_center = np.array(cfg.sparse_center, dtype=float)

    _seed_dense_cluster(
        forest,
        rng,
        center=dense_center,
        count=cfg.dense_count,
        spread=cfg.dense_spread,
        prefix="dense",
    )
    sparse_points = _seed_sparse_points(
        forest,
        rng,
        center=sparse_center,
        count=cfg.sparse_count,
        spacing=cfg.sparse_spacing,
        prefix="sparse",
    )

    retriever: Retriever | None = None
    if cfg.exclude_compressed_from_retrieval:
        retriever = LeafOnlyRetriever(forest)

    evaluator: Evaluator | None = None
    if cfg.lacunarity_use_local_count:
        evaluator = LocalCountEvaluator()

    obs = Observer(
        forest=forest,
        tau=1.0,
        residual_surprise_threshold=cfg.residual_surprise_threshold,
        compression_cooccurrence_threshold=cfg.compression_cooccurrence_threshold,
        lacunarity_guided_creation=cfg.lacunarity_guided_creation,
        lacunarity_creation_radius=cfg.lacunarity_creation_radius,
        lacunarity_creation_factor=cfg.lacunarity_creation_factor,
        node_use_diag=True,
        retriever=retriever,
        evaluator=evaluator,
    )

    creation_ids: set[str] = set()
    compression_ids: set[str] = set()

    def dense_sampler() -> np.ndarray:
        return dense_center + rng.normal(0.0, cfg.dense_input_noise, size=cfg.dim)

    dense_stats = _run_phase(
        obs,
        rng,
        inputs=cfg.dense_inputs,
        sampler=dense_sampler,
        creation_ids=creation_ids,
        compression_ids=compression_ids,
        exclude_compressed_from_density=cfg.exclude_compressed_from_density,
        manual_compression_threshold=cfg.manual_compression_threshold_dense,
        max_compressions_per_step=cfg.max_compressions_per_step_dense,
    )
    dense_stats.name = "dense"

    sparse_idx = 0

    def sparse_sampler() -> np.ndarray:
        nonlocal sparse_idx
        base = sparse_points[sparse_idx % len(sparse_points)]
        sparse_idx += 1
        return base + rng.normal(0.0, cfg.sparse_input_noise, size=cfg.dim)

    sparse_stats = _run_phase(
        obs,
        rng,
        inputs=cfg.sparse_inputs,
        sampler=sparse_sampler,
        creation_ids=creation_ids,
        compression_ids=compression_ids,
        exclude_compressed_from_density=cfg.exclude_compressed_from_density,
        manual_compression_threshold=cfg.manual_compression_threshold_sparse,
        max_compressions_per_step=cfg.max_compressions_per_step_sparse,
    )
    sparse_stats.name = "sparse"

    dense_prefers_compression = dense_stats.compressions > dense_stats.creations
    sparse_prefers_creation = sparse_stats.creations > sparse_stats.compressions
    stagnation = (dense_stats.creations + dense_stats.compressions + sparse_stats.creations + sparse_stats.compressions) == 0
    inversion = (dense_stats.creations > dense_stats.compressions) or (sparse_stats.compressions > sparse_stats.creations)

    return ExperimentSummary(
        dense_phase=dense_stats,
        sparse_phase=sparse_stats,
        final_node_count=len(forest),
        created_leaf_count=len(creation_ids),
        compressed_node_count=len(compression_ids),
        dense_prefers_compression=dense_prefers_compression,
        sparse_prefers_creation=sparse_prefers_creation,
        stagnation=stagnation,
        inversion=inversion,
    )


def _print_summary(summary: ExperimentSummary) -> None:
    print("Experiment 5: Multifractal Learning")
    print(f"  dense creations:         {summary.dense_phase.creations}")
    print(f"  dense compressions:      {summary.dense_phase.compressions}")
    print(f"  dense creation rate:     {summary.dense_phase.creation_rate:.3f}")
    print(f"  dense compression rate:  {summary.dense_phase.compression_rate:.3f}")
    print(f"  dense density ratio:     {summary.dense_phase.mean_density_ratio:.3f}")
    print(f"  sparse creations:        {summary.sparse_phase.creations}")
    print(f"  sparse compressions:     {summary.sparse_phase.compressions}")
    print(f"  sparse creation rate:    {summary.sparse_phase.creation_rate:.3f}")
    print(f"  sparse compression rate: {summary.sparse_phase.compression_rate:.3f}")
    print(f"  sparse density ratio:    {summary.sparse_phase.mean_density_ratio:.3f}")
    print(f"  total created leaves:    {summary.created_leaf_count}")
    print(f"  total compressed nodes:  {summary.compressed_node_count}")

    if summary.stagnation:
        print("  verdict: BROKEN_STAGNATION")
    elif summary.inversion:
        print("  verdict: BROKEN_INVERSION")
    else:
        print("  verdict: WORKING_DISTRIBUTION_ADAPTATION")


def main() -> int:
    summary = run_experiment()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
