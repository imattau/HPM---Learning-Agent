"""
Experiment 2: Absorption as Generalisation.

Setup: create composite variants (ABC, ABD, ABE) and then pressure the system
with observations centered on the shared AB manifold. Absorption should merge
variant nodes into a more general ABX structure without destroying AB core.
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
    dim: int = 5
    seed: int = 13
    warmup_steps: int = 9
    pressure_steps: int = 30
    probe_samples: int = 25
    warmup_noise: float = 0.02
    pressure_noise: float = 0.02
    probe_noise: float = 0.03
    tau: float = 2.0
    absorption_overlap_threshold: float = 0.3
    absorption_miss_threshold: int = 3
    residual_surprise_threshold: float = 999.0
    compression_cooccurrence_threshold: int = 999
    sigma_shared: float = 0.06
    sigma_variant_base: float = 0.12
    sigma_variant_boost: float = 0.24


@dataclass
class ExperimentSummary:
    initial_variant_nodes: int
    final_variant_nodes: int
    absorbed_ids: list[str]
    merge_step: int | None
    generalised_node_id: str | None
    pre_variant_sigma_mean: float
    post_variant_sigma_mean: float
    variant_axis_sigma_gain: float
    pre_shared_sigma_mean: float
    post_shared_sigma_mean: float
    shared_axis_sigma_gain: float
    pre_probe_reuse_rate: float
    post_probe_reuse_rate: float
    created_leaf_nodes: int
    no_generalisation: bool
    overmerged: bool
    lost_shared_core: bool


def _make_leaf(node_id: str, mu: np.ndarray, sigma_diag: float) -> HFN:
    return HFN(mu=mu, sigma=np.ones(mu.shape[0]) * sigma_diag, id=node_id, use_diag=True)


def _make_composite(
    node_id: str,
    children: list[HFN],
    sigma_shared: float,
    sigma_variant_base: float,
    sigma_variant_boost: float,
    variant_idx: int,
) -> HFN:
    mu = np.mean([c.mu for c in children], axis=0)
    sigma = np.ones(mu.shape[0]) * sigma_variant_base
    sigma[0] = sigma_shared
    sigma[1] = sigma_shared
    sigma[variant_idx] = sigma_variant_boost
    node = HFN(mu=mu, sigma=sigma, id=node_id, use_diag=True)
    for child in children:
        node.add_child(child)
    return node


def _leaf_descendants(node: HFN, seen: set[str]) -> set[str]:
    if node.id in seen:
        return set()
    seen.add(node.id)
    if not node.children():
        return {node.id}
    out: set[str] = set()
    for c in node.children():
        out |= _leaf_descendants(c, seen)
    return out


def _is_ab_variant(node: HFN) -> bool:
    leaves = _leaf_descendants(node, set())
    return "A" in leaves and "B" in leaves and any(x in leaves for x in ("C", "D", "E"))


def _variant_nodes(forest: Forest) -> list[HFN]:
    return [n for n in forest.active_nodes() if n.children() and _is_ab_variant(n)]


def _probe_reuse_rate(
    obs: Observer,
    centers: list[np.ndarray],
    rng: np.random.Generator,
    samples: int,
    noise: float,
) -> float:
    hits = 0
    total = 0
    for center in centers:
        for _ in range(samples):
            x = center + rng.normal(0.0, noise, size=center.shape[0])
            res = obs.expand(x)
            if any(_is_ab_variant(n) for n in res.explanation_tree):
                hits += 1
            total += 1
    return hits / max(1, total)


def _sigma_means(node: HFN) -> tuple[float, float]:
    sigma = node.sigma
    shared = float(np.mean(sigma[0:2]))
    variant = float(np.mean(sigma[2:5]))
    return shared, variant


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)

    forest = Forest(D=cfg.dim, forest_id="absorption_as_generalisation")
    mu_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    mu_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    mu_c = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    mu_d = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    mu_e = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    a = _make_leaf("A", mu_a, sigma_diag=0.05)
    b = _make_leaf("B", mu_b, sigma_diag=0.05)
    c = _make_leaf("C", mu_c, sigma_diag=0.05)
    d = _make_leaf("D", mu_d, sigma_diag=0.05)
    e = _make_leaf("E", mu_e, sigma_diag=0.05)
    for n in [a, b, c, d, e]:
        forest.register(n)

    abc = _make_composite(
        "ABC", [a, b, c],
        sigma_shared=cfg.sigma_shared,
        sigma_variant_base=cfg.sigma_variant_base,
        sigma_variant_boost=cfg.sigma_variant_boost,
        variant_idx=2,
    )
    abd = _make_composite(
        "ABD", [a, b, d],
        sigma_shared=cfg.sigma_shared,
        sigma_variant_base=cfg.sigma_variant_base,
        sigma_variant_boost=cfg.sigma_variant_boost,
        variant_idx=3,
    )
    abe = _make_composite(
        "ABE", [a, b, e],
        sigma_shared=cfg.sigma_shared,
        sigma_variant_base=cfg.sigma_variant_base,
        sigma_variant_boost=cfg.sigma_variant_boost,
        variant_idx=4,
    )
    for n in [abc, abd, abe]:
        forest.register(n)

    obs = Observer(
        forest=forest,
        tau=cfg.tau,
        beta_loss=0.0,
        absorption_overlap_threshold=cfg.absorption_overlap_threshold,
        absorption_miss_threshold=cfg.absorption_miss_threshold,
        residual_surprise_threshold=cfg.residual_surprise_threshold,
        compression_cooccurrence_threshold=cfg.compression_cooccurrence_threshold,
        node_use_diag=True,
    )

    initial_variant = _variant_nodes(forest)
    initial_variant_nodes = len(initial_variant)
    pre_shared_vals = []
    pre_variant_vals = []
    for node in initial_variant:
        shared_mean, variant_mean = _sigma_means(node)
        pre_shared_vals.append(shared_mean)
        pre_variant_vals.append(variant_mean)
    pre_shared_sigma_mean = float(np.mean(pre_shared_vals)) if pre_shared_vals else 0.0
    pre_variant_sigma_mean = float(np.mean(pre_variant_vals)) if pre_variant_vals else 0.0

    composite_centers = [abc.mu, abd.mu, abe.mu]

    # Warmup: expose each variant so all three are active explainers
    for i in range(cfg.warmup_steps):
        center = composite_centers[i % len(composite_centers)]
        x = center + rng.normal(0.0, cfg.warmup_noise, size=cfg.dim)
        obs.observe(x)

    pre_probe_reuse_rate = _probe_reuse_rate(
        obs,
        composite_centers,
        rng,
        samples=cfg.probe_samples,
        noise=cfg.probe_noise,
    )

    general_center = np.mean(np.stack(composite_centers, axis=0), axis=0)
    merge_step: int | None = None
    absorbed_before = len(obs.absorbed_ids)
    if absorbed_before > 0:
        merge_step = 0

    for step in range(cfg.pressure_steps):
        x = general_center + rng.normal(0.0, cfg.pressure_noise, size=cfg.dim)
        obs.observe(x)
        if merge_step is None and len(obs.absorbed_ids) > absorbed_before:
            merge_step = step
        absorbed_before = len(obs.absorbed_ids)

    post_probe_reuse_rate = _probe_reuse_rate(
        obs,
        composite_centers,
        rng,
        samples=cfg.probe_samples,
        noise=cfg.probe_noise,
    )

    final_variant = _variant_nodes(forest)
    final_variant_nodes = len(final_variant)

    # Identify generalised node: closest AB* node to general_center
    generalised_node = None
    if final_variant:
        generalised_node = min(final_variant, key=lambda n: float(np.linalg.norm(n.mu - general_center)))
    generalised_node_id = generalised_node.id if generalised_node is not None else None

    if generalised_node is not None:
        post_shared_sigma_mean, post_variant_sigma_mean = _sigma_means(generalised_node)
    else:
        post_shared_sigma_mean = 0.0
        post_variant_sigma_mean = 0.0

    variant_axis_sigma_gain = post_variant_sigma_mean - pre_variant_sigma_mean
    shared_axis_sigma_gain = post_shared_sigma_mean - pre_shared_sigma_mean

    final_ids = {n.id for n in forest.active_nodes()}
    created_leaf_nodes = len([nid for nid in final_ids if nid.startswith("leaf_")])

    lost_shared_core = generalised_node is None or not _is_ab_variant(generalised_node)
    no_generalisation = len(obs.absorbed_ids) == 0 or final_variant_nodes >= initial_variant_nodes
    overmerged = (final_variant_nodes == 1 and post_probe_reuse_rate < pre_probe_reuse_rate)

    return ExperimentSummary(
        initial_variant_nodes=initial_variant_nodes,
        final_variant_nodes=final_variant_nodes,
        absorbed_ids=sorted(obs.absorbed_ids),
        merge_step=merge_step,
        generalised_node_id=generalised_node_id,
        pre_variant_sigma_mean=pre_variant_sigma_mean,
        post_variant_sigma_mean=post_variant_sigma_mean,
        variant_axis_sigma_gain=variant_axis_sigma_gain,
        pre_shared_sigma_mean=pre_shared_sigma_mean,
        post_shared_sigma_mean=post_shared_sigma_mean,
        shared_axis_sigma_gain=shared_axis_sigma_gain,
        pre_probe_reuse_rate=pre_probe_reuse_rate,
        post_probe_reuse_rate=post_probe_reuse_rate,
        created_leaf_nodes=created_leaf_nodes,
        no_generalisation=no_generalisation,
        overmerged=overmerged,
        lost_shared_core=lost_shared_core,
    )


def _print_summary(summary: ExperimentSummary) -> None:
    print("Experiment 2: Absorption as Generalisation")
    print(f"  initial AB* nodes:       {summary.initial_variant_nodes}")
    print(f"  final AB* nodes:         {summary.final_variant_nodes}")
    print(f"  absorbed ids:            {summary.absorbed_ids}")
    print(f"  merge step:              {summary.merge_step}")
    print(f"  generalised node:        {summary.generalised_node_id}")
    print(f"  pre variant sigma mean:  {summary.pre_variant_sigma_mean:.4f}")
    print(f"  post variant sigma mean: {summary.post_variant_sigma_mean:.4f}")
    print(f"  pre shared sigma mean:   {summary.pre_shared_sigma_mean:.4f}")
    print(f"  post shared sigma mean:  {summary.post_shared_sigma_mean:.4f}")
    print(f"  pre probe reuse rate:    {summary.pre_probe_reuse_rate:.3f}")
    print(f"  post probe reuse rate:   {summary.post_probe_reuse_rate:.3f}")
    print(f"  created leaf nodes:      {summary.created_leaf_nodes}")

    if summary.no_generalisation:
        print("  verdict: BROKEN_NO_GENERALISATION")
    elif summary.overmerged or summary.lost_shared_core:
        print("  verdict: BROKEN_OVERMERGE")
    else:
        print("  verdict: WORKING_ABSTRACTION")


def main() -> int:
    summary = run_experiment()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
