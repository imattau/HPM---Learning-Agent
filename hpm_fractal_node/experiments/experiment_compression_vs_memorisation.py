"""
Experiment 1: Compression vs Memorisation (core HPM test).

Sequence phases:
1) AB stream: A B A B ...
2) ABC stream: A B C A B C ...

The experiment feeds *composite observations* derived from the sliding window
so co-occurrence compression can form AB and then ABC.
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
    dim: int = 4
    seed: int = 3
    ab_steps: int = 30
    abc_steps: int = 30
    compression_cooccurrence_threshold: int = 3
    residual_surprise_threshold: float = 999.0
    tau: float = 2.5
    sigma_diag: float = 2.0
    use_diag: bool = True
    c_scale: float = 5.0


@dataclass
class ExperimentSummary:
    node_counts: list[int]
    ab_first_step: int | None
    abc_first_step: int | None
    ab_reuse_rate: float
    abc_reuse_rate: float


def _make_leaf(node_id: str, mu: np.ndarray, sigma_diag: float, use_diag: bool) -> HFN:
    sigma = np.ones(mu.shape[0]) * sigma_diag if use_diag else np.eye(mu.shape[0]) * sigma_diag
    return HFN(mu=mu, sigma=sigma, id=node_id, use_diag=use_diag)


def _has_ab(node: HFN) -> bool:
    child_ids = {c.id for c in node.children()}
    return "A" in child_ids and "B" in child_ids


def _has_abc(node: HFN) -> bool:
    leaf_ids = _leaf_descendants(node, set())
    return {"A", "B", "C"}.issubset(leaf_ids)


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


def _composites(forest: Forest, predicate) -> list[HFN]:
    return [n for n in forest.active_nodes() if n.children() and predicate(n)]


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)

    forest = Forest(D=cfg.dim, forest_id="compression_vs_memorisation")
    mu_a = np.array([1.0, 0.0, 0.0, 0.0])
    mu_b = np.array([0.0, 1.0, 0.0, 0.0])
    mu_c = np.array([0.0, 0.0, cfg.c_scale, 0.0])
    a = _make_leaf("A", mu_a, cfg.sigma_diag, cfg.use_diag)
    b = _make_leaf("B", mu_b, cfg.sigma_diag, cfg.use_diag)
    c = _make_leaf("C", mu_c, cfg.sigma_diag, cfg.use_diag)
    for n in [a, b, c]:
        forest.register(n)

    obs = Observer(
        forest=forest,
        tau=cfg.tau,
        compression_cooccurrence_threshold=cfg.compression_cooccurrence_threshold,
        residual_surprise_threshold=cfg.residual_surprise_threshold,
        node_use_diag=cfg.use_diag,
    )

    node_counts: list[int] = []
    ab_first_step: int | None = None
    abc_first_step: int | None = None
    ab_reuse = 0
    abc_reuse = 0
    total_ab_obs = 0
    total_abc_obs = 0

    def observe_vec(x: np.ndarray, step: int, phase: str) -> None:
        nonlocal ab_first_step, abc_first_step, ab_reuse, abc_reuse
        res = obs.observe(x)
        node_counts.append(len(forest))

        ab_nodes = _composites(forest, _has_ab)
        if ab_nodes and ab_first_step is None:
            ab_first_step = step

        abc_nodes = _composites(forest, _has_abc)
        if abc_nodes and abc_first_step is None:
            abc_first_step = step

        if phase == "ab":
            if any(_has_ab(n) for n in res.explanation_tree):
                ab_reuse += 1
        else:
            if any(_has_ab(n) for n in res.explanation_tree):
                ab_reuse += 1
            if any(_has_abc(n) for n in res.explanation_tree):
                abc_reuse += 1

    # Phase 1: AB stream (sliding window composite)
    for i in range(cfg.ab_steps):
        token = "A" if i % 2 == 0 else "B"
        vec = (mu_a + mu_b) / 2.0
        observe_vec(vec, i, phase="ab")
        total_ab_obs += 1

    # Phase 2: ABC stream (encourage AB + C co-occurrence when AB exists)
    for j in range(cfg.abc_steps):
        ab_nodes = _composites(forest, _has_ab)
        if ab_nodes:
            vec = (ab_nodes[0].mu + mu_c) / 2.0
        else:
            vec = (mu_a + mu_b + mu_c) / 3.0
        observe_vec(vec, cfg.ab_steps + j, phase="abc")
        total_abc_obs += 1

    ab_reuse_rate = ab_reuse / max(1, total_ab_obs + total_abc_obs)
    abc_reuse_rate = abc_reuse / max(1, total_abc_obs)

    return ExperimentSummary(
        node_counts=node_counts,
        ab_first_step=ab_first_step,
        abc_first_step=abc_first_step,
        ab_reuse_rate=ab_reuse_rate,
        abc_reuse_rate=abc_reuse_rate,
    )


def main() -> int:
    summary = run_experiment()
    print("Experiment 1: Compression vs Memorisation")
    print(f"  AB first step:  {summary.ab_first_step}")
    print(f"  ABC first step: {summary.abc_first_step}")
    print(f"  AB reuse rate:  {summary.ab_reuse_rate:.3f}")
    print(f"  ABC reuse rate: {summary.abc_reuse_rate:.3f}")
    print(f"  Final nodes:    {summary.node_counts[-1] if summary.node_counts else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
