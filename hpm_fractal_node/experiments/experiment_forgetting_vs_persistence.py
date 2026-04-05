"""
Experiment 6: Forgetting vs Persistence.

Phase 1: ABC repeated
Phase 2: XYZ repeated (with weight decay + pruning)
Phase 3: ABC replay

Expected:
- ABC weights decay during Phase 2
- XYZ emerges and stabilises
- Replay reuses surviving ABC structure (or rebuilds if fully forgotten)
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
    dim: int = 6
    seed: int = 7
    abc_steps: int = 20
    xyz_steps: int = 20
    replay_steps: int = 12
    tau: float = 1.2
    sigma_diag: float = 1.2
    use_diag: bool = True
    compression_cooccurrence_threshold: int = 999
    triplet_threshold: int = 3
    residual_surprise_threshold: float = 999.0
    weight_decay_rate_phase1: float = 0.02
    weight_decay_rate_phase2: float = 0.15
    weight_decay_rate_replay: float = 0.02
    prune_every: int = 4
    prune_min_weight: float = 0.12
    obs_noise: float = 0.02


@dataclass
class ExperimentSummary:
    abc_first_step: int | None
    xyz_first_step: int | None
    abc_weight_curve: list[float]
    xyz_weight_curve: list[float]
    abc_weight_end_phase1: float
    abc_weight_end_phase2: float
    xyz_weight_end_phase2: float
    abc_survived_phase2: bool
    abc_survived_replay: bool
    abc_replay_reuse_rate: float
    abc_rebuilt_on_replay: bool
    phase1_node_count: int
    phase2_node_count: int
    final_node_count: int
    total_pruned: int
    catastrophic_interference: bool
    verdict: str


def _make_leaf(node_id: str, mu: np.ndarray, sigma_diag: float, use_diag: bool) -> HFN:
    sigma = np.ones(mu.shape[0]) * sigma_diag if use_diag else np.eye(mu.shape[0]) * sigma_diag
    return HFN(mu=mu, sigma=sigma, id=node_id, use_diag=use_diag)


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


def _is_abc(node: HFN) -> bool:
    leaves = _leaf_descendants(node, set())
    return {"A", "B", "C"}.issubset(leaves)


def _is_xyz(node: HFN) -> bool:
    leaves = _leaf_descendants(node, set())
    return {"X", "Y", "Z"}.issubset(leaves)


def _family_nodes(forest: Forest, predicate) -> list[HFN]:
    return [n for n in forest.active_nodes() if n.children() and predicate(n)]


def _best_family_weight(forest: Forest, obs: Observer, predicate) -> tuple[float, str | None]:
    candidates = _family_nodes(forest, predicate)
    if not candidates:
        return 0.0, None
    weights = obs.state_store.weights_dict()
    best = max(candidates, key=lambda n: weights.get(n.id, 0.0))
    return float(weights.get(best.id, 0.0)), best.id


def _leaf_nodes(forest: Forest) -> list[HFN]:
    return [n for n in forest.active_nodes() if not n.children()]


def _nearest_leaf_ids(forest: Forest, x: np.ndarray, k: int = 3) -> set[str]:
    leaves = _leaf_nodes(forest)
    if not leaves:
        return set()
    ranked = sorted(leaves, key=lambda n: float(np.linalg.norm(n.mu - x)))
    return {n.id for n in ranked[:k]}


def _ensure_composite(
    forest: Forest,
    a_id: str,
    b_id: str,
    c_id: str,
    composite_id: str,
    intermediate_id: str,
) -> None:
    if composite_id in forest:
        return
    a = forest.get(a_id)
    b = forest.get(b_id)
    c = forest.get(c_id)
    if a is None or b is None or c is None:
        return
    if intermediate_id in forest:
        ab = forest.get(intermediate_id)
    else:
        ab = a.recombine(b)
        ab.id = intermediate_id  # type: ignore[misc]
        forest.register(ab)
    if ab is None:
        return
    abc = ab.recombine(c)
    abc.id = composite_id  # type: ignore[misc]
    forest.register(abc)


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentSummary:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)

    forest = Forest(D=cfg.dim, forest_id="forgetting_vs_persistence")
    mu_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    mu_c = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mu_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    mu_y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    mu_z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    for node in [
        _make_leaf("A", mu_a, cfg.sigma_diag, cfg.use_diag),
        _make_leaf("B", mu_b, cfg.sigma_diag, cfg.use_diag),
        _make_leaf("C", mu_c, cfg.sigma_diag, cfg.use_diag),
        _make_leaf("X", mu_x, cfg.sigma_diag, cfg.use_diag),
        _make_leaf("Y", mu_y, cfg.sigma_diag, cfg.use_diag),
        _make_leaf("Z", mu_z, cfg.sigma_diag, cfg.use_diag),
    ]:
        forest.register(node)

    obs = Observer(
        forest=forest,
        tau=cfg.tau,
        compression_cooccurrence_threshold=cfg.compression_cooccurrence_threshold,
        residual_surprise_threshold=cfg.residual_surprise_threshold,
        weight_decay_rate=cfg.weight_decay_rate_phase1,
        node_use_diag=cfg.use_diag,
    )

    abc_center = (mu_a + mu_b + mu_c) / 3.0
    xyz_center = (mu_x + mu_y + mu_z) / 3.0

    abc_weight_curve: list[float] = []
    xyz_weight_curve: list[float] = []
    total_pruned = 0
    abc_first_step: int | None = None
    xyz_first_step: int | None = None

    # Phase 1: ABC training
    abc_triplet_hits = 0
    for step in range(cfg.abc_steps):
        x = abc_center + rng.normal(0.0, cfg.obs_noise, size=cfg.dim)
        obs.observe(x)
        if _nearest_leaf_ids(forest, x, k=3) == {"A", "B", "C"}:
            abc_triplet_hits += 1
        if abc_triplet_hits >= cfg.triplet_threshold:
            _ensure_composite(forest, "A", "B", "C", "ABC", "AB")

        abc_weight, _ = _best_family_weight(forest, obs, _is_abc)
        xyz_weight, _ = _best_family_weight(forest, obs, _is_xyz)
        abc_weight_curve.append(abc_weight)
        xyz_weight_curve.append(xyz_weight)

        if abc_first_step is None and _family_nodes(forest, _is_abc):
            abc_first_step = step

    phase1_node_count = len(forest)
    abc_family_phase1 = {n.id for n in _family_nodes(forest, _is_abc)}
    abc_weight_end_phase1 = abc_weight_curve[-1] if abc_weight_curve else 0.0

    # Phase 2: XYZ interference
    xyz_triplet_hits = 0
    obs.weight_decay_rate = cfg.weight_decay_rate_phase2
    for step in range(cfg.xyz_steps):
        x = xyz_center + rng.normal(0.0, cfg.obs_noise, size=cfg.dim)
        res = obs.observe(x)
        if _nearest_leaf_ids(forest, x, k=3) == {"X", "Y", "Z"}:
            xyz_triplet_hits += 1
        if xyz_triplet_hits >= cfg.triplet_threshold:
            _ensure_composite(forest, "X", "Y", "Z", "XYZ", "XY")

        if cfg.prune_every > 0 and (step + 1) % cfg.prune_every == 0:
            total_pruned += obs.prune(cfg.prune_min_weight)

        abc_weight, _ = _best_family_weight(forest, obs, _is_abc)
        xyz_weight, _ = _best_family_weight(forest, obs, _is_xyz)
        abc_weight_curve.append(abc_weight)
        xyz_weight_curve.append(xyz_weight)

        if xyz_first_step is None and _family_nodes(forest, _is_xyz):
            xyz_first_step = cfg.abc_steps + step

    phase2_node_count = len(forest)
    abc_family_phase2 = {n.id for n in _family_nodes(forest, _is_abc)}
    abc_survived_phase2 = bool(abc_family_phase1 & abc_family_phase2)
    abc_weight_end_phase2 = abc_weight_curve[-1] if abc_weight_curve else 0.0
    xyz_weight_end_phase2 = xyz_weight_curve[-1] if xyz_weight_curve else 0.0

    # Phase 3: ABC replay
    abc_replay_reuse = 0
    abc_rebuilt_on_replay = False
    had_abc_on_replay_start = bool(_family_nodes(forest, _is_abc))

    obs.weight_decay_rate = cfg.weight_decay_rate_replay
    for step in range(cfg.replay_steps):
        x = abc_center + rng.normal(0.0, cfg.obs_noise, size=cfg.dim)
        res = obs.observe(x)
        if any(_is_abc(n) for n in res.explanation_tree):
            abc_replay_reuse += 1

        if not had_abc_on_replay_start and not abc_rebuilt_on_replay:
            if _family_nodes(forest, _is_abc):
                abc_rebuilt_on_replay = True

        abc_weight, _ = _best_family_weight(forest, obs, _is_abc)
        xyz_weight, _ = _best_family_weight(forest, obs, _is_xyz)
        abc_weight_curve.append(abc_weight)
        xyz_weight_curve.append(xyz_weight)

    abc_replay_reuse_rate = abc_replay_reuse / max(1, cfg.replay_steps)
    abc_family_replay = {n.id for n in _family_nodes(forest, _is_abc)}
    abc_survived_replay = bool(abc_family_phase1 & abc_family_replay)
    final_node_count = len(forest)

    catastrophic_interference = (not abc_survived_phase2) and abc_replay_reuse_rate < 0.2
    verdict = "WORKING_CONTINUAL_LEARNING"
    if abc_weight_end_phase2 >= abc_weight_end_phase1:
        verdict = "BROKEN_NO_FORGETTING"
    elif xyz_weight_end_phase2 <= 0.0:
        verdict = "BROKEN_NO_NEW_STABILISATION"
    elif catastrophic_interference:
        verdict = "BROKEN_CATASTROPHIC_INTERFERENCE"

    return ExperimentSummary(
        abc_first_step=abc_first_step,
        xyz_first_step=xyz_first_step,
        abc_weight_curve=abc_weight_curve,
        xyz_weight_curve=xyz_weight_curve,
        abc_weight_end_phase1=abc_weight_end_phase1,
        abc_weight_end_phase2=abc_weight_end_phase2,
        xyz_weight_end_phase2=xyz_weight_end_phase2,
        abc_survived_phase2=abc_survived_phase2,
        abc_survived_replay=abc_survived_replay,
        abc_replay_reuse_rate=abc_replay_reuse_rate,
        abc_rebuilt_on_replay=abc_rebuilt_on_replay,
        phase1_node_count=phase1_node_count,
        phase2_node_count=phase2_node_count,
        final_node_count=final_node_count,
        total_pruned=total_pruned,
        catastrophic_interference=catastrophic_interference,
        verdict=verdict,
    )


def _print_summary(summary: ExperimentSummary) -> None:
    print("Experiment 6: Forgetting vs Persistence")
    print(f"  ABC first step:         {summary.abc_first_step}")
    print(f"  XYZ first step:         {summary.xyz_first_step}")
    print(f"  ABC weight end phase1:  {summary.abc_weight_end_phase1:.3f}")
    print(f"  ABC weight end phase2:  {summary.abc_weight_end_phase2:.3f}")
    print(f"  XYZ weight end phase2:  {summary.xyz_weight_end_phase2:.3f}")
    print(f"  ABC survived phase2:    {summary.abc_survived_phase2}")
    print(f"  ABC survived replay:    {summary.abc_survived_replay}")
    print(f"  ABC replay reuse rate:  {summary.abc_replay_reuse_rate:.3f}")
    print(f"  ABC rebuilt on replay:  {summary.abc_rebuilt_on_replay}")
    print(f"  total pruned:           {summary.total_pruned}")
    print(f"  final nodes:            {summary.final_node_count}")
    print(f"  verdict:                {summary.verdict}")


def main() -> int:
    summary = run_experiment()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
