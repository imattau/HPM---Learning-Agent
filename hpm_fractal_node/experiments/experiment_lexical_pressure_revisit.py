"""WordNet lexical pressure-and-revisit experiment.

This experiment reuses the WordNet-backed lexical-semantic prior forest and
pushes it through a curriculum of progressively harder lexical streams, then
applies explicit memory pressure before replaying the original seed stream:
  - in-domain Peter Rabbit + repository markdown
  - medium out-of-domain vocabulary from ``nltk.corpus.words``
  - hard out-of-domain vocabulary with a stricter lexical filter
  - a pressure shock stage with a much smaller hot cache and aggressive sweeps
  - the original seed batch again, after the forest has been stressed and
    partially forgotten

The goal is to test whether HFN/HPM can survive real forgetting pressure:
does the revisit stage reuse earlier learned structure, or does it have to
rebuild the seed concepts after the shock stage?
"""

from __future__ import annotations

import gc
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.experiments import experiment_lexical_curriculum as curriculum
from hpm_fractal_node.experiments import experiment_lexical_semantic_forest as base

D = curriculum.D
SEED = curriculum.SEED
N_SAMPLES = 400
N_PASSES = 1
TAU_SIGMA = curriculum.TAU_SIGMA
TAU_MARGIN = curriculum.TAU_MARGIN

OOD_MILD_TOKEN_MIN_LEN = curriculum.OOD_MILD_TOKEN_MIN_LEN
OOD_MILD_MAX_TOKEN_POOL = curriculum.OOD_MILD_MAX_TOKEN_POOL
OOD_MILD_TOKENS_PER_OBS = curriculum.OOD_MILD_TOKENS_PER_OBS
OOD_HARD_TOKEN_MIN_LEN = curriculum.OOD_HARD_TOKEN_MIN_LEN
OOD_HARD_MAX_TOKEN_POOL = curriculum.OOD_HARD_MAX_TOKEN_POOL
OOD_HARD_TOKENS_PER_OBS = curriculum.OOD_HARD_TOKENS_PER_OBS

PRESSURE_MAX_HOT = 96
PRESSURE_SWEEP_EVERY = 25
PRESSURE_MIN_FREE_RAM_MB = 10**9
PRESSURE_TOKEN_MIN_LEN = 9
PRESSURE_MAX_TOKEN_POOL = 20000
PRESSURE_TOKENS_PER_OBS = (18, 28)

ObservationBatch = curriculum.ObservationBatch


@dataclass(frozen=True)
class StageStats:
    stage_name: str
    difficulty: str
    source_name: str
    wall_clock_s: float
    rss_delta_mb: float
    prior_nodes: int
    final_node_count: int
    learned_nodes_surviving: int
    new_learned_nodes_surviving: int
    learned_nodes_explained: int
    coverage_pct: float
    mean_explaining_layer: float
    layer_counts: dict[str, int]
    known_token_rate: float
    fallback_rate: float
    avg_tokens: float
    unique_tokens: int
    mean_purity: float
    n_purity_nodes: int
    hot_start: int
    max_hot: int
    sweep_every: int
    min_free_ram_mb: int


@dataclass(frozen=True)
class StageResult:
    stats: StageStats
    created_learned_ids: frozenset[str]
    explained_learned_ids: frozenset[str]


@dataclass(frozen=True)
class PressureConfig:
    max_hot: int
    sweep_every: int
    min_free_ram_mb: int


def _build_world_model(mode: str = "large"):
    return curriculum._build_world_model(mode)


def _make_observer(forest: TieredForest, prior_ids: set[str]) -> Observer:
    return curriculum._make_observer(forest, prior_ids)


def _pressure_pool(inventory) -> tuple[str, ...]:
    return curriculum._ood_token_pool(
        inventory,
        min_len=PRESSURE_TOKEN_MIN_LEN,
        max_pool=PRESSURE_MAX_TOKEN_POOL,
        require_hard_chars=True,
    )


def _apply_pressure(forest: TieredForest) -> PressureConfig:
    baseline = PressureConfig(
        max_hot=getattr(forest, "_max_hot", 500),
        sweep_every=getattr(forest, "_sweep_every", 100),
        min_free_ram_mb=getattr(forest, "_min_free_ram_mb", 500),
    )
    forest._max_hot = min(baseline.max_hot, PRESSURE_MAX_HOT)
    forest._sweep_every = PRESSURE_SWEEP_EVERY
    forest._min_free_ram_mb = PRESSURE_MIN_FREE_RAM_MB
    try:
        forest._evict_lru_if_needed()
    except Exception:
        pass
    return baseline


def _restore_pressure(forest: TieredForest, baseline: PressureConfig) -> None:
    forest._max_hot = baseline.max_hot
    forest._sweep_every = baseline.sweep_every
    forest._min_free_ram_mb = baseline.min_free_ram_mb
    try:
        forest._evict_lru_if_needed()
    except Exception:
        pass


def _run_stage(
    stage_name: str,
    difficulty: str,
    batch: ObservationBatch,
    forest: TieredForest,
    prior_ids: set[str],
    observer: Observer,
) -> StageResult:
    print("\n" + "=" * 72)
    print(f"  Stage: {stage_name} ({difficulty})")
    print("=" * 72)

    start_ids = curriculum._forest_node_ids(forest)
    counts = {
        "lemmas": sum(1 for pid in prior_ids if pid.startswith("lemma_")),
        "synsets": sum(1 for pid in prior_ids if pid.startswith("syn_")),
        "relations": sum(1 for pid in prior_ids if pid.startswith("rel_")),
        "pos": sum(1 for pid in prior_ids if pid.startswith("pos_")),
        "lex": sum(1 for pid in prior_ids if pid.startswith("lex_")),
        "depth": sum(1 for pid in prior_ids if pid.startswith("depth_")),
        "ontology": sum(1 for pid in prior_ids if pid == "ontology_root"),
    }
    print(
        f"  Priors: total={len(prior_ids)}  lemmas={counts['lemmas']}  synsets={counts['synsets']}  "
        f"relations={counts['relations']}  pos={counts['pos']}  lex={counts['lex']}  "
        f"depth={counts['depth']}  ontology={counts['ontology']}"
    )
    print(f"  Hot priors on start: {forest.hot_count()}")
    print(f"  Pressure settings: max_hot={getattr(forest, '_max_hot', '?')} sweep_every={getattr(forest, '_sweep_every', '?')} min_free_ram_mb={getattr(forest, '_min_free_ram_mb', '?')}")
    print(f"  Known token rate: {batch.known_token_hits / max(batch.token_total, 1):.2%}")
    print(f"  Fallback rate: {batch.fallback_obs / max(len(batch.observations), 1):.2%}")
    print(f"  Avg tokens / obs: {batch.token_total / max(len(batch.observations), 1):.2f}")
    print(f"  Unique tokens: {batch.unique_tokens}")
    print(f"  tau = {observer.tau:.2f}")

    node_explanations: dict[str, list[str]] = defaultdict(list)
    process = psutil.Process()
    gc.collect()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    t0 = time.perf_counter()

    for p in range(N_PASSES):
        n_explained = 0
        n_unexplained = 0
        order = np.random.default_rng(SEED + p).permutation(len(batch.observations))
        for idx in order:
            vec, _, category = batch.observations[idx]
            result = observer.observe(vec.astype(np.float64))
            forest._on_observe()
            if result.explanation_tree:
                best_id = max(result.accuracy_scores, key=lambda key: result.accuracy_scores[key])
                node_explanations[best_id].append(category)
                n_explained += 1
            else:
                n_unexplained += 1
            peak_rss = max(peak_rss, process.memory_info().rss)
        print(
            f"  Pass {p + 1}: explained {n_explained}/{len(batch.observations)} "
            f"({100.0 * n_explained / len(batch.observations):.1f}%), unexplained {n_unexplained}"
        )

    elapsed = time.perf_counter() - t0
    gc.collect()
    peak_rss = max(peak_rss, process.memory_info().rss)

    end_ids = curriculum._forest_node_ids(forest)
    created_learned_ids = frozenset(node_id for node_id in (end_ids - start_ids) if node_id not in prior_ids)
    explained_learned_ids = frozenset(node_id for node_id in node_explanations if node_id not in prior_ids)
    prior_explained = sum(len(labels) for node_id, labels in node_explanations.items() if node_id in prior_ids)
    learned_explained = sum(len(node_explanations[node_id]) for node_id in explained_learned_ids)
    total_attributed = prior_explained + learned_explained
    n_obs_total = N_PASSES * len(batch.observations)

    layer_counts: dict[str, int] = defaultdict(int)
    layer_rank_sum = 0.0
    layer_rank_n = 0
    for node_id, labels in node_explanations.items():
        layer = base._layer_of(node_id)
        layer_counts[layer] += len(labels)
        layer_rank_sum += base._layer_rank(layer) * len(labels)
        layer_rank_n += len(labels)

    cat_purities: list[float] = []
    for node_id in explained_learned_ids:
        if forest.get(node_id) is None:
            continue
        labels = node_explanations[node_id]
        if len(labels) < 5:
            continue
        cat_counts: dict[str, int] = defaultdict(int)
        for category in labels:
            cat_counts[category] += 1
        cat_purities.append(base.purity(cat_counts))

    return StageResult(
        stats=StageStats(
            stage_name=stage_name,
            difficulty=difficulty,
            source_name=batch.source_name,
            wall_clock_s=elapsed,
            rss_delta_mb=(peak_rss - start_rss) / (1024**2),
            prior_nodes=len(prior_ids),
            final_node_count=len(forest),
            learned_nodes_surviving=sum(1 for node_id in end_ids if node_id not in prior_ids),
            new_learned_nodes_surviving=len(created_learned_ids),
            learned_nodes_explained=len(explained_learned_ids),
            coverage_pct=100.0 * total_attributed / max(n_obs_total, 1),
            mean_explaining_layer=(layer_rank_sum / layer_rank_n) if layer_rank_n else 0.0,
            layer_counts=dict(layer_counts),
            known_token_rate=batch.known_token_hits / max(batch.token_total, 1),
            fallback_rate=batch.fallback_obs / max(len(batch.observations), 1),
            avg_tokens=batch.token_total / max(len(batch.observations), 1),
            unique_tokens=batch.unique_tokens,
            mean_purity=float(np.mean(cat_purities)) if cat_purities else float("nan"),
            n_purity_nodes=len(cat_purities),
            hot_start=forest.hot_count(),
            max_hot=getattr(forest, "_max_hot", -1),
            sweep_every=getattr(forest, "_sweep_every", -1),
            min_free_ram_mb=getattr(forest, "_min_free_ram_mb", -1),
        ),
        created_learned_ids=created_learned_ids,
        explained_learned_ids=explained_learned_ids,
    )


def _print_summary(results: list[StageResult], forest: TieredForest, pressure_seed_remaining: int) -> None:
    stages = [result.stats for result in results]
    result_by_name = {result.stats.stage_name: result for result in results}

    print("\n" + "=" * 72)
    print("  PRESSURE SUMMARY")
    print("=" * 72)
    print(f"  {'Stage':<24}{'Difficulty':<18}{'Coverage':>12}{'New learned':>14}{'Surviving':>12}{'Mean layer':>12}")
    print("  " + "-" * 94)
    for stage in stages:
        print(
            f"  {stage.stage_name:<24}{stage.difficulty:<18}"
            f"{stage.coverage_pct:>11.2f}%"
            f"{stage.new_learned_nodes_surviving:>14}"
            f"{stage.learned_nodes_surviving:>12}"
            f"{stage.mean_explaining_layer:>12.2f}"
        )

    print("\n  Stage details")
    for stage in stages:
        print(
            f"  {stage.stage_name:<24}"
            f"known={stage.known_token_rate:.2%} "
            f"fallback={stage.fallback_rate:.2%} "
            f"avg_tokens={stage.avg_tokens:.2f} "
            f"unique={stage.unique_tokens} "
            f"learned_explained={stage.learned_nodes_explained} "
            f"purity_nodes={stage.n_purity_nodes} "
            f"mean_purity={stage.mean_purity:.3f} "
            f"rss_delta={stage.rss_delta_mb:.1f}MB "
            f"hot_start={stage.hot_start} "
            f"max_hot={stage.max_hot} "
            f"sweep_every={stage.sweep_every} "
            f"min_free_ram_mb={stage.min_free_ram_mb} "
            f"source={stage.source_name}"
        )
        for key in ("lemma", "synset", "relation", "pos", "lexname", "depth", "ontology", "learned"):
            print(f"    layer[{key:<10}] {stage.layer_counts.get(key, 0):>10}")

    if stages:
        first = stages[0]
        last = stages[-1]
        print(f"\n  Coverage delta (last-first): {last.coverage_pct - first.coverage_pct:+.4f}%")
        print(f"  Learned nodes delta (last-first): {last.learned_nodes_surviving - first.learned_nodes_surviving:+d}")
        print(f"  Mean explaining-layer delta (last-first): {last.mean_explaining_layer - first.mean_explaining_layer:+.2f}")

    seed_result = result_by_name.get("seed")
    revisit_result = result_by_name.get("revisit_seed")
    pressure_result = result_by_name.get("pressure_shock")
    if seed_result and revisit_result and pressure_result:
        seed_learned = seed_result.created_learned_ids
        revisit_explained = revisit_result.explained_learned_ids
        pressure_remaining = sum(1 for node_id in seed_learned if forest.get(node_id) is not None)
        reused_seed_learned = len(seed_learned & revisit_explained)
        print("\n  Pressure focus")
        print(f"  Seed learned nodes created: {len(seed_learned)}")
        print(f"  Seed learned nodes remaining after pressure: {pressure_remaining}")
        print(f"  Seed learned nodes reused during revisit: {reused_seed_learned}")
        print(f"  Pressure stage learned explanations: {pressure_result.stats.learned_nodes_explained}")
        print(f"  Revisit coverage delta vs seed: {revisit_result.stats.coverage_pct - seed_result.stats.coverage_pct:+.4f}%")
        print(
            f"  Revisit mean layer delta vs seed: "
            f"{revisit_result.stats.mean_explaining_layer - seed_result.stats.mean_explaining_layer:+.2f}"
        )
        print(
            f"  Revisit learned explanations delta vs seed: "
            f"{revisit_result.stats.learned_nodes_explained - seed_result.stats.learned_nodes_explained:+d}"
        )


def main() -> None:
    print("WordNet lexical pressure-and-revisit experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print("  Observation stream stretches a WordNet forest, forces a memory shock, then replays the seed stream.")

    mode = "compact" if N_SAMPLES <= 5 else "large"
    forest, prior_ids, inventory = _build_world_model(mode)
    segments = curriculum._in_domain_segments()
    mild_pool = curriculum._ood_token_pool(
        inventory,
        min_len=OOD_MILD_TOKEN_MIN_LEN,
        max_pool=OOD_MILD_MAX_TOKEN_POOL,
        require_hard_chars=False,
    )
    hard_pool = curriculum._ood_token_pool(
        inventory,
        min_len=OOD_HARD_TOKEN_MIN_LEN,
        max_pool=OOD_HARD_MAX_TOKEN_POOL,
        require_hard_chars=True,
    )
    pressure_pool = _pressure_pool(inventory)
    print(f"  In-domain segments available: {len(segments)}")
    print(f"  Mild OOD token pool size: {len(mild_pool)}")
    print(f"  Hard OOD token pool size: {len(hard_pool)}")
    print(f"  Pressure token pool size: {len(pressure_pool)}")
    print(f"  World-model mode: {mode}")

    observer = _make_observer(forest, prior_ids)
    baseline_pressure = PressureConfig(
        max_hot=getattr(forest, "_max_hot", 500),
        sweep_every=getattr(forest, "_sweep_every", 100),
        min_free_ram_mb=getattr(forest, "_min_free_ram_mb", 500),
    )

    seed_batch = curriculum._make_in_domain_batch(inventory, N_SAMPLES, seed=SEED)
    stretch_batch = curriculum._make_ood_batch(
        inventory,
        N_SAMPLES,
        seed=SEED + 1,
        token_pool=mild_pool,
        tokens_per_obs=OOD_MILD_TOKENS_PER_OBS,
        source_name="mild_ood",
    )
    hard_batch = curriculum._make_ood_batch(
        inventory,
        N_SAMPLES,
        seed=SEED + 2,
        token_pool=hard_pool,
        tokens_per_obs=OOD_HARD_TOKENS_PER_OBS,
        source_name="hard_ood",
    )
    pressure_batch = curriculum._make_ood_batch(
        inventory,
        N_SAMPLES,
        seed=SEED + 3,
        token_pool=pressure_pool,
        tokens_per_obs=PRESSURE_TOKENS_PER_OBS,
        source_name="pressure_ood",
    )
    revisit_batch = seed_batch

    print(f"  Generated {len(seed_batch.observations)} seed observations")
    print(f"  Generated {len(stretch_batch.observations)} stretch observations")
    print(f"  Generated {len(hard_batch.observations)} hard observations")
    print(f"  Generated {len(pressure_batch.observations)} pressure observations")
    print(f"  Generated {len(revisit_batch.observations)} revisit observations")

    stage_plan = [
        ("seed", "easy", seed_batch, None),
        ("stretch", "medium", stretch_batch, None),
        ("stretch_hard", "hard", hard_batch, None),
        ("pressure_shock", "shock", pressure_batch, "apply"),
        ("revisit_seed", "replay", revisit_batch, "restore"),
    ]

    results: list[StageResult] = []
    pressure_applied = False
    pressure_seed_remaining_ids: set[str] = set()
    for stage_name, difficulty, batch, pressure_action in stage_plan:
        if pressure_action == "apply" and not pressure_applied:
            _apply_pressure(forest)
            pressure_applied = True
        elif pressure_action == "restore" and pressure_applied:
            pressure_seed_remaining_ids = {
                node_id for node_id in results[0].created_learned_ids if forest.get(node_id) is not None
            }
            _restore_pressure(forest, baseline_pressure)
            pressure_applied = False
        results.append(_run_stage(stage_name, difficulty, batch, forest, prior_ids, observer))

    if pressure_applied:
        pressure_seed_remaining_ids = {
            node_id for node_id in results[0].created_learned_ids if forest.get(node_id) is not None
        }
        _restore_pressure(forest, baseline_pressure)

    _print_summary(results, forest, len(pressure_seed_remaining_ids))


if __name__ == "__main__":
    main()
