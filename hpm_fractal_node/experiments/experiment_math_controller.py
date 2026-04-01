"""Math controller adapter experiment.

Compares the direct synchronous HFN loop against the async controller layer
introduced in hfn.hfn_controller.

The goal is not to beat the direct path. The goal is to show that a thin async
adapter can sit above HFN and still preserve the same learning semantics while
handling ingest, replay, prefetch, and state export as separate responsibilities.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_math_controller.py
"""
from __future__ import annotations

import asyncio
import gc
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.tiered_forest import TieredForest
from hfn.hfn_controller import AsyncHFNController
from hpm_fractal_node.math.math_loader import CATEGORY_NAMES, D, generate_observations, get_category
from hpm_fractal_node.math.math_world_model import build_math_world_model

N_SAMPLES = 250
N_PASSES = 1
SEED = 42
REPLAY_COUNT = 25
PREFETCH_COUNT = 12

TAU_SIGMA = 1.0
TAU_MARGIN = 5.0

N_CATEGORIES = len(CATEGORY_NAMES)
RANDOM_BASELINE = 1.0 / N_CATEGORIES


def purity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


@dataclass(frozen=True)
class RunResult:
    label: str
    elapsed_s: float
    peak_rss_delta_mb: float
    n_priors: int
    final_node_count: int
    learned_nodes_surviving: int
    learned_nodes_explained: int
    coverage_pct: float
    mean_purity: float
    n_purity_nodes: int
    obs_per_s: float
    replay_obs_per_s: float
    snapshot_queue_size: int
    snapshot_replayed: int
    snapshot_prefetched: int
    snapshot_gap_queries: int
    snapshot_last_event: str | None
    prefetched_found: int
    replay_count: int


def _build_math_stack():
    cold_dir = Path(__file__).parents[2] / 'data' / 'hfn_math_cold'
    cold_dir.mkdir(parents=True, exist_ok=True)
    forest, prior_ids = build_math_world_model(
        forest_cls=TieredForest,
        cold_dir=cold_dir,
        max_hot=600,
    )
    forest.set_protected(prior_ids)
    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
    observer = Observer(
        forest,
        tau=tau,
        protected_ids=prior_ids,
        recombination_strategy='nearest_prior',
        hausdorff_absorption_threshold=0.35,
        hausdorff_absorption_weight_floor=0.4,
        absorption_miss_threshold=20,
        persistence_guided_absorption=True,
        lacunarity_guided_creation=True,
        lacunarity_creation_radius=0.08,
        multifractal_guided_absorption=False,
        gap_query_threshold=None,
        max_expand_depth=2,
        node_use_diag=True,
    )
    return forest, prior_ids, observer


def _summarise(node_explanations: dict[str, list[str]], prior_ids: set[str], forest: TieredForest, elapsed: float, start_rss: int, peak_rss: int, n_samples: int, replay_obs_per_s: float = 0.0) -> RunResult:
    n_obs_total = n_samples * N_PASSES
    total_attributed = sum(len(v) for v in node_explanations.values())
    coverage_pct = 100.0 * total_attributed / max(n_obs_total, 1)

    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    active_nodes = list(forest.active_nodes())
    learned_nodes_surviving = sum(1 for n in active_nodes if n.id not in prior_ids)
    total_node_count = len(forest)

    cat_purities: list[float] = []
    for nid in learned_nodes:
        labels = node_explanations[nid]
        if len(labels) < 5:
            continue
        cat_counts: dict[str, int] = defaultdict(int)
        for cat in labels:
            cat_counts[cat] += 1
        cat_purities.append(purity(cat_counts))

    return RunResult(
        label='controller',
        elapsed_s=elapsed,
        peak_rss_delta_mb=(peak_rss - start_rss) / (1024 ** 2),
        n_priors=len(prior_ids),
        final_node_count=total_node_count,
        learned_nodes_surviving=learned_nodes_surviving,
        learned_nodes_explained=len(learned_nodes),
        coverage_pct=coverage_pct,
        mean_purity=float(np.mean(cat_purities)) if cat_purities else float('nan'),
        n_purity_nodes=len(cat_purities),
        obs_per_s=n_obs_total / max(elapsed, 1e-9),
        replay_obs_per_s=replay_obs_per_s,
        snapshot_queue_size=0,
        snapshot_replayed=0,
        snapshot_prefetched=0,
        snapshot_gap_queries=0,
        snapshot_last_event=None,
        prefetched_found=0,
        replay_count=0,
    )


def _run_direct(data):
    forest, prior_ids, observer = _build_math_stack()
    node_explanations: dict[str, list[str]] = defaultdict(list)
    process = psutil.Process()
    gc.collect()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    t0 = time.perf_counter()
    for p in range(N_PASSES):
        order = np.random.default_rng(SEED + p).permutation(len(data))
        for i in order:
            vec, (left, op, right, result) = data[i]
            obs_result = observer.observe(vec.astype(np.float64))
            forest._on_observe()
            if obs_result.explanation_tree:
                best_id = max(obs_result.accuracy_scores, key=lambda k: obs_result.accuracy_scores[k])
                node_explanations[best_id].append(get_category(left, op, right, result))
            peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed = time.perf_counter() - t0
    return _summarise(node_explanations, prior_ids, forest, elapsed, start_rss, peak_rss, len(data))


async def _run_controller(data):
    forest, prior_ids, observer = _build_math_stack()

    async def gap_query_fn(gap_mu: np.ndarray, context=None):
        _ = float(np.linalg.norm(gap_mu))
        return ['gap', 'query']

    controller = AsyncHFNController(forest, observer, gap_query_fn=gap_query_fn)
    node_explanations: dict[str, list[str]] = defaultdict(list)
    process = psutil.Process()
    gc.collect()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    t0 = time.perf_counter()

    async with controller:
        prefetch_result = await controller.prefetch(list(prior_ids)[:PREFETCH_COUNT])
        for p in range(N_PASSES):
            order = np.random.default_rng(SEED + p).permutation(len(data))
            for i in order:
                vec, (left, op, right, result) = data[i]
                obs_result = await controller.ingest(vec, label=get_category(left, op, right, result))
                if obs_result['explained_ids']:
                    node_explanations[obs_result['explained_ids'][0]].append(get_category(left, op, right, result))
                peak_rss = max(peak_rss, process.memory_info().rss)

        replay_obs = [data[i][0] for i in range(min(REPLAY_COUNT, len(data)))]
        replay_t0 = time.perf_counter()
        replay_results = await controller.replay(replay_obs, label='replay')
        replay_elapsed = time.perf_counter() - replay_t0
        snapshot = await controller.snapshot_state()
        gap_result = await controller.request_gap_query(np.mean(np.stack([vec for vec, _ in data[:5]]), axis=0), context={'source': 'math_controller'})
        peak_rss = max(peak_rss, process.memory_info().rss)

    elapsed = time.perf_counter() - t0
    result = _summarise(node_explanations, prior_ids, forest, elapsed, start_rss, peak_rss, len(data), replay_obs_per_s=len(replay_obs) / max(replay_elapsed, 1e-9))
    return RunResult(
        **{**result.__dict__,
           'snapshot_queue_size': snapshot.queue_size,
           'snapshot_replayed': snapshot.replayed_observations,
           'snapshot_prefetched': snapshot.prefetched_nodes,
           'snapshot_gap_queries': snapshot.gap_queries,
           'snapshot_last_event': snapshot.last_event,
           'prefetched_found': len(prefetch_result.found_ids),
           'replay_count': len(replay_results),
        }
    )


def _print_report(direct: RunResult, controller: RunResult) -> None:
    print()
    print('=' * 72)
    print('  CONTROLLER COMPARISON')
    print('=' * 72)
    print(f"  {'Metric':<26}{'Direct':>18}{'Controller':>18}")
    print('  ' + '-' * 62)
    print(f"  {'Obs/s':<26}{direct.obs_per_s:>18.1f}{controller.obs_per_s:>18.1f}")
    print(f"  {'Replay obs/s':<26}{'-':>18}{controller.replay_obs_per_s:>18.1f}")
    print(f"  {'Coverage %':<26}{direct.coverage_pct:>17.2f}%{controller.coverage_pct:>17.2f}%")
    print(f"  {'Peak RSS delta (MB)':<26}{direct.peak_rss_delta_mb:>18.1f}{controller.peak_rss_delta_mb:>18.1f}")
    print(f"  {'Learned surviving':<26}{direct.learned_nodes_surviving:>18d}{controller.learned_nodes_surviving:>18d}")
    print(f"  {'Snapshot queue size':<26}{'-':>18}{controller.snapshot_queue_size:>18d}")
    print(f"  {'Prefetched found':<26}{'-':>18}{controller.prefetched_found:>18d}")
    print(f"  {'Replay count':<26}{'-':>18}{controller.replay_count:>18d}")
    print(f"  {'Snapshot replayed':<26}{'-':>18}{controller.snapshot_replayed:>18d}")
    print(f"  {'Snapshot gap queries':<26}{'-':>18}{controller.snapshot_gap_queries:>18d}")
    print(f"  {'Snapshot last event':<26}{'-':>18}{str(controller.snapshot_last_event):>18}")
    print(f"  {'Random baseline':<26}{RANDOM_BASELINE:>18.3f}{RANDOM_BASELINE:>18.3f}")


def main() -> None:
    print('Math controller adapter experiment')
    print(f'  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}')
    data = generate_observations(n=N_SAMPLES, seed=SEED)
    direct = _run_direct(data)
    controller = asyncio.run(_run_controller(data))
    _print_report(direct, controller)


if __name__ == '__main__':
    main()
