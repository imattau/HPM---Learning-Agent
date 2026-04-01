"""Math throughput benchmark: observations per second under different node storage modes.

Sweeps the arithmetic observation stream across multiple sample sizes and compares
node_use_diag=False (full sigma storage for learned nodes) against node_use_diag=True
(diagonal sigma storage for learned nodes).

For each run it measures:
- total wall-clock time
- observations per second overall
- first-pass and second-pass throughput
- peak RSS delta
- coverage
- learned-node survival and purity

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_math_throughput.py
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

from hfn import Observer, calibrate_tau
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.math.math_loader import CATEGORY_NAMES, D, generate_observations, get_category
from hpm_fractal_node.math.math_world_model import build_math_world_model

SAMPLE_SIZES = (50, 100, 250)
N_PASSES = 2
SEED = 42

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
    use_diag: bool
    n_samples: int
    elapsed_s: float
    pass_elapsed_s: tuple[float, ...]
    peak_rss_delta_mb: float
    n_priors: int
    final_node_count: int
    n_active_learned: int
    learned_nodes_surviving: int
    learned_nodes_explained: int
    coverage_pct: float
    mean_purity: float
    n_purity_nodes: int
    total_obs_per_s: float
    first_pass_obs_per_s: float
    second_pass_obs_per_s: float
    learned_nodes_per_s: float
    learned_explained_per_s: float
    throughput_drop_pct: float


def _run_once(use_diag: bool, n_samples: int) -> RunResult:
    label = 'diag' if use_diag else 'full'
    print()
    print('=' * 72)
    print(f'  Mode: node_use_diag={use_diag}  ({label} sigma storage) | samples={n_samples}')
    print('=' * 72)

    data = generate_observations(n=n_samples, seed=SEED)

    cold_dir = Path(__file__).parents[2] / 'data' / 'hfn_math_cold'
    cold_dir.mkdir(parents=True, exist_ok=True)

    forest, prior_ids = build_math_world_model(
        forest_cls=TieredForest,
        cold_dir=cold_dir,
        max_hot=600,
    )
    forest.set_protected(prior_ids)
    n_priors = len(prior_ids)
    print(f'  {n_priors} priors registered and protected')

    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)

    obs = Observer(
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
        node_use_diag=use_diag,
    )

    node_explanations: dict[str, list[str]] = defaultdict(list)

    process = psutil.Process()
    gc.collect()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    t0 = time.perf_counter()
    pass_elapsed_s: list[float] = []

    for p in range(N_PASSES):
        pass_t0 = time.perf_counter()
        n_explained = 0
        n_unexplained = 0
        rng = np.random.default_rng(SEED + p)
        order = rng.permutation(len(data))

        for i in order:
            vec, (left, op, right, result) = data[i]
            obs_result = obs.observe(vec.astype(np.float64))
            forest._on_observe()

            category = get_category(left, op, right, result)
            if obs_result.explanation_tree:
                best_id = max(obs_result.accuracy_scores, key=lambda k: obs_result.accuracy_scores[k])
                node_explanations[best_id].append(category)
                n_explained += 1
            else:
                n_unexplained += 1

            rss = process.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss

        elapsed = time.perf_counter() - pass_t0
        pass_elapsed_s.append(elapsed)
        n_total = len(data)
        print(
            f'  Pass {p + 1}: explained {n_explained}/{n_total} '
            f'({100.0 * n_explained / n_total:.1f}%), elapsed={elapsed:.2f}s, '
            f'throughput={n_total / max(elapsed, 1e-9):.1f} obs/s'
        )

    elapsed = time.perf_counter() - t0
    gc.collect()
    peak_rss = max(peak_rss, process.memory_info().rss)

    n_obs_total = N_PASSES * n_samples
    total_attributed = sum(len(v) for v in node_explanations.values())
    coverage_pct = 100.0 * total_attributed / max(n_obs_total, 1)

    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    active_nodes = list(forest.active_nodes())
    n_active_learned = sum(1 for n in active_nodes if n.id not in prior_ids)
    total_node_count = len(forest)
    learned_nodes_surviving = sum(1 for n in active_nodes if n.id not in prior_ids)

    cat_purities: list[float] = []
    for nid in learned_nodes:
        labels = node_explanations[nid]
        if len(labels) < 5:
            continue
        cat_counts: dict[str, int] = defaultdict(int)
        for cat in labels:
            cat_counts[cat] += 1
        cat_purities.append(purity(cat_counts))

    mean_purity = float(np.mean(cat_purities)) if cat_purities else float('nan')
    first_pass_obs_per_s = n_samples / max(pass_elapsed_s[0], 1e-9)
    second_pass_obs_per_s = n_samples / max(pass_elapsed_s[1], 1e-9) if len(pass_elapsed_s) > 1 else first_pass_obs_per_s
    throughput_drop_pct = 0.0
    if len(pass_elapsed_s) > 1 and first_pass_obs_per_s > 0:
        throughput_drop_pct = 100.0 * (first_pass_obs_per_s - second_pass_obs_per_s) / first_pass_obs_per_s

    return RunResult(
        label=label,
        use_diag=use_diag,
        n_samples=n_samples,
        elapsed_s=elapsed,
        pass_elapsed_s=tuple(pass_elapsed_s),
        peak_rss_delta_mb=(peak_rss - start_rss) / (1024 ** 2),
        n_priors=n_priors,
        final_node_count=total_node_count,
        n_active_learned=n_active_learned,
        learned_nodes_surviving=learned_nodes_surviving,
        learned_nodes_explained=len(learned_nodes),
        coverage_pct=coverage_pct,
        mean_purity=mean_purity,
        n_purity_nodes=len(cat_purities),
        total_obs_per_s=n_obs_total / max(elapsed, 1e-9),
        first_pass_obs_per_s=first_pass_obs_per_s,
        second_pass_obs_per_s=second_pass_obs_per_s,
        learned_nodes_per_s=learned_nodes_surviving / max(elapsed, 1e-9),
        learned_explained_per_s=len(learned_nodes) / max(elapsed, 1e-9),
        throughput_drop_pct=throughput_drop_pct,
    )


def _print_comparison(rows: list[RunResult]) -> None:
    print()
    print('=' * 72)
    print('  THROUGHPUT SUMMARY')
    print('=' * 72)
    print(
        f"  {'Samples':<10}{'Mode':<10}{'Obs/s':>12}{'Pass1':>12}{'Pass2':>12}"
        f"{'Coverage':>12}{'RSS MB':>10}{'Learned':>10}"
    )
    print('  ' + '-' * 84)
    for row in rows:
        print(
            f"  {row.n_samples:<10d}{row.label:<10}"
            f"{row.total_obs_per_s:>11.1f}"
            f"{row.first_pass_obs_per_s:>12.1f}"
            f"{row.second_pass_obs_per_s:>12.1f}"
            f"{row.coverage_pct:>11.2f}%"
            f"{row.peak_rss_delta_mb:>10.1f}"
            f"{row.learned_nodes_surviving:>10d}"
        )

    print()
    print('  Detailed metrics')
    for row in rows:
        print(
            f"  samples={row.n_samples:<5d} mode={row.label:<4} "
            f"elapsed={row.elapsed_s:.2f}s pass_drop={row.throughput_drop_pct:+.1f}% "
            f"learned_explained={row.learned_nodes_explained} "
            f"learned_per_s={row.learned_nodes_per_s:.2f} "
            f"learned_explained_per_s={row.learned_explained_per_s:.2f} "
            f"mean_purity={row.mean_purity:.3f} purity_nodes={row.n_purity_nodes} "
            f"final_nodes={row.final_node_count} active_learned={row.n_active_learned}"
        )

    baseline = {row.n_samples: row for row in rows if not row.use_diag}
    diag = {row.n_samples: row for row in rows if row.use_diag}
    print()
    print('  Diag vs full speedup')
    for n_samples in SAMPLE_SIZES:
        if n_samples not in baseline or n_samples not in diag:
            continue
        full = baseline[n_samples]
        d = diag[n_samples]
        speedup = full.total_obs_per_s / max(d.total_obs_per_s, 1e-9)
        print(
            f"  samples={n_samples:<5d} full={full.total_obs_per_s:>8.1f} obs/s "
            f"diag={d.total_obs_per_s:>8.1f} obs/s speedup={speedup:>5.2f}x"
        )

    print()
    print(f'  Random category baseline: {RANDOM_BASELINE:.3f} (1/{N_CATEGORIES})')


def main() -> None:
    print('Math throughput benchmark')
    print(f'  D={D}, N_PASSES={N_PASSES}, SEED={SEED}')
    print('  Measures observations per second across sample sizes and sigma storage modes.')

    results: list[RunResult] = []
    for n_samples in SAMPLE_SIZES:
        for use_diag in (False, True):
            results.append(_run_once(use_diag=use_diag, n_samples=n_samples))

    _print_comparison(results)


if __name__ == '__main__':
    main()
