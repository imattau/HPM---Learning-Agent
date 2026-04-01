"""Multi-process multi-observer lifecycle experiment.

Parallelizes HFN observations using one process per observer.
- math observer -> worker process 1
- text observer -> worker process 2
- mixed observer -> worker process 3 (on-demand)
- controller -> routes batches of work and merges results

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_multi_observer_lifecycle.py
"""
from __future__ import annotations

import gc
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.math.math_loader import D as MATH_D, generate_observations, get_category
from hpm_fractal_node.math.math_world_model import build_math_world_model
from hpm_fractal_node.nlp.nlp_loader import D as TEXT_D, category_names, generate_sentences
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model

SEED = 42
STAGE_SIZE = 120
BATCH_SIZE = 20
TAU_SIGMA = 1.0
TAU_MARGIN = 5.0

COMMON_D = MATH_D + TEXT_D
MATH_SLICE = slice(0, MATH_D)
TEXT_SLICE = slice(MATH_D, MATH_D + TEXT_D)

MATH_HOT = 220
TEXT_HOT = 220
MONO_HOT = 320
MIX_HOT = 260
OFFSLICE_VAR = 1.0


@dataclass(frozen=True)
class DomainSample:
    vec: np.ndarray
    label: str
    detail: str


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str  # pure | mixed
    domains: tuple[str, ...]


@dataclass
class StageResult:
    name: str
    observations: int
    explained: int
    new_nodes: int
    net_node_delta: int
    hot_learned: int
    total_learned: int


@dataclass
class SystemResult:
    label: str
    elapsed_s: float
    peak_rss_delta_mb: float
    coverage_pct: float
    observations: int
    explained: int
    total_nodes: int
    learned_total: int
    learned_hot: int
    mixed_cache_creations: int = 0
    mixed_cache_hits: int = 0
    specialist_reactivations: int = 0
    stage_results: list[StageResult] = field(default_factory=list)


def _sample_math(n: int, seed: int) -> list[DomainSample]:
    data = generate_observations(n=n, seed=seed)
    out: list[DomainSample] = []
    for vec, (left, op, right, result) in data:
        out.append(DomainSample(vec=vec.astype(np.float64), label=get_category(left, op, right, result), detail=f"{left}{op}{right}={result}"))
    return out


def _sample_text(n: int, seed: int) -> list[DomainSample]:
    data = generate_sentences(seed=seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))[:n]
    out: list[DomainSample] = []
    for idx in order:
        vec, true_word, category = data[idx]
        out.append(DomainSample(vec=vec.astype(np.float64), label=category, detail=true_word))
    return out


def _embed(sample: DomainSample, common_slice: slice, common_d: int = COMMON_D) -> np.ndarray:
    x = np.zeros(common_d, dtype=np.float64)
    x[common_slice] = sample.vec
    return x


def _clone_node(node: HFN, *, prefix: str, common_d: int, slice_: slice | None) -> HFN:
    mu = np.zeros(common_d, dtype=np.float64)
    sigma = np.full(common_d, OFFSLICE_VAR, dtype=np.float64)
    if node.use_diag:
        diag = np.asarray(node.sigma, dtype=np.float64)
    else:
        sigma_arr = np.asarray(node.sigma, dtype=np.float64)
        diag = np.diag(sigma_arr) if sigma_arr.ndim == 2 else sigma_arr
    if slice_ is None:
        if node.mu.shape[0] != common_d:
            raise ValueError(f"Cannot clone node {node.id} of dim {node.mu.shape[0]} into {common_d}")
        mu[:] = np.asarray(node.mu, dtype=np.float64)
        sigma[:] = diag
    else:
        mu[slice_] = np.asarray(node.mu, dtype=np.float64)
        sigma[slice_] = diag
    return HFN(mu=mu, sigma=sigma, id=f"{prefix}{node.id}", use_diag=True)


def _register_clones_into_forest(
    forest: TieredForest,
    source_nodes: list[HFN],
    *,
    source_prior_ids: set[str],
    prefix: str,
    common_d: int,
    slice_: slice | None,
) -> set[str]:
    node_map: dict[str, HFN] = {}
    for node in source_nodes:
        clone = _clone_node(node, prefix=prefix, common_d=common_d, slice_=slice_)
        node_map[node.id] = clone
        forest.register(clone)
    for node in source_nodes:
        parent = node_map[node.id]
        for child in node.children():
            if child.id in node_map:
                parent.add_child(node_map[child.id])
        for edge in node.edges():
            if edge.source.id in node_map and edge.target.id in node_map:
                parent.add_edge(node_map[edge.source.id], node_map[edge.target.id], edge.relation)
    return {node_map[nid].id for nid in source_prior_ids if nid in node_map}


@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    max_hot: int
    common_d: int
    slice_: slice | None
    source_nodes: list[HFN] = field(default_factory=list)
    source_prior_ids: set[str] = field(default_factory=set)
    prefix: str = ""


class HFNWorker(mp.Process):
    def __init__(
        self,
        config: WorkerConfig,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
    ):
        super().__init__(name=f"HFNWorker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        # Initialize in child process
        if self.config.cold_dir.exists():
            shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(
            D=self.config.common_d,
            forest_id=self.config.forest_id,
            cold_dir=self.config.cold_dir,
            max_hot=self.config.max_hot,
        )
        
        # Register clones
        prior_ids = _register_clones_into_forest(
            self.forest,
            self.config.source_nodes,
            source_prior_ids=self.config.source_prior_ids,
            prefix=self.config.prefix,
            common_d=self.config.common_d,
            slice_=self.config.slice_
        )
        self.forest.set_protected(prior_ids)
        self.prior_ids = prior_ids
        
        tau = calibrate_tau(self.config.common_d, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
        self.observer = Observer(
            forest=self.forest,
            tau=tau,
            protected_ids=prior_ids,
            recombination_strategy="nearest_prior",
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

        while True:
            task = self.task_queue.get()
            if task is None: # Terminate
                break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE_BATCH":
                xs = task["xs"]
                explained_list = []
                for x in xs:
                    res = self.observer.observe(x)
                    self.forest._on_observe()
                    explained_list.append(bool(res.explanation_tree))
                self.result_queue.put({"type": "OBSERVE_BATCH_RES", "explained": explained_list})
            
            elif cmd == "STATS":
                total_nodes = len(self.forest)
                learned_total = total_nodes - len(self.prior_ids)
                learned_hot = sum(1 for n in self.forest.active_nodes() if n.id not in self.prior_ids)
                self.result_queue.put({
                    "type": "STATS_RES",
                    "total_nodes": total_nodes,
                    "learned_total": learned_total,
                    "learned_hot": learned_hot
                })
            
            elif cmd == "GET_ACTIVE_NODES":
                nodes = list(self.forest.active_nodes())
                self.result_queue.put({
                    "type": "ACTIVE_NODES_RES",
                    "nodes": nodes,
                    "prior_ids": self.prior_ids
                })

def _build_stage_plan() -> list[StageSpec]:
    return [
        StageSpec("math_seed", "pure", ("math",)),
        StageSpec("text_seed", "pure", ("text",)),
        StageSpec("math_text_mix", "mixed", ("math", "text")),
        StageSpec("math_revisit", "pure", ("math",)),
        StageSpec("text_revisit", "pure", ("text",)),
        StageSpec("math_text_mix_revisit", "mixed", ("math", "text")),
    ]

def _build_data() -> dict[str, Any]:
    math_samples = _sample_math(STAGE_SIZE, SEED)
    text_samples = _sample_text(STAGE_SIZE, SEED + 17)
    mixed_pairs = []
    for math_sample, text_sample in zip(math_samples, text_samples):
        mixed_pairs.append(
            {
                "math": math_sample,
                "text": text_sample,
                "full": _embed(math_sample, MATH_SLICE) + _embed(text_sample, TEXT_SLICE),
                "label": f"{math_sample.label}|{text_sample.label}",
            }
        )
    return {"math": math_samples, "text": text_samples, "mixed": mixed_pairs}

def _run_multi_process(
    label: str,
    worker_configs: dict[str, WorkerConfig],
    stage_plan: list[StageSpec],
    data: dict[str, Any],
    mixed_mode: str,
) -> SystemResult:
    process = psutil.Process()
    gc.collect()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    t0 = time.perf_counter()

    task_queues = {name: mp.Queue() for name in worker_configs}
    result_queue = mp.Queue()
    workers = {}
    
    for name, config in worker_configs.items():
        w = HFNWorker(config, task_queues[name], result_queue)
        w.start()
        workers[name] = w

    stage_results: list[StageResult] = []
    explained_total = 0
    obs_total = 0
    
    for stage in stage_plan:
        active_names = list(stage.domains)
        
        if stage.kind == "mixed" and mixed_mode == "cross_domain":
            key = tuple(stage.domains)
            mixed_name = f"mixed_{'_'.join(key)}"
            if mixed_name not in workers:
                # Gather state from specialists
                source_nodes = []
                source_prior_ids = set()
                for name in active_names:
                    task_queues[name].put({"cmd": "GET_ACTIVE_NODES"})
                    res = result_queue.get()
                    source_nodes.extend(res["nodes"])
                    source_prior_ids.update(res["prior_ids"])
                
                # Spawn mixed worker
                m_config = WorkerConfig(
                    name=mixed_name,
                    forest_id=f"{mixed_name}_lifecycle",
                    cold_dir=Path(f"data/hfn_mp_{mixed_name}_cold"),
                    max_hot=MIX_HOT,
                    common_d=COMMON_D,
                    slice_=None, # Use full
                    source_nodes=source_nodes,
                    source_prior_ids=source_prior_ids,
                    prefix=f"mix::{'_'.join(key)}::"
                )
                task_queues[mixed_name] = mp.Queue()
                w = HFNWorker(m_config, task_queues[mixed_name], result_queue)
                w.start()
                workers[mixed_name] = w
                worker_configs[mixed_name] = m_config
            
            active_names = active_names + [mixed_name]

        samples = data[stage.domains[0]] if stage.kind == "pure" else data["mixed"]
        stage_explained = 0
        
        # Before stage stats
        for name in active_names:
            task_queues[name].put({"cmd": "STATS"})
        
        before_nodes = 0
        for _ in active_names:
            res = result_queue.get()
            before_nodes += res["total_nodes"]

        # Batch processing
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i + BATCH_SIZE]
            obs_total += len(batch)
            
            # Put tasks for all active workers
            for name in active_names:
                xs = []
                for sample in batch:
                    if name.startswith("math"):
                        xs.append(_embed(sample, MATH_SLICE) if stage.kind == "pure" else _embed(sample["math"], MATH_SLICE))
                    elif name.startswith("text"):
                        xs.append(_embed(sample, TEXT_SLICE) if stage.kind == "pure" else _embed(sample["text"], TEXT_SLICE))
                    else: # mixed
                        xs.append(sample["full"])
                task_queues[name].put({"cmd": "OBSERVE_BATCH", "xs": xs})
            
            # Merge batch results
            batch_results = [] # list of list of bools
            for _ in active_names:
                res = result_queue.get()
                batch_results.append(res["explained"])
            
            # Any observer explained each sample in the batch
            for j in range(len(batch)):
                any_explained = False
                for k in range(len(active_names)):
                    if batch_results[k][j]:
                        any_explained = True
                        break
                if any_explained:
                    stage_explained += 1
                    explained_total += 1
            
            peak_rss = max(peak_rss, process.memory_info().rss)

        # After stage stats
        after_nodes = 0
        hot_learned = 0
        total_learned = 0
        for name in active_names:
            task_queues[name].put({"cmd": "STATS"})
        
        for _ in active_names:
            res = result_queue.get()
            after_nodes += res["total_nodes"]
            hot_learned += res["learned_hot"]
            total_learned += res["learned_total"]

        stage_results.append(
            StageResult(
                name=stage.name,
                observations=len(samples),
                explained=stage_explained,
                new_nodes=max(0, after_nodes - before_nodes),
                net_node_delta=after_nodes - before_nodes,
                hot_learned=hot_learned,
                total_learned=total_learned,
            )
        )

    # Final stats and cleanup
    total_nodes = 0
    learned_total = 0
    learned_hot = 0
    for name, w in workers.items():
        task_queues[name].put({"cmd": "STATS"})
        res = result_queue.get()
        total_nodes += res["total_nodes"]
        learned_total += res["learned_total"]
        learned_hot += res["learned_hot"]
        
        task_queues[name].put(None) # Terminate
        w.join()

    elapsed = time.perf_counter() - t0
    return SystemResult(
        label=label,
        elapsed_s=elapsed,
        peak_rss_delta_mb=(peak_rss - start_rss) / (1024 ** 2),
        coverage_pct=100.0 * explained_total / max(obs_total, 1),
        observations=obs_total,
        explained=explained_total,
        total_nodes=total_nodes,
        learned_total=learned_total,
        learned_hot=learned_hot,
        stage_results=stage_results,
    )

def _print_result(result: SystemResult) -> None:
    print()
    print("=" * 78)
    print(f"  {result.label.upper()} (MULTI-PROCESS)")
    print("=" * 78)
    print(f"  Coverage:               {result.coverage_pct:8.2f}%")
    print(f"  Obs/s:                  {result.observations / max(result.elapsed_s, 1e-9):8.1f}")
    print(f"  Peak RSS delta (MB):    {result.peak_rss_delta_mb:8.1f}")
    print(f"  Total nodes:            {result.total_nodes:8d}")
    print(f"  Learned nodes total:    {result.learned_total:8d}")
    print(f"  Learned nodes hot:      {result.learned_hot:8d}")
    print()
    print("  Stage summary")
    print("  " + "-" * 68)
    for stage in result.stage_results:
        print(
            f"  {stage.name:<24s} obs={stage.observations:4d}  explained={stage.explained:4d}  "
            f"new_nodes={stage.new_nodes:4d}  net={stage.net_node_delta:4d}  hot_learned={stage.hot_learned:4d}"
        )

def main() -> None:
    mp.set_start_method("spawn", force=True)
    print("Multi-process multi-observer lifecycle experiment")
    
    print("\nBuilding native world models ...")
    math_base, math_prior_ids = build_math_world_model(
        TieredForest,
        cold_dir=Path("data/hfn_lifecycle_math_base_cold"),
        max_hot=600,
    )
    text_base, text_prior_ids = build_nlp_world_model(
        forest_cls=TieredForest,
        cold_dir=Path("data/hfn_lifecycle_text_base_cold"),
        max_hot=500,
    )
    
    data = _build_data()
    stage_plan = _build_stage_plan()

    # Define systems
    systems = [
        ("specialists", "specialist_only"),
        ("specialists+mixed", "cross_domain"),
    ]

    for label, mixed_mode in systems:
        worker_configs = {
            "math": WorkerConfig(
                name="math",
                forest_id="math_spec",
                cold_dir=Path(f"data/hfn_lifecycle_{label}_math_cold"),
                max_hot=MATH_HOT,
                common_d=COMMON_D,
                slice_=MATH_SLICE,
                source_nodes=list(math_base.active_nodes()),
                source_prior_ids=math_prior_ids,
                prefix="math::"
            ),
            "text": WorkerConfig(
                name="text",
                forest_id="text_spec",
                cold_dir=Path(f"data/hfn_lifecycle_{label}_text_cold"),
                max_hot=TEXT_HOT,
                common_d=COMMON_D,
                slice_=TEXT_SLICE,
                source_nodes=list(text_base.active_nodes()),
                source_prior_ids=text_prior_ids,
                prefix="text::"
            )
        }

        result = _run_multi_process(
            label,
            worker_configs,
            stage_plan,
            data,
            mixed_mode=mixed_mode
        )
        _print_result(result)

if __name__ == "__main__":
    main()
