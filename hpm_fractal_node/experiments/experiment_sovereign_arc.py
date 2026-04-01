"""Sovereign ARC Cluster Experiment (Multi-Process).

Evaluates ARC-AGI-2 tasks using a distributed cluster of specialists:
1. Spatial Specialist: Focused on pixel-wise 10x10 deltas.
2. Symbolic Specialist: Focused on numerical invariants (counts, dimensions).
3. Explorer: Captures cross-modal novelty and residuals.

The Governor tracks "Stereo Vision" (simultaneous domain recognition) 
to identify complex mapping rules.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_arc.py
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
from hpm_fractal_node.arc.arc_sovereign_loader import (
    load_sovereign_tasks, COMMON_D, S_SLICE, M_SLICE, C_SLICE
)
from hpm_fractal_node.arc.arc_prior_forest import build_prior_forest
from hpm_fractal_node.math.math_world_model import build_math_world_model

SEED = 42
OFFSLICE_VAR = 1.0

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    max_hot: int
    degree: float
    tau_sigma: float
    source_nodes: list[HFN] = field(default_factory=list)
    source_prior_ids: set[str] = field(default_factory=set)
    prefix: str = ""
    slice_: slice | None = None

def _clone_node(node: HFN, *, prefix: str, common_d: int, slice_: slice | None) -> HFN:
    mu = np.zeros(common_d, dtype=np.float64)
    sigma = np.full(common_d, OFFSLICE_VAR, dtype=np.float64)
    
    # Extract source mu and diag sigma
    src_mu = np.asarray(node.mu, dtype=np.float64)
    if node.use_diag:
        src_diag = np.asarray(node.sigma, dtype=np.float64)
    else:
        sigma_arr = np.asarray(node.sigma, dtype=np.float64)
        src_diag = np.diag(sigma_arr) if sigma_arr.ndim == 2 else sigma_arr

    if slice_ is None:
        # Full copy if no slice
        r = min(common_d, src_mu.shape[0])
        mu[:r] = src_mu[:r]
        sigma[:r] = src_diag[:r]
    else:
        # Targeted slice copy
        start, stop = slice_.start, slice_.stop
        length = stop - start
        r = min(length, src_mu.shape[0])
        mu[start : start + r] = src_mu[:r]
        sigma[start : start + r] = src_diag[:r]
        
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
    return {node_map[nid].id for nid in source_prior_ids if nid in node_map}

class SovereignWorker(mp.Process):
    def __init__(self, config: WorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"SovereignWorker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        if self.config.cold_dir.exists():
            shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(
            D=COMMON_D,
            forest_id=self.config.forest_id,
            cold_dir=self.config.cold_dir,
            max_hot=self.config.max_hot,
        )
        
        prior_ids = _register_clones_into_forest(
            self.forest,
            self.config.source_nodes,
            source_prior_ids=self.config.source_prior_ids,
            prefix=self.config.prefix,
            common_d=COMMON_D,
            slice_=self.config.slice_
        )
        
        if self.config.degree > 0.0:
            self.forest.set_protected(prior_ids)
        else:
            self.forest.set_protected(set())

        # Calibration based on slice dimension
        slice_d = (self.config.slice_.stop - self.config.slice_.start) if self.config.slice_ else COMMON_D
        tau = calibrate_tau(COMMON_D, sigma_scale=self.config.tau_sigma, margin=5.0)
        
        self.observer = Observer(
            forest=self.forest,
            tau=tau,
            protected_ids=prior_ids if self.config.degree > 0.0 else set(),
            recombination_strategy="nearest_prior",
            hausdorff_absorption_threshold=0.35,
            hausdorff_absorption_weight_floor=0.4,
            absorption_miss_threshold=20,
            persistence_guided_absorption=True,
            lacunarity_guided_creation=True,
            lacunarity_creation_radius=0.1,
            multifractal_guided_absorption=False,
            gap_query_threshold=None,
            max_expand_depth=2,
            node_use_diag=True,
        )

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE_TASK":
                xs = task["xs"]
                explained_list = []
                for x in xs:
                    res = self.observer.observe(x)
                    self.forest._on_observe()
                    explained_list.append(bool(res.explanation_tree))
                self.result_queue.put({"name": self.config.name, "explained": explained_list})
                
            elif cmd == "STATS":
                self.result_queue.put({
                    "name": self.config.name,
                    "total_nodes": len(self.forest),
                    "hot_nodes": len(self.forest.active_nodes())
                })

def main():
    mp.set_start_method("spawn", force=True)
    print("Sovereign ARC Cluster Experiment")
    print("Loading ARC Puzzles and World Models...\n")

    tasks = load_sovereign_tasks()
    print(f"Loaded {len(tasks)} ARC tasks.")

    math_base, math_prior_ids = build_math_world_model(TieredForest, Path("data/sarc_math_cold"), 600)
    spatial_forest, spatial_prior_registry = build_prior_forest(10, 10)
    spatial_prior_ids = set(spatial_prior_registry.keys())

    worker_configs = [
        WorkerConfig(
            name="Spatial_Spec",
            forest_id="sarc_spatial", cold_dir=Path("data/sarc_run_spatial"), max_hot=400,
            degree=1.0, tau_sigma=1.0, source_nodes=list(spatial_forest.active_nodes()),
            source_prior_ids=spatial_prior_ids, prefix="spatial::", slice_=S_SLICE
        ),
        WorkerConfig(
            name="Symbolic_Spec",
            forest_id="sarc_symbolic", cold_dir=Path("data/sarc_run_symbolic"), max_hot=600,
            degree=1.0, tau_sigma=1.2, source_nodes=list(math_base.active_nodes()),
            source_prior_ids=math_prior_ids, prefix="math::", slice_=M_SLICE
        ),
        WorkerConfig(
            name="Explorer",
            forest_id="sarc_explorer", cold_dir=Path("data/sarc_run_explorer"), max_hot=500,
            degree=0.0, tau_sigma=2.5, source_nodes=[], source_prior_ids=set(), prefix="exp::"
        )
    ]

    task_queues = {c.name: mp.Queue() for c in worker_configs}
    result_queue = mp.Queue()
    workers = {}

    print("Spawning Sovereign ARC Workers...")
    for c in worker_configs:
        w = SovereignWorker(c, task_queues[c.name], result_queue)
        w.start()
        workers[c.name] = w
        print(f"  [+] {c.name}")

    t0 = time.perf_counter()
    stereo_vision_count = 0
    spatial_dominant = 0
    symbolic_dominant = 0
    explorer_catch = 0
    total_obs = 0

    # Limit to first 200 tasks for verification
    limit = 200 
    print(f"\n--- Processing {limit} Tasks ---")
    
    for task in tasks[:limit]:
        xs = [ex["vec"] for ex in task["train"]]
        total_obs += len(xs)

        for name in workers:
            task_queues[name].put({"cmd": "OBSERVE_TASK", "xs": xs})

        batch_res = {name: [] for name in workers}
        for _ in workers:
            res = result_queue.get()
            batch_res[res["name"]] = res["explained"]

        # Aggregate puzzle-level explanation
        task_spatial = any(batch_res["Spatial_Spec"])
        task_symbolic = any(batch_res["Symbolic_Spec"])
        task_explorer = any(batch_res["Explorer"])

        if task_spatial and task_symbolic:
            stereo_vision_count += 1
        elif task_spatial:
            spatial_dominant += 1
        elif task_symbolic:
            symbolic_dominant += 1
        elif task_explorer:
            explorer_catch += 1

    print("\n--- Sovereign ARC Report ---")
    print(f"  Tasks Evaluated:      {limit}")
    print(f"  Stereo Vision Tasks:  {stereo_vision_count} (Spatial + Symbolic)")
    print(f"  Spatial Dominant:     {spatial_dominant}")
    print(f"  Symbolic Dominant:    {symbolic_dominant}")
    print(f"  Explorer Catch:       {explorer_catch}")
    print(f"  Unexplained:          {limit - (stereo_vision_count + spatial_dominant + symbolic_dominant + explorer_catch)}")

    # Shutdown
    for name in workers: task_queues[name].put(None)
    for name in workers: workers[name].join()

    print(f"\nExperiment concluded in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()
