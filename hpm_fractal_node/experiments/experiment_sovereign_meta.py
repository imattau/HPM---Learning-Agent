"""Sovereign Meta-Hierarchy Experiment (Multi-Tier).

Implements SP19 architecture:
- Tier 1: Perceptual Specialists (Spatial, Symbolic).
- Tier 2: Relational Synthesizer (L2 Bridge).

The L2 Bridge observes the concatenated "Explanation Winners" from L1 
to stabilize cross-domain identities (e.g. Rotate 90 <-> Count 1).

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_meta.py
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
from hpm_fractal_node.arc.arc_rosetta_generator import generate_rosetta_samples
from hpm_fractal_node.arc.arc_sovereign_loader import COMMON_D, S_SLICE, M_SLICE
from hpm_fractal_node.arc.arc_prior_forest import build_prior_forest
from hpm_fractal_node.math.math_world_model import build_math_world_model

SEED = 42
OFFSLICE_VAR = 1.0

# L2 Dimensions: Spatial (100) + Symbolic (30)
L2_D = 130 

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    max_hot: int
    degree: float
    tau_sigma: float
    common_d: int = COMMON_D
    source_nodes: list[HFN] = field(default_factory=list)
    source_prior_ids: set[str] = field(default_factory=set)
    prefix: str = ""
    slice_: slice | None = None

def _clone_node(node: HFN, *, prefix: str, common_d: int, slice_: slice | None) -> HFN:
    mu = np.zeros(common_d, dtype=np.float64)
    sigma = np.full(common_d, OFFSLICE_VAR, dtype=np.float64)
    src_mu = np.asarray(node.mu, dtype=np.float64)
    src_diag = np.asarray(node.sigma, dtype=np.float64) if node.use_diag else np.diag(node.sigma)

    if slice_ is None:
        r = min(common_d, src_mu.shape[0])
        mu[:r], sigma[:r] = src_mu[:r], src_diag[:r]
    else:
        start, stop = slice_.start, slice_.stop
        r = min(stop - start, src_mu.shape[0])
        mu[start : start + r], sigma[start : start + r] = src_mu[:r], src_diag[:r]
    return HFN(mu=mu, sigma=sigma, id=f"{prefix}{node.id}", use_diag=True)

def _register_clones_into_forest(forest: TieredForest, source_nodes: list[HFN], **kwargs) -> set[str]:
    node_map: dict[str, HFN] = {}
    source_prior_ids = kwargs.pop("source_prior_ids", [])
    for node in source_nodes:
        clone = _clone_node(node, **kwargs)
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
        if self.config.cold_dir.exists(): shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(D=self.config.common_d, forest_id=self.config.forest_id,
                                   cold_dir=self.config.cold_dir, max_hot=self.config.max_hot)
        
        prior_ids = _register_clones_into_forest(
            self.forest, self.config.source_nodes, prefix=self.config.prefix,
            common_d=self.config.common_d, slice_=self.config.slice_,
            source_prior_ids=self.config.source_prior_ids
        )
        if self.config.degree > 0.0: self.forest.set_protected(prior_ids)

        tau = calibrate_tau(self.config.common_d, sigma_scale=self.config.tau_sigma, margin=5.0)
        self.observer = Observer(
            forest=self.forest, tau=tau, protected_ids=prior_ids if self.config.degree > 0.0 else set(),
            recombination_strategy="nearest_prior", hausdorff_absorption_threshold=0.35,
            persistence_guided_absorption=True, lacunarity_guided_creation=True,
            lacunarity_creation_radius=0.1, node_use_diag=True,
            residual_surprise_threshold=2.0
        )

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE":
                res = self.observer.observe(task["x"])
                self.forest._on_observe()
                # Return the mu of the best matching node (Explanation Winner)
                winner_mu = None
                if res.explanation_tree:
                    # Explanation tree is a list of HFN objects
                    winner_mu = res.explanation_tree[0].mu
                self.result_queue.put({"name": self.config.name, "winner_mu": winner_mu})
                
            elif cmd == "STATS":
                self.result_queue.put({"name": self.config.name, "total_nodes": len(self.forest)})

def main():
    mp.set_start_method("spawn", force=True)
    print("SP19: Sovereign Meta-Hierarchy Experiment")
    
    # 1. Build L1 Priors
    math_base, math_prior_ids = build_math_world_model(TieredForest, Path("data/smeta_math_cold"), 600)
    spatial_forest, spatial_prior_registry = build_prior_forest(10, 10)
    spatial_prior_ids = set(spatial_prior_registry.keys())

    # 2. Worker Configs
    configs = [
        WorkerConfig(name="L1_Spatial", forest_id="smeta_l1_s", cold_dir=Path("data/smeta_run_l1_s"), 
                     max_hot=400, degree=1.0, tau_sigma=1.0, source_nodes=list(spatial_forest.active_nodes()),
                     source_prior_ids=spatial_prior_ids, prefix="s::", slice_=S_SLICE),
        WorkerConfig(name="L1_Symbolic", forest_id="smeta_l1_m", cold_dir=Path("data/smeta_run_l1_m"), 
                     max_hot=600, degree=1.0, tau_sigma=1.2, source_nodes=list(math_base.active_nodes()),
                     source_prior_ids=math_prior_ids, prefix="m::", slice_=M_SLICE),
        WorkerConfig(name="L2_Synthesizer", forest_id="smeta_l2", cold_dir=Path("data/smeta_run_l2"), 
                     max_hot=500, degree=0.0, tau_sigma=0.5, common_d=L2_D, source_nodes=[], prefix="l2::")
    ]

    task_queues = {c.name: mp.Queue() for c in configs}
    result_queue = mp.Queue()
    workers = {c.name: SovereignWorker(c, task_queues[c.name], result_queue) for c in configs}
    for w in workers.values(): w.start()

    # 3. Generate Rosetta Dataset
    samples = generate_rosetta_samples(n_per_rule=30, seed=SEED)
    print(f"Loaded {len(samples)} Rosetta samples (Count-Governed Rotation).\n")

    t0 = time.perf_counter()
    l2_success = 0
    l1_s_success = 0
    l1_m_success = 0

    print("--- Commencing Two-Tier Routing ---")
    for i, s in enumerate(samples):
        # Step A: Tier 1 (Parallel)
        task_queues["L1_Spatial"].put({"cmd": "OBSERVE", "x": s["vec"]})
        task_queues["L1_Symbolic"].put({"cmd": "OBSERVE", "x": s["vec"]})
        
        l1_res = {}
        for _ in range(2):
            r = result_queue.get()
            l1_res[r["name"]] = r["winner_mu"]

        mu_s = l1_res["L1_Spatial"]
        mu_m = l1_res["L1_Symbolic"]
        
        if mu_s is not None: l1_s_success += 1
        if mu_m is not None: l1_m_success += 1

        # Step B: Tier 2 (Synthesis)
        # If both L1 specialists identified a prior, we synthesize the L2 message
        if mu_s is not None and mu_m is not None:
            # We only send the relevant slices from the L1 winner mus
            # mu_s is COMMON_D (256), mu_m is COMMON_D (256)
            # L2 expects 130D: Spatial(100) + Symbolic(30)
            msg_l2 = np.concatenate([mu_s[S_SLICE], mu_m[M_SLICE]])
            
            task_queues["L2_Synthesizer"].put({"cmd": "OBSERVE", "x": msg_l2})
            l2_res = result_queue.get()
            if l2_res["winner_mu"] is not None:
                l2_success += 1

        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")

    # 4. Final Report
    print("\n--- SP19 Meta-Hierarchy Report ---")
    print(f"  L1 Spatial Accuracy:  {l1_s_success}/{len(samples)} ({100*l1_s_success/len(samples):.1f}%)")
    print(f"  L1 Symbolic Accuracy: {l1_m_success}/{len(samples)} ({100*l1_m_success/len(samples):.1f}%)")
    print(f"  L2 Synthesized:       {l2_success}/{len(samples)} (Attempted only on L1 double-match)")
    
    for name in workers:
        task_queues[name].put({"cmd": "STATS"})
        res = result_queue.get()
        print(f"  {name:20s} Total Nodes: {res['total_nodes']:4d}")

    # Shutdown
    for name in workers: task_queues[name].put(None)
    for name in workers: workers[name].join()
    print(f"\nExperiment concluded in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()
