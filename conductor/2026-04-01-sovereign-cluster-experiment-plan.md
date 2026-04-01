# Implementation Plan: Sovereign Cluster Experiment

## Objective
Develop a new multi-process experiment (`experiment_sovereign_cluster.py`) that demonstrates the scaling architecture of the "Sovereign AI". It will implement a simplified 5-process cluster based on the "Foundational Five" domains and the "Sovereignty Spectrum".

## Architecture
The experiment will use a 256-dimensional common latent space, sliced into distinct domains:
1.  **Math Specialist** (Degree 1.0): Rigid, uses protected priors. Explains Math slice.
2.  **Text Specialist** (Degree 0.5): Adaptive, partial protection. Explains Text slice.
3.  **Spatial Specialist** (Degree 1.0): Axiomatic spatial geometry. Explains Spatial slice.
4.  **Explorer** (Degree 0.0): Zero priors, high $\tau$. Catches cross-domain and novel "Residual Surprise".
5.  **Governor/Controller**: Orchestrates the multi-process batching and tracks "Stereo Vision" (when multiple specialists explain the same observation).

## Implementation Steps

1.  **Create `hpm_fractal_node/experiments/experiment_sovereign_cluster.py`**
    *   Define the 256D latent space and slices.
    *   Implement data generators for Math, Text, synthetic Spatial, and Mixed/Novelty data.
    *   Implement the `WorkerConfig` and `SovereignWorker` classes (inheriting from `multiprocessing.Process`) to run isolated `TieredForest` and `Observer` instances.
    *   Implement the `main` Governor loop that routes data batches to all workers simultaneously.
    *   Track specialist explanations, Explorer catch-rates, and Stereo Vision overlaps.

## Verification
*   Run `PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_cluster.py`
*   Verify that 5 processes are launched.
*   Verify that the Math, Text, and Spatial specialists effectively explain their respective domain grounding stages.
*   Verify that the Explorer node count increases significantly during the "Novelty Shock" stage, acting as the system's "nursery" for unexplained data.

## Code Draft

```python
\"\"\"Sovereign Cluster Experiment (Multi-Process).

Implements a scaled "8-Process Cluster" architecture (simplified to 5 core processes
for this experiment) focusing on the "Foundational Domains" and the "Sovereignty Spectrum".

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_cluster.py
\"\"\"
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

# We define a 256D common latent space
SPATIAL_D = 40
COMMON_D = MATH_D + TEXT_D + SPATIAL_D
MATH_SLICE = slice(0, MATH_D)
TEXT_SLICE = slice(MATH_D, MATH_D + TEXT_D)
SPATIAL_SLICE = slice(MATH_D + TEXT_D, COMMON_D)

OFFSLICE_VAR = 1.0

@dataclass(frozen=True)
class DomainSample:
    vec: np.ndarray
    label: str
    source_domain: str

def _sample_math(n: int, seed: int) -> list[DomainSample]:
    data = generate_observations(n=n, seed=seed)
    out: list[DomainSample] = []
    for vec, (left, op, right, result) in data:
        full_vec = np.zeros(COMMON_D, dtype=np.float64)
        full_vec[MATH_SLICE] = vec.astype(np.float64)
        out.append(DomainSample(vec=full_vec, label="math", source_domain="math"))
    return out

def _sample_text(n: int, seed: int) -> list[DomainSample]:
    data = generate_sentences(seed=seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))[:n]
    out: list[DomainSample] = []
    for idx in order:
        vec, _, _ = data[idx]
        full_vec = np.zeros(COMMON_D, dtype=np.float64)
        full_vec[TEXT_SLICE] = vec.astype(np.float64)
        out.append(DomainSample(vec=full_vec, label="text", source_domain="text"))
    return out

def _sample_spatial(n: int, seed: int) -> list[DomainSample]:
    rng = np.random.default_rng(seed)
    out: list[DomainSample] = []
    for _ in range(n):
        vec = rng.uniform(-1, 1, size=SPATIAL_D)
        vec[:5] += 2.0 
        full_vec = np.zeros(COMMON_D, dtype=np.float64)
        full_vec[SPATIAL_SLICE] = vec
        out.append(DomainSample(vec=full_vec, label="spatial", source_domain="spatial"))
    return out

def _sample_mixed(n: int, seed: int) -> list[DomainSample]:
    rng = np.random.default_rng(seed)
    out: list[DomainSample] = []
    for _ in range(n):
        full_vec = rng.uniform(-5, 5, size=COMMON_D)
        out.append(DomainSample(vec=full_vec, label="novelty", source_domain="novelty"))
    return out

def _clone_node(node: HFN, *, prefix: str, common_d: int, slice_: slice | None) -> HFN:
    mu = np.zeros(common_d, dtype=np.float64)
    sigma = np.full(common_d, OFFSLICE_VAR, dtype=np.float64)
    if node.use_diag:
        diag = np.asarray(node.sigma, dtype=np.float64)
    else:
        sigma_arr = np.asarray(node.sigma, dtype=np.float64)
        diag = np.diag(sigma_arr) if sigma_arr.ndim == 2 else sigma_arr
    if slice_ is None:
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
    return {node_map[nid].id for nid in source_prior_ids if nid in node_map}

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

        tau = calibrate_tau(COMMON_D, sigma_scale=self.config.tau_sigma, margin=5.0)
        creation_radius = 0.08 if self.config.degree > 0.5 else 0.15
        
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
            lacunarity_creation_radius=creation_radius,
            multifractal_guided_absorption=False,
            gap_query_threshold=None,
            max_expand_depth=2,
            node_use_diag=True,
        )

        while True:
            task = self.task_queue.get()
            if task is None:
                break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE_BATCH":
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

def _build_spatial_priors() -> tuple[list[HFN], set[str]]:
    nodes = []
    priors = set()
    rng = np.random.default_rng(42)
    for i in range(5):
        mu = rng.uniform(-1, 1, size=SPATIAL_D)
        mu[:5] += 2.0
        sigma = np.ones(SPATIAL_D) * 0.1
        node = HFN(mu=mu, sigma=sigma, id=f"spatial_prior_{i}", use_diag=True)
        nodes.append(node)
        priors.add(node.id)
    return nodes, priors

def main():
    mp.set_start_method("spawn", force=True)
    print("Sovereign Cluster Experiment (Multi-Process)")
    print("Initializing the Systemic Octet (Simulated Foundation 5)...\\n")

    math_base, math_prior_ids = build_math_world_model(TieredForest, Path("data/sc_math_cold"), 600)
    text_base, text_prior_ids = build_nlp_world_model(TieredForest, Path("data/sc_text_cold"), 500)
    spatial_base, spatial_prior_ids = _build_spatial_priors()

    worker_configs = [
        WorkerConfig(
            name="Math_Specialist",
            forest_id="sc_math", cold_dir=Path("data/sc_run_math"), max_hot=300,
            degree=1.0, tau_sigma=1.0, source_nodes=list(math_base.active_nodes()),
            source_prior_ids=math_prior_ids, prefix="math::", slice_=MATH_SLICE
        ),
        WorkerConfig(
            name="Text_Specialist",
            forest_id="sc_text", cold_dir=Path("data/sc_run_text"), max_hot=300,
            degree=0.5, tau_sigma=1.5, source_nodes=list(text_base.active_nodes()),
            source_prior_ids=text_prior_ids, prefix="text::", slice_=TEXT_SLICE
        ),
        WorkerConfig(
            name="Spatial_Specialist",
            forest_id="sc_spatial", cold_dir=Path("data/sc_run_spatial"), max_hot=200,
            degree=1.0, tau_sigma=1.0, source_nodes=spatial_base,
            source_prior_ids=spatial_prior_ids, prefix="spatial::", slice_=SPATIAL_SLICE
        ),
        WorkerConfig(
            name="Explorer",
            forest_id="sc_explorer", cold_dir=Path("data/sc_run_explorer"), max_hot=500,
            degree=0.0, tau_sigma=2.5, source_nodes=[], source_prior_ids=set(), prefix="exp::"
        )
    ]

    task_queues = {c.name: mp.Queue() for c in worker_configs}
    result_queue = mp.Queue()
    workers = {}

    print("Spawning Sovereign Workers...")
    for c in worker_configs:
        w = SovereignWorker(c, task_queues[c.name], result_queue)
        w.start()
        workers[c.name] = w
        print(f"  [+] {c.name} (Degree: {c.degree})")

    stages = {
        "1. Domain Grounding": _sample_math(50, 1) + _sample_text(50, 2) + _sample_spatial(50, 3),
        "2. The Novelty Shock": _sample_mixed(100, 4),
        "3. Cross-Domain Synthesis": _sample_math(20, 5) + _sample_mixed(50, 6) + _sample_spatial(20, 7)
    }

    t0 = time.perf_counter()
    
    print("\\n--- Commencing Governor Routing ---")
    for stage_name, samples in stages.items():
        print(f"\\n>> Stage: {stage_name} (Samples: {len(samples)})")
        
        for name in workers:
            task_queues[name].put({"cmd": "STATS"})
        nodes_before = {res["name"]: res["total_nodes"] for _ in range(len(workers)) if (res := result_queue.get())}

        stage_obs = 0
        stage_spec_exp = 0
        stage_exp_exp = 0
        stage_stereo = 0
        
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i + BATCH_SIZE]
            xs = [s.vec for s in batch]
            stage_obs += len(xs)

            for name in workers:
                task_queues[name].put({"cmd": "OBSERVE_BATCH", "xs": xs})

            batch_res = {name: [] for name in workers}
            for _ in workers:
                res = result_queue.get()
                batch_res[res["name"]] = res["explained"]

            for j in range(len(xs)):
                spec_claims = [name for name in ["Math_Specialist", "Text_Specialist", "Spatial_Specialist"] if batch_res[name][j]]
                exp_claim = batch_res["Explorer"][j]

                if len(spec_claims) > 0:
                    stage_spec_exp += 1
                    if len(spec_claims) > 1:
                        stage_stereo += 1
                elif exp_claim:
                    stage_exp_exp += 1

        for name in workers:
            task_queues[name].put({"cmd": "STATS"})
        nodes_after = {res["name"]: res["total_nodes"] for _ in range(len(workers)) if (res := result_queue.get())}
        
        explorer_new = nodes_after["Explorer"] - nodes_before["Explorer"]
        
        print(f"  Specialist Explanations: {stage_spec_exp}/{stage_obs}")
        print(f"  Explorer Explanations:   {stage_exp_exp}/{stage_obs}")
        print(f"  Stereo Vision Claims:    {stage_stereo}")
        print(f"  New Explorer Nodes:      {explorer_new}")

    print("\\n--- Final Systemic Topography ---")
    for name in workers:
        task_queues[name].put({"cmd": "STATS"})
        res = result_queue.get()
        print(f"  {name:20s} Total Nodes: {res['total_nodes']:4d}")
        
    for name in workers:
        task_queues[name].put(None)
    for name in workers:
        workers[name].join()

    print(f"\\nExperiment concluded in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()
```
