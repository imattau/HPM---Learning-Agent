"""
SP26: Emergent Routing (Decentralized Sovereignty).

Refactors SP25 to remove the central Governor routing table.
Specialists now use their own Forest's global identity as a "Competence Gate" 
to decide whether to process an observation.

Architecture:
1. Broadcaster (Governor): Sends all raw observations to all active workers.
2. Sovereign Worker: 
   - Uses Evaluator.accuracy(x, self.forest) for high-speed screening.
   - If competent (> threshold), runs full HPM expansion.
   - If not, remains dormant.
"""
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import time
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hfn import calibrate_tau, Evaluator
from hfn.decoder import Decoder

# --- Constants ---
D = 2
SEED = 42

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    competence_threshold: float = 0.05 # Accuracy required to claim observation
    source_nodes: list[HFN] = field(default_factory=list)

class SovereignWorker(mp.Process):
    def __init__(self, config: WorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"Worker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        if self.config.cold_dir.exists(): shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(D=D, forest_id=self.config.forest_id, 
                                   cold_dir=self.config.cold_dir, sweep_every=20)
        
        def register_deep(n, registry=None):
            if registry is None: registry = {}
            if n.id in registry: return registry[n.id]
            clone = HFN(mu=n.mu.copy(), sigma=n.sigma.copy(), id=n.id, use_diag=n.use_diag)
            registry[n.id] = clone
            for child in n.children():
                child_clone = register_deep(child, registry)
                clone.add_child(child_clone)
            self.forest.register(clone)
            return clone

        for node in self.config.source_nodes:
            register_deep(node)
            
        tau = calibrate_tau(D, sigma_scale=1.0, margin=5.0)
        self.evaluator = Evaluator()
        
        self.observer = Observer(
            forest=self.forest, tau=tau,
            adaptive_compression=True,
            compression_cooccurrence_threshold=2,
            persistence_guided_absorption=True, lacunarity_guided_creation=True,
            lacunarity_creation_radius=0.5, node_use_diag=True,
            residual_surprise_threshold=0.5,
            node_prefix=self.config.name + "_",
            weight_decay_rate=0.01 
        )

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE":
                x = task["x"]
                # --- PHASE A: High-Speed Competence Screen ---
                # We use the Forest's global identity (mean mu/sigma) as a filter
                comp = self.evaluator.accuracy(x, self.forest)
                
                if comp < self.config.competence_threshold:
                    # Not my domain. Return negative response immediately.
                    self.result_queue.put({
                        "name": self.config.name, 
                        "explained": False, 
                        "competent": False,
                        "accuracy": comp
                    })
                else:
                    # --- PHASE B: Deep Thought ---
                    res = self.observer.observe(x)
                    self.forest._on_observe()
                    self.result_queue.put({
                        "name": self.config.name, 
                        "explained": bool(res.explanation_tree), 
                        "competent": True,
                        "accuracy": comp
                    })
                
            elif cmd == "STATS":
                self.result_queue.put({
                    "name": self.config.name, 
                    "total_nodes": len(self.forest),
                    "hot_count": self.forest.hot_count()
                })
                
            elif cmd == "GET_MATURE_NODES":
                mature = []
                for n in self.forest.active_nodes():
                    if not n.is_leaf():
                        weight = self.observer._weights.get(n.id, 0.0)
                        if weight >= 0.1:
                            mature.append({"node": n, "weight": weight})
                self.result_queue.put({"name": self.config.name, "mature": mature})

def run_experiment():
    print("SP26: Emergent Routing Experiment (Decentralized Sovereignty)\n")
    mp.set_start_method("spawn", force=True)

    # --- 1. Initial State: Explorer ONLY ---
    task_queues = {"Explorer": mp.Queue()}
    result_queue = mp.Queue()
    # Explorer has very low competence threshold (it's a generalist)
    workers = {"Explorer": SovereignWorker(
        WorkerConfig("Explorer", "exp_0", Path("data/sp26_exp"), competence_threshold=0.0),
        task_queues["Explorer"], result_queue
    )}
    workers["Explorer"].start()
    print("  [+] Explorer process started (Generalist, Threshold=0.0).")

    # --- 2. Phase 1: Local Maturation ---
    print("\n--- Phase 1: Local Concept Maturation ---")
    rng = np.random.default_rng(SEED)
    for i in range(400):
        x = rng.normal(5.0, 0.1, D) if rng.random() > 0.1 else rng.uniform(-10, 10, D)
        task_queues["Explorer"].put({"cmd": "OBSERVE", "x": x})
    for _ in range(400): result_queue.get()

    # --- 3. Phase 2: Autonomous Promotion ---
    print("\n--- Phase 2: Promotion via Extraction ---")
    task_queues["Explorer"].put({"cmd": "GET_MATURE_NODES"})
    res = result_queue.get()
    mature_list = res["mature"]
    
    if mature_list:
        best_mature = sorted(mature_list, key=lambda x: x["weight"], reverse=True)[0]["node"]
        print(f"  [!] Governor identified persistent concept: {best_mature.id}")
        
        # Sub-tree extraction (simplified)
        source_nodes = [best_mature] + best_mature.children()
        
        # Promote to Sovereign Worker with a TIGHT competence threshold
        s_name = "Spec_Signal"
        task_queues[s_name] = mp.Queue()
        # Threshold 0.05 means it will only claim points within ~3 sigma of its mean
        workers[s_name] = SovereignWorker(
            WorkerConfig(s_name, "spec_1", Path("data/sp26_spec"), competence_threshold=0.05, source_nodes=source_nodes),
            task_queues[s_name], result_queue
        )
        workers[s_name].start()
        print(f"  [+] {s_name} promoted with Competence Threshold 0.05.")
    else:
        print("  [FAIL] No mature concepts found."); return

    # --- 4. Phase 3: Decentralized Execution ---
    print("\n--- Phase 3: Decentralized Routing (Broadcasting) ---")
    print("  Governor will now broadcast ALL points to ALL workers.")
    
    claim_stats = {name: 0 for name in workers}
    
    for i in range(100):
        # Mixed data
        is_signal = rng.random() > 0.5
        x = rng.normal(5.0, 0.1, D) if is_signal else rng.uniform(-10, 10, D)
        
        # BROADCAST: No routing table used here
        for name in workers:
            task_queues[name].put({"cmd": "OBSERVE", "x": x})
            
        for _ in workers:
            res = result_queue.get()
            if res["competent"]:
                claim_stats[res["name"]] += 1

    print("\n--- Emergent Routing Results (Broadcast Phase) ---")
    for name, claims in claim_stats.items():
        print(f"  {name:15s} claimed {claims}/100 observations.")

    # --- 5. Final Topography ---
    print("\n--- Final Topography ---")
    for name in workers:
        task_queues[name].put({"cmd": "STATS"})
        res = result_queue.get()
        print(f"  {name:15s} Total Nodes: {res['total_nodes']:4d}")

    for name in workers: task_queues[name].put(None)
    for name in workers: workers[name].join()
    print(f"\nExperiment concluded.")

if __name__ == "__main__":
    run_experiment()
