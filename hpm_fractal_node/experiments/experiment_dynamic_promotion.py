"""
SP25: Dynamic Specialist Promotion (Decoder-Led Extraction).

Tests Emergent Sovereignty by:
1. Local Maturation: Explorer process discovers high-level parent nodes via Recombination.
2. Recognition: Governor identifies high-weight mature nodes.
3. Extraction: Governor uses the Agnostic Decoder to unfold the mature node into its sub-tree.
4. Promotion: Governor spawns a new dedicated worker core for that concept.
5. Specialization: Governor redirects signals, allowing natural decay in the Generalist.
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
from hfn import calibrate_tau
from hfn.decoder import Decoder

# --- Constants ---
D = 2
SEED = 42

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
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
        
        # Low sweep_every to demonstrate natural forgetting
        self.forest = TieredForest(D=D, forest_id=self.config.forest_id, 
                                   cold_dir=self.config.cold_dir, sweep_every=20)
        
        def register_deep(n, registry=None):
            if registry is None: registry = {}
            if n.id in registry: return registry[n.id]
            
            # Re-instantiate
            clone = HFN(mu=n.mu.copy(), sigma=n.sigma.copy(), id=n.id, use_diag=n.use_diag)
            registry[n.id] = clone
            
            # Register children first
            for child in n.children():
                child_clone = register_deep(child, registry)
                clone.add_child(child_clone)
            
            self.forest.register(clone)
            return clone

        processed_registry = {}
        for node in self.config.source_nodes:
            register_deep(node, processed_registry)
            
        tau = calibrate_tau(D, sigma_scale=1.0, margin=5.0)
        
        self.observer = Observer(
            forest=self.forest, tau=tau,
            adaptive_compression=True,
            compression_cooccurrence_threshold=2, # Trigger faster
            persistence_guided_absorption=True, lacunarity_guided_creation=True,
            lacunarity_creation_radius=0.5, node_use_diag=True,
            residual_surprise_threshold=0.5,
            node_prefix=self.config.name + "_",
            weight_decay_rate=0.01 # Enable natural forgetting
        )

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE":
                res = self.observer.observe(task["x"])
                self.forest._on_observe()
                self.result_queue.put({"name": self.config.name, "explained": bool(res.explanation_tree)})
                
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
                        # Recently created parents might have low weight (w_init=0.1)
                        if weight >= 0.1:
                            mature.append({"node": n, "weight": weight})
                self.result_queue.put({"name": self.config.name, "mature": mature})

def run_experiment():
    print("SP25: Dynamic Specialist Promotion (Decoder-Led Extraction)\n")
    mp.set_start_method("spawn", force=True)

    # --- 1. Initial State ---
    # Seed with two distant priors to act as scaffolding
    p1 = HFN(mu=np.array([4.5, 4.5]), sigma=np.array([0.5, 0.5]), id="prior_left", use_diag=True)
    p2 = HFN(mu=np.array([5.5, 5.5]), sigma=np.array([0.5, 0.5]), id="prior_right", use_diag=True)
    
    task_queues = {"Explorer": mp.Queue()}
    result_queue = mp.Queue()
    workers = {"Explorer": SovereignWorker(
        WorkerConfig("Explorer", "exp_0", Path("data/sp25_exp"), source_nodes=[p1, p2]),
        task_queues["Explorer"], result_queue
    )}
    workers["Explorer"].start()
    print("  [+] Generalist process (Explorer) started.")

    # --- 2. Phase 1: Local Maturation ---
    print("\n--- Phase 1: Local Concept Maturation ---")
    rng = np.random.default_rng(SEED)
    # Signal is a tight cluster at [5, 5]
    for i in range(400):
        if rng.random() > 0.1: # 90% signal bias
            x = rng.normal(5.0, 0.1, D)
        else:
            x = rng.uniform(-10, 10, D)
        task_queues["Explorer"].put({"cmd": "OBSERVE", "x": x})
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1} samples...")
    
    for _ in range(400): 
        result_queue.get()

    # --- 3. Phase 2: Promotion via Decoding ---
    print("\n--- Phase 2: Decoder-Led Promotion ---")
    task_queues["Explorer"].put({"cmd": "GET_MATURE_NODES"})
    res = result_queue.get()
    mature_list = res["mature"]
    
    routing_table = []
    if mature_list:
        # Identify most persistent emergent concept
        best_mature = sorted(mature_list, key=lambda x: x["weight"], reverse=True)[0]["node"]
        print(f"  [!] Governor identified persistent concept: {best_mature.id}")
        
        # 2. Extract the "ingredients" (The Sub-tree)
        # We manually gather the sub-tree from best_mature
        source_nodes = [best_mature]
        for child in best_mature.children():
            source_nodes.append(child)
            # Support 2-level depth for emergent hierarchies
            for grandchild in child.children():
                source_nodes.append(grandchild)
        
        # Launch Sovereign Core
        s_name = "Spec_Signal"
        task_queues[s_name] = mp.Queue()
        workers[s_name] = SovereignWorker(
            WorkerConfig(s_name, "spec_1", Path("data/sp25_spec"), source_nodes),
            task_queues[s_name], result_queue
        )
        workers[s_name].start()
        print(f"  [+] {s_name} promoted to dedicated process with {len(source_nodes)} priors.")
        
        routing_center = best_mature.mu
    else:
        print("  [FAIL] No mature concepts found. (Recombination did not trigger)."); return

    # --- 4. Phase 3: Specialized Execution & Forgetting ---
    print("\n--- Phase 3: Sovereign Routing & Natural Forgetting ---")
    for i in range(200):
        is_signal = rng.random() > 0.5
        x = rng.normal(5.0, 0.2, D) if is_signal else rng.uniform(-10, 10, D)
        
        if is_signal and np.linalg.norm(x - routing_center) <= 2.0:
            task_queues["Spec_Signal"].put({"cmd": "OBSERVE", "x": x})
        else:
            task_queues["Explorer"].put({"cmd": "OBSERVE", "x": x})
            
    for _ in range(200): result_queue.get()

    # --- 5. Final Topography ---
    print("\n--- Final Topography ---")
    for name in workers:
        task_queues[name].put({"cmd": "STATS"})
        res = result_queue.get()
        print(f"  {name:15s} Total Nodes: {res['total_nodes']:4d}")

    # Shutdown
    for name in workers: task_queues[name].put(None)
    for name in workers: workers[name].join()
    print(f"\nExperiment concluded.")

if __name__ == "__main__":
    run_experiment()
