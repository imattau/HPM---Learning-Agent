# Implementation Plan: Dynamic Specialist Promotion (SP25)

## 1. Goal
Demonstrate "Emergent Sovereignty" by allowing a Generalist process to locally discover high-level parent nodes via standard HPM Recombination dynamics, which are then autonomously promoted by the Governor to dedicated worker processes.

## 2. Architecture: "The Maturation Loop"
The system follows a uniform HPM maturation lifecycle:
1.  **Maturation (Explorer)**: The Generalist worker (`Explorer`) runs with `adaptive_compression=True`. It naturally creates parent HFNs when it detects recurring co-occurrences in the "Signal" cluster.
2.  **Detection (Governor)**: The Governor periodically scans the Explorer for any node that has (a) children and (b) high predictive weight. This represents a "Mature Concept."
3.  **Promotion (Governor)**: The Governor "declares sovereignty" for that node. It spawns a new process and seeds it with that node and its sub-tree.
4.  **Specialization (Routing)**: The Governor redirects matching signals to the new Specialist. The Explorer's redundant copy of the concept naturally decays and is garbage-collected.

## 3. The "Signal in the Noise" Experiment
- **Signal**: 2D Gaussian cluster.
- **Noise**: Uniform 2D scatter.
- **Trigger**: Explorer must independently discover a `compressed(...)` node for the signal before the Governor will promote it.

## 4. Full Implementation Code

```python
\"\"\"
SP25: Dynamic Specialist Promotion (Uniform Architecture).
Tests Emergent Sovereignty by promoting locally-recombined HFN parents 
into dedicated specialist processes.
\"\"\"
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import time
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer, calibrate_tau

# --- Constants ---
D = 2
SEED = 42

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    degree: float
    tau_sigma: float
    source_nodes: list[HFN] = field(default_factory=list)

class DynamicWorker(mp.Process):
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
        
        for node in self.config.source_nodes:
            self.forest.register(node)
            
        tau = calibrate_tau(D, sigma_scale=self.config.tau_sigma, margin=5.0)
        
        # Enable adaptive compression so it can locally discover parent nodes
        self.observer = Observer(
            forest=self.forest, tau=tau,
            adaptive_compression=True,
            compression_cooccurrence_threshold=5,
            persistence_guided_absorption=True, lacunarity_guided_creation=True,
            lacunarity_creation_radius=0.5, node_use_diag=True,
            residual_surprise_threshold=0.5,
            node_prefix=self.config.name + "_"
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
                # Find nodes with children and high weight
                mature = []
                for n in self.forest.active_nodes():
                    if not n.is_leaf():
                        weight = self.observer._weights.get(n.id, 0.0)
                        if weight > 0.7:
                            # We pass the full sub-tree data
                            subtree = [n] + n.children()
                            mature.append({
                                "id": n.id, "mu": n.mu.copy(), "sigma": n.sigma.copy(),
                                "use_diag": n.use_diag, "weight": weight,
                                "subtree_nodes": subtree
                            })
                self.result_queue.put({"name": self.config.name, "mature": mature})

def run_experiment():
    print("SP25: Dynamic Specialist Promotion (Emergent Sovereignty)\\n")
    mp.set_start_method("spawn", force=True)

    # --- 1. Initial State: Single Generalist ---
    task_queues = {"Explorer": mp.Queue()}
    result_queue = mp.Queue()
    workers = {"Explorer": DynamicWorker(
        WorkerConfig("Explorer", "exp_0", Path("data/sp25_exp"), 0.0, 2.0),
        task_queues["Explorer"], result_queue
    )}
    workers["Explorer"].start()
    print("  [+] Explorer process started (Generalist).")

    # --- 2. Curriculum: Forced Maturation (400 points) ---
    print("\\n--- Phase 1: Local Concept Maturation ---")
    rng = np.random.default_rng(SEED)
    for i in range(400):
        # Heavy signal bias to force rapid co-occurrence detection
        x = rng.normal(5.0, 0.2, D) if rng.random() > 0.3 else rng.uniform(-10, 10, D)
        task_queues["Explorer"].put({"cmd": "OBSERVE", "x": x})
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1} samples...")
    for _ in range(400): result_queue.get()

    # --- 3. Promotion: Governor Declaration ---
    print("\\n--- Phase 2: Governor Promotion Scan ---")
    task_queues["Explorer"].put({"cmd": "GET_MATURE_NODES"})
    mature_nodes = result_queue.get()["mature"]
    
    routing_table = []
    if mature_nodes:
        # Sort by weight and pick the most successful emergent concept
        best = sorted(mature_nodes, key=lambda x: x["weight"], reverse=True)[0]
        print(f"  [!] Mature concept detected: {best['id']} (Weight: {best['weight']:.2f})")
        
        # Spawn Specialist
        s_name = "Spec_Emergent"
        task_queues[s_name] = mp.Queue()
        workers[s_name] = DynamicWorker(
            WorkerConfig(s_name, "spec_1", Path("data/sp25_spec"), 0.5, 0.5, best["subtree_nodes"]),
            task_queues[s_name], result_queue
        )
        workers[s_name].start()
        print(f"  [+] {s_name} promoted to Sovereign core.")
        
        # Dynamic Routing Table
        routing_table.append((s_name, best["mu"], 2.0))
    else:
        print("  [FAIL] No mature concepts found. Recombination failed to trigger."); return

    # --- 4. Specialization: Distributed Signal ---
    print("\\n--- Phase 3: Sovereign Routing & Natural Forgetting ---")
    for _ in range(200):
        is_signal = rng.random() > 0.5
        x = rng.normal(5.0, 0.2, D) if is_signal else rng.uniform(-10, 10, D)
        
        # Governor routes signal to specialist if it matches the promoted concept mu
        routed = False
        for spec_name, center, radius in routing_table:
            if np.linalg.norm(x - center) <= radius:
                task_queues[spec_name].put({"cmd": "OBSERVE", "x": x})
                routed = True; break
        if not routed:
            task_queues["Explorer"].put({"cmd": "OBSERVE", "x": x})
            
    for _ in range(200): result_queue.get()

    # --- 5. Final Topography ---
    print("\\n--- Final Topography ---")
    for name in workers:
        task_queues[name].put({"cmd": "STATS"})
        res = result_queue.get()
        print(f"  {name:15s} Total Nodes: {res['total_nodes']:4d}")

    for name in workers: task_queues[name].put(None)
    for name in workers: workers[name].join()

if __name__ == "__main__":
    run_experiment()
```

## 5. Review against Specification
- **Agnostic Recombination**: *Pass*. The creation of the `compressed(...)` parent node now happens entirely inside the `Explorer`'s `Observer` loop using the standard `Recombination` operator.
- **Governor as Resource Manager**: *Pass*. The Governor no longer "builds" the parent. It only *identifies* it via `GET_MATURE_NODES` and moves it to a new core.
- **Dynamic Spawning**: *Pass*. Specialist process is spawned at runtime when maturation criteria are met.
- **Uniformity**: *Pass*. All dynamics (Maturation, Evaluation, Forgetting) use standard HFN/HPM primitives.
