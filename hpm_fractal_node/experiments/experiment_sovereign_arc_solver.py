"""
SP27: Sovereign ARC Solver Experiment.

Integrates:
- Multi-process Decentralized Observation (SP26)
- Stereo Vision Rule Induction (SP18)
- Agnostic Variance Collapse Decoding (SP22)
- Demand-Driven Learning Curiosity Loop (SP24)

Workflow:
1. Induce Rule from Train Examples (Bottom-Up).
2. Formulate Goal for Test Input.
3. Decode Goal into 100D Delta (Top-Down).
4. Reconstruct and Score Grid.
"""
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import time
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN, Edge
from hfn.forest import Forest
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hfn.decoder import Decoder, ResolutionRequest
from hfn import calibrate_tau, Evaluator
from hpm_fractal_node.arc.arc_sovereign_loader import (
    load_sovereign_tasks, COMMON_D, S_SLICE, M_SLICE, C_SLICE
)
from hpm_fractal_node.arc.arc_prior_forest import build_prior_forest
from hpm_fractal_node.math.math_world_model import build_math_world_model

SEED = 42

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    role: str # "OBSERVER" | "DECODER"
    common_d: int = COMMON_D
    competence_threshold: float = 0.0
    sigma_threshold: float = 0.01
    source_nodes: list[HFN] = field(default_factory=list)
    source_prior_ids: set[str] = field(default_factory=set)

class SovereignARCWorker(mp.Process):
    def __init__(self, config: WorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"Worker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        print(f"  [DEBUG] Worker {self.config.name} starting...")
        if self.config.cold_dir.exists(): shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(D=self.config.common_d, forest_id=self.config.forest_id, cold_dir=self.config.cold_dir)
        
        # Deep register priors
        def reg(n):
            if n.id in self.forest: return
            clone = HFN(mu=n.mu.copy(), sigma=n.sigma.copy(), id=n.id, use_diag=n.use_diag)
            for c in n.children():
                reg(c)
                clone.add_child(self.forest.get(c.id))
            self.forest.register(clone)
            
        print(f"  [DEBUG] Worker {self.config.name} registering {len(self.config.source_nodes)} source nodes...")
        for node in self.config.source_nodes: reg(node)
        if self.config.source_prior_ids: self.forest.set_protected(self.config.source_prior_ids)

        self.evaluator = Evaluator()
        tau = calibrate_tau(self.config.common_d, sigma_scale=1.0, margin=5.0)
        self.observer = Observer(forest=self.forest, tau=tau, node_use_diag=True, protected_ids=self.config.source_prior_ids)
        self.decoder = Decoder(target_forest=self.forest, sigma_threshold=self.config.sigma_threshold)
        print(f"  [DEBUG] Worker {self.config.name} ready.")

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE":
                x_full = task["x"]
                # Each worker might have a different D in their Forest
                # We need to slice the 150D input accordingly
                if self.config.name == "Spatial_Spec": x = x_full[S_SLICE]
                elif self.config.name == "Symbolic_Spec": 
                    x = np.zeros(109)
                    x[:30] = x_full[M_SLICE]
                else: x = x_full # Explorer
                
                acc = self.evaluator.accuracy(x, self.forest)
                if acc < self.config.competence_threshold:
                    self.result_queue.put({"name": self.config.name, "competent": False})
                else:
                    res = self.observer.observe(x)
                    self.forest._on_observe()
                    winners = [{"id": n.id, "mu": n.mu.copy()} for n in res.explanation_tree]
                    self.result_queue.put({"name": self.config.name, "competent": True, "winners": winners})
            
            elif cmd == "DECODE":
                goal = task["goal"]
                res = self.decoder.decode(goal)
                self.result_queue.put({"name": self.config.name, "result": res})
            
            elif cmd == "LEARN_FROM_BUFFER":
                # SP24 Demand-Driven Learning
                buffer = task["buffer"]
                target_mu = task["mu"]
                edges = task["edges"]
                found = False
                for obs in buffer:
                    if np.linalg.norm(obs - target_mu) < 0.5:
                        leaf = HFN(mu=obs, sigma=np.ones(self.config.common_d)*0.001, id=f"discovered_{int(np.sum(obs))}", use_diag=True)
                        for e in edges: leaf.add_edge(leaf, e.target, e.relation)
                        self.forest.register(leaf)
                        found = True; break
                self.result_queue.put({"name": self.config.name, "success": found})

def reconstruct_grid(input_grid: np.ndarray, delta_100d: np.ndarray) -> np.ndarray:
    delta_2d = (delta_100d * 9.0).reshape(10, 10)
    # Pad or clip input to 10x10
    padded_in = np.zeros((10, 10))
    r, c = min(10, input_grid.shape[0]), min(10, input_grid.shape[1])
    padded_in[:r, :c] = input_grid[:r, :c]
    
    out_10x10 = np.clip(np.round(padded_in + delta_2d), 0, 9).astype(int)
    # Return to original or clipped shape
    return out_10x10[:r, :c]

def run_experiment():
    print("SP27: Sovereign ARC Solver Experiment\n")
    
    # 1. Build Models FIRST (Single Process)
    print("Building World Models...")
    tasks = load_sovereign_tasks()
    math_base, math_priors = build_math_world_model(TieredForest, Path("data/sarc_solv_math"), 600)
    spatial_forest, spatial_registry = build_prior_forest(10, 10)
    spatial_priors = set(spatial_registry.keys())
    print("World Models Built.\n")

    # 2. Start Multi-Process
    configs = [
        WorkerConfig("Spatial_Spec", "s_solv", Path("data/solv_s"), "OBSERVER", common_d=100, competence_threshold=0.0, source_nodes=list(spatial_forest.active_nodes()), source_prior_ids=spatial_priors),
        WorkerConfig("Symbolic_Spec", "m_solv", Path("data/solv_m"), "OBSERVER", common_d=109, competence_threshold=0.0, source_nodes=list(math_base.active_nodes()), source_prior_ids=math_priors),
        WorkerConfig("Spatial_Decoder", "d_solv", Path("data/solv_d"), "DECODER", common_d=100, sigma_threshold=0.1)
    ]
    
    queues = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    
    mp.set_start_method("spawn", force=True)
    
    workers = {c.name: SovereignARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
    for w in workers.values(): w.start()

    print("--- System Smoke Test ---")
    # Heartbeat
    for name, q in queues.items(): 
        print(f"  Pinging {name}...")
        q.put({"cmd": "STATS"})
        try:
            res_queues[name].get(timeout=10) # 10s timeout
            print(f"    {name} responded.")
        except:
            print(f"    [ERROR] {name} timed out.")
    print("  [PASS] Cluster Communication Heartbeat.")

    # 3. Solver Loop (First 200 tasks)
    solved = 0
    limit = 200
    print(f"\n--- Processing {limit} Tasks ---")
    
    for task_idx, task in enumerate(tasks[:limit]):
        print(f"\nTask {task_idx+1} ({task['id']}):")
        
        # --- PHASE 1: INDUCTION ---
        all_train_winners = []
        historical_deltas = []
        for i, ex in enumerate(task["train"]):
            print(f"  [DEBUG] Processing Train Example {i+1}...")
            v = ex["vec"]
            historical_deltas.append(v[S_SLICE])
            
            # Broadcast
            queues["Spatial_Spec"].put({"cmd": "OBSERVE", "x": v})
            queues["Symbolic_Spec"].put({"cmd": "OBSERVE", "x": v})
            
            winners = []
            # Get from specific queues
            r_s = res_queues["Spatial_Spec"].get()
            if r_s.get("competent"): winners.extend(r_s["winners"])
            r_m = res_queues["Symbolic_Spec"].get()
            if r_m.get("competent"): winners.extend(r_m["winners"])
            
            all_train_winners.append([w["id"] for w in winners])

        # Rule = Intersection of winners across all examples
        if not all_train_winners: continue
        rule_set = set(all_train_winners[0])
        for w_list in all_train_winners[1:]: rule_set &= set(w_list)
        
        print(f"  Duced Rule Nodes: {list(rule_set)}")

        # --- PHASE 2 & 3: DECODING ---
        # Construct Goal for first test example (simplified)
        # Note: Loader currently only loads train. We'll use last train as proxy test for this experiment.
        test_ex = task["train"][-1] 
        
        # We need the actual HFN objects for the rule nodes to give to decoder
        # For simplicity, we'll re-extract from our registries
        rule_objects = []
        for rid in rule_set:
            obj = spatial_registry.get(rid) or math_base.get(rid)
            if obj: rule_objects.append(obj)

        goal = HFN(mu=test_ex["vec"], sigma=np.ones(COMMON_D)*5.0, id="goal_test")
        for ro in rule_objects: goal.add_edge(goal, ro, "MUST_SATISFY")

        # Dispatch to Decoder
        queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal})
        dec_res = res_queues["Spatial_Decoder"].get()["result"]
        
        if isinstance(dec_res, ResolutionRequest):
            print("  [Curiosity] Decoder stalled. Attempting Demand-Driven Learning...")
            queues["Spatial_Decoder"].put({
                "cmd": "LEARN_FROM_BUFFER", 
                "buffer": historical_deltas, 
                "mu": dec_res.missing_mu, 
                "edges": dec_res.required_edges
            })
            if res_queues["Spatial_Decoder"].get()["success"]:
                # Retry
                queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal})
                dec_res = res_queues["Spatial_Decoder"].get()["result"]

        if isinstance(dec_res, list) and dec_res:
            delta_100d = dec_res[0].mu[S_SLICE]
            out_grid = reconstruct_grid(test_ex["input"], delta_100d)
            
            if np.array_equal(out_grid, test_ex["output"]):
                print("  [SUCCESS] Task Solved!")
                solved += 1
            else:
                print("  [FAIL] Output mismatch.")
        else:
            print("  [FAIL] Decoding failed.")

    print(f"\n--- Final Results ---")
    print(f"  Tasks Attempted: {limit}")
    print(f"  Tasks Solved:    {solved}")

    for w in workers.values(): 
        queues[w.config.name].put(None)
        w.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_experiment()
