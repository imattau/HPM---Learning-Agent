"""
SP28: Thinking ARC Solver Experiment (Iterative & Negative Anchoring).

Integrates:
- Multi-process Decentralized Observation (SP26)
- Hypothesis Testing Loop (Iterative Refinement)
- 30x30 Sovereign Manifold (950D)
- Negative Anchoring: Records failed hypotheses as HFNs to measure distance to solution.
"""
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import time
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from itertools import product

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN, Edge
from hfn.forest import Forest
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hfn.decoder import Decoder, ResolutionRequest
from hfn import calibrate_tau, Evaluator
from hpm_fractal_node.arc.arc_sovereign_loader import (
    load_sovereign_tasks, COMMON_D, S_SLICE, M_SLICE, C_SLICE, S_DIM
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
        if self.config.cold_dir.exists(): shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(D=self.config.common_d, forest_id=self.config.forest_id, cold_dir=self.config.cold_dir)
        
        def reg(n):
            if n.id in self.forest: return
            clone = HFN(mu=n.mu.copy(), sigma=n.sigma.copy(), id=n.id, use_diag=n.use_diag)
            for c in n.children():
                reg(c)
                clone.add_child(self.forest.get(c.id))
            self.forest.register(clone)
            
        for node in self.config.source_nodes: reg(node)
        if self.config.source_prior_ids: self.forest.set_protected(self.config.source_prior_ids)

        self.evaluator = Evaluator()
        tau = calibrate_tau(self.config.common_d, sigma_scale=1.0, margin=5.0)
        self.observer = Observer(forest=self.forest, tau=tau, node_use_diag=True, protected_ids=self.config.source_prior_ids)
        self.decoder = Decoder(target_forest=self.forest, sigma_threshold=self.config.sigma_threshold)

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE":
                x_full = task["x"]
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
                    winners = [{"id": n.id, "mu": n.mu.copy()} for n in res.explanation_tree[:3]]
                    self.result_queue.put({"name": self.config.name, "competent": True, "winners": winners})
            
            elif cmd == "DECODE":
                goal = task["goal"]
                res = self.decoder.decode(goal)
                self.result_queue.put({"name": self.config.name, "result": res})
            
            elif cmd == "LEARN_FROM_BUFFER":
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
            
            elif cmd == "STATS":
                self.result_queue.put({"name": self.config.name, "status": "OK"})

def reconstruct_grid(input_grid: np.ndarray, delta_900d: np.ndarray) -> np.ndarray:
    delta_2d = (delta_900d * 9.0).reshape(30, 30)
    padded_in = np.zeros((30, 30))
    r, c = min(30, input_grid.shape[0]), min(30, input_grid.shape[1])
    padded_in[:r, :c] = input_grid[:r, :c]
    
    out_30x30 = np.clip(np.round(padded_in + delta_2d), 0, 9).astype(int)
    orig_r, orig_c = input_grid.shape
    return out_30x30[:orig_r, :orig_c]

def run_experiment():
    print("SP28: Thinking ARC Solver Experiment (Iterative & Negative Anchoring)\n")
    
    # 1. Build Models FIRST
    print("Building World Models (30x30)...")
    tasks = load_sovereign_tasks()
    math_base, math_priors = build_math_world_model(TieredForest, Path("data/think_math_sp28"), 600)
    spatial_forest, spatial_registry = build_prior_forest(30, 30)
    spatial_priors = set(spatial_registry.keys())
    print("World Models Built.\n")

    # 2. Start Multi-Process
    mp.set_start_method("spawn", force=True)

    configs = [
        WorkerConfig("Spatial_Spec", "s_think_sp28", Path("data/think_s_sp28"), "OBSERVER", common_d=900, competence_threshold=0.0, source_nodes=list(spatial_forest.active_nodes()), source_prior_ids=spatial_priors),
        WorkerConfig("Symbolic_Spec", "m_think_sp28", Path("data/think_m_sp28"), "OBSERVER", common_d=109, competence_threshold=0.0, source_nodes=list(math_base.active_nodes()), source_prior_ids=math_priors),
        WorkerConfig("Spatial_Decoder", "d_think_sp28", Path("data/think_d_sp28"), "DECODER", common_d=900, sigma_threshold=0.1)
    ]
    
    queues = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    workers = {c.name: SovereignARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
    for w in workers.values(): w.start()

    print("--- System Smoke Test ---")
    for name, q in queues.items(): 
        q.put({"cmd": "STATS"})
        res_queues[name].get()
    print("  [PASS] Cluster Communication Heartbeat.")

    # 3. Thinking Solver Loop
    solved = 0
    limit = 20
    failure_manifold = Forest(D=950, forest_id="failure_manifold")
    print(f"\n--- Processing {limit} Tasks with Iterative Refinement ---")
    
    for task_idx, task in enumerate(tasks[:limit]):
        print(f"\nTask {task_idx+1} ({task['id']}):")
        
        # --- PHASE 1: INDUCTION (Collect Top-K) ---
        train_examples_winners = []
        historical_deltas = []
        
        for ex in task["train"]:
            v = ex["vec"]
            historical_deltas.append(v[S_SLICE])
            queues["Spatial_Spec"].put({"cmd": "OBSERVE", "x": v})
            queues["Symbolic_Spec"].put({"cmd": "OBSERVE", "x": v})
            
            ex_winners = []
            r_s = res_queues["Spatial_Spec"].get()
            if r_s.get("competent"): ex_winners.extend(r_s["winners"])
            r_m = res_queues["Symbolic_Spec"].get()
            if r_m.get("competent"): ex_winners.extend(r_m["winners"])
            train_examples_winners.append([w["id"] for w in ex_winners])

        shared_rules = set(train_examples_winners[0]) if train_examples_winners else set()
        for winners in train_examples_winners[1:]: shared_rules &= set(winners)
        
        hypotheses = list(shared_rules)
        print(f"  Generated {len(hypotheses)} Rule Hypotheses.")

        # --- PHASE 2: THINKING (Internal Validation & Negative Anchoring) ---
        valid_hypothesis = None
        for hyp_id in hypotheses:
            print(f"    Testing Hypothesis: {hyp_id}...")
            sim_ex = task["train"][0]
            hyp_obj = spatial_registry.get(hyp_id) or math_base.get(hyp_id)
            if not hyp_obj: continue
            
            goal_sim = HFN(mu=sim_ex["vec"][S_SLICE], sigma=np.ones(900)*5.0, id="goal_sim")
            goal_sim.add_edge(goal_sim, hyp_obj, "MUST_SATISFY")
            
            queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal_sim})
            dec_res = res_queues["Spatial_Decoder"].get()["result"]
            
            if isinstance(dec_res, list) and dec_res:
                sim_delta = dec_res[0].mu
                sim_output = reconstruct_grid(sim_ex["input"], sim_delta)
                if np.array_equal(sim_output, sim_ex["output"]):
                    print(f"      [VALIDATED] Rule works.")
                    valid_hypothesis = hyp_id
                    break
                else:
                    error_dist = np.linalg.norm(sim_output.flatten() - sim_ex["output"].flatten())
                    print(f"      [REJECTED] Error Distance: {error_dist:.2f}")
                    # Negative Anchoring
                    neg_node = HFN(mu=sim_delta, sigma=np.zeros(900), id=f"failed_{hyp_id}")
                    failure_manifold.register(neg_node)
            else:
                print(f"      [REJECTED] Decoding stall.")

        # --- PHASE 3: EXECUTION ---
        if valid_hypothesis:
            test_ex = task["test"][0]
            hyp_obj = spatial_registry.get(valid_hypothesis) or math_base.get(valid_hypothesis)
            goal_test = HFN(mu=np.zeros(900), sigma=np.ones(900)*10.0, id="goal_test")
            goal_test.add_edge(goal_test, hyp_obj, "MUST_SATISFY")
            queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal_test})
            dec_res = res_queues["Spatial_Decoder"].get()["result"]
            
            if isinstance(dec_res, list) and dec_res:
                final_delta = dec_res[0].mu
                final_output = reconstruct_grid(test_ex["input"], final_delta)
                if test_ex["output"] is not None and np.array_equal(final_output, test_ex["output"]):
                    print("  [SUCCESS] Puzzle Solved via Thinking!")
                    solved += 1
                else:
                    print("  [FAIL] Test output mismatch.")
        else:
            print("  [FAIL] No valid hypotheses found.")

    print(f"\n--- Final Results ---")
    print(f"  Tasks Attempted: {limit}")
    print(f"  Tasks Solved:    {solved}")
    print(f"  Negative Anchors Recorded: {len(failure_manifold)}")

    for w in workers.values(): queues[w.config.name].put(None); w.join()

if __name__ == "__main__":
    run_experiment()
