"""
SP30: Structured Curriculum Experiment (Framework-Native).

Uses framework-native Evaluator.task_complexity for ranking
and CognitiveSolver (via solve_task) for reasoning.
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

from hfn.hfn import HFN, Edge
from hfn.forest import Forest
from hfn.tiered_forest import TieredForest
from hfn import Evaluator
from hpm_fractal_node.arc.arc_sovereign_loader import load_sovereign_tasks, S_SLICE
from hpm_fractal_node.arc.arc_prior_forest import build_prior_forest
from hpm_fractal_node.math.math_world_model import build_math_world_model
from hpm_fractal_node.experiments.experiment_thinking_arc_solver import (
    WorkerConfig, SovereignARCWorker, reconstruct_grid
)

def calculate_complexity(task: dict) -> float:
    evaluator = Evaluator()
    obs = [ex["vec"] for ex in task["train"]]
    return evaluator.task_complexity(obs)

def run_experiment():
    print("SP30: Structured Curriculum Experiment (Framework-Native)\n")
    print("Building World Models...")
    for d in ["data/curr_math", "data/curr_s", "data/curr_m", "data/curr_d", "data/curr_e"]:
        if Path(d).exists(): shutil.rmtree(Path(d))
    all_tasks = load_sovereign_tasks()
    print("World Models Built.\n")
    print(f"Ranking {len(all_tasks)} tasks by framework-native complexity...")
    ranked = sorted(all_tasks, key=calculate_complexity)
    study_set = ranked[:10]
    test_set = ranked[-10:]
    print(f"Study Set IDs: {[t['id'] for t in study_set]}")
    print(f"Test Set IDs:  {[t['id'] for t in test_set]}\n")
    
    from hpm_fractal_node.arc.arc_functional_priors import build_functional_spatial_priors, build_functional_symbolic_priors
    spatial_nodes = build_functional_spatial_priors(D=1800)
    spatial_priors = {n.id for n in spatial_nodes}
    
    attr_nodes = build_functional_symbolic_priors()
    attr_priors = {n.id for n in attr_nodes}

    # Explorer Priors (1850D)
    explorer_nodes = build_functional_spatial_priors(D=1850)
    
    # Pad Attribute nodes to 1850D for Explorer
    for node_30d in attr_nodes:
        mu_1850 = np.zeros(1850)
        mu_1850[1800:1830] = node_30d.mu
        # VAGUE PADDING: 10.0 variance everywhere except the attribute slice
        sig_1850 = np.ones(1850) * 10.0 
        sig_1850[1800:1830] = np.diag(node_30d.sigma) if not node_30d.use_diag else node_30d.sigma
        explorer_nodes.append(HFN(mu=mu_1850, sigma=sig_1850, id=f"e_{node_30d.id}", use_diag=True))
    
    explorer_priors = {n.id for n in explorer_nodes}
    
    mp.set_start_method("spawn", force=True)
    configs = [
        WorkerConfig("Spatial_Spec", "s_curr", Path("data/curr_s"), "OBSERVER", common_d=1800, competence_threshold=0.0, 
                     source_nodes=spatial_nodes, source_prior_ids=spatial_priors),
        WorkerConfig("Symbolic_Spec", "m_curr", Path("data/curr_m"), "OBSERVER", common_d=30, competence_threshold=0.0,
                     source_nodes=attr_nodes, source_prior_ids=attr_priors),
        WorkerConfig("Explorer", "e_curr", Path("data/curr_e"), "OBSERVER", common_d=1850, competence_threshold=0.0,
                     source_nodes=explorer_nodes, source_prior_ids=explorer_priors, sigma_threshold=2.0),
        WorkerConfig("Spatial_Decoder", "d_curr", Path("data/curr_d"), "DECODER", common_d=1800, sigma_threshold=0.01,
                     source_nodes=spatial_nodes)
    ]
    queues = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    workers = {c.name: SovereignARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
    for w in workers.values(): w.start()
    for name, q in queues.items(): 
        q.put({"cmd": "STATS"}); res_queues[name].get()
    # 3. Phase 1: Bootstrapped Study (Target 6/10)
    print("--- PHASE 1: BOOTSTRAPPED STUDY (Target: 6/10 Correct) ---")
    solved_ids = set()
    for iteration in range(1, 11): 
        if len(solved_ids) >= 6: break
        print(f"\n  Iteration {iteration} (Mastery: {len(solved_ids)}/10)...")
        for task in study_set:
            if task["id"] in solved_ids: continue

            # STUDY LOOP: Try multiple attempts at the same task
            print(f"    Studying Task {task['id']}...")
            for attempt in range(1, 6): # More attempts

                history = [ex["vec"] for ex in task["train"]]
                history_raw = task["train"]
                test_ex = task["test"][0]

                queues["Explorer"].put({
                    "cmd": "SOLVE", 
                    "history": history, 
                    "test_input": test_ex["vec"],
                    "history_raw": history_raw,
                    "test_input_raw": test_ex
                })
                res = res_queues["Explorer"].get()["result"]

                if res is not None:
                    out_grid = reconstruct_grid(test_ex["input"], res)
                    if test_ex["output"] is not None and np.array_equal(out_grid, test_ex["output"]):
                        print(f"      [SUCCESS] Mastered {task['id']} on attempt {attempt}")
                        solved_ids.add(task["id"])
                        break
                    else:
                        print(f"      [FAIL] Attempt {attempt} output mismatch.")
                else:
                    print(f"      [FAIL] Attempt {attempt} no resolution.")

                
    print("\n--- PHASE 2: TRANSFER TEST (Complex Tasks) ---")
    final_solved = 0
    for task_idx, task in enumerate(test_set):
        print(f"\n  Testing Task {task_idx+1} ({task['id']}):")
        history = [ex["vec"] for ex in task["train"]]
        history_raw = task["train"]
        test_ex = task["test"][0]
        queues["Explorer"].put({
            "cmd": "SOLVE", 
            "history": history, 
            "test_input": test_ex["vec"],
            "history_raw": history_raw,
            "test_input_raw": test_ex
        })
        res = res_queues["Explorer"].get()["result"]
        if res is not None:
            out_grid = reconstruct_grid(test_ex["input"], res)
            if test_ex["output"] is not None and np.array_equal(out_grid, test_ex["output"]):
                print("    [SUCCESS] Solved!")
                final_solved += 1
            else:
                print("    [FAIL] Mismatch.")
        else:
            print("    [FAIL] No resolution.")
            
    print(f"\n--- SP30 Curriculum Report ---")
    print(f"  Study Mastery: {len(solved_ids)}/10")
    print(f"  Test Solved:   {final_solved}/10")
    for w in workers.values(): queues[w.config.name].put(None); w.join()

if __name__ == "__main__":
    run_experiment()
