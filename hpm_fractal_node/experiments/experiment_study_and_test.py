"""
SP29: Sovereign Study-and-Test Experiment.

Workflow:
1. Build Models (30x30).
2. Spawn Persistent Workers (one set for the entire run).
3. Phase 1 (Study): Process 10 tasks. Specialists create new nodes.
4. Phase 2 (Test): Process 10 new tasks. Track reuse of Study Nodes.
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

# Reuse components from experiment_thinking_arc_solver.py
from hpm_fractal_node.experiments.experiment_thinking_arc_solver import (
    WorkerConfig, SovereignARCWorker, reconstruct_grid
)

SEED = 42

def run_experiment():
    print("SP29: Sovereign Study-and-Test Experiment\n")
    mp.set_start_method("spawn", force=True)

    # 1. Build Models
    print("Building Baseline World Models (30x30)...")
    tasks = load_sovereign_tasks()
    math_base, math_priors = build_math_world_model(TieredForest, Path("data/study_math"), 600)
    spatial_forest, spatial_registry = build_prior_forest(30, 30)
    spatial_priors = set(spatial_registry.keys())
    print("World Models Built.\n")

    # 2. Setup Persistent Workers
    # We use 'study_' prefix for any nodes created in this experiment
    configs = [
        WorkerConfig("Spatial_Spec", "s_study", Path("data/study_s"), "OBSERVER", common_d=900, source_nodes=list(spatial_forest.active_nodes()), source_prior_ids=spatial_priors),
        WorkerConfig("Symbolic_Spec", "m_study", Path("data/study_m"), "OBSERVER", common_d=109, source_nodes=list(math_base.active_nodes()), source_prior_ids=math_priors),
        WorkerConfig("Spatial_Decoder", "d_study", Path("data/study_d"), "DECODER", common_d=900, sigma_threshold=0.1)
    ]
    
    queues = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    workers = {c.name: SovereignARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
    for w in workers.values(): w.start()

    # Heartbeat
    for name, q in queues.items(): 
        q.put({"cmd": "STATS"})
        res_queues[name].get()
    print("  [PASS] Cluster Communication Heartbeat.")

    # 3. Phase 1: Study Phase (First 10 tasks)
    print("\n--- PHASE 1: STUDY PHASE (Tasks 1-10) ---")
    study_set = tasks[:10]
    for task in study_set:
        print(f"  Studying Task {task['id']}...")
        # Broadcast all train examples to trigger learning/stabilisation
        for ex in task["train"]:
            queues["Spatial_Spec"].put({"cmd": "OBSERVE", "x": ex["vec"]})
            queues["Symbolic_Spec"].put({"cmd": "OBSERVE", "x": ex["vec"]})
            res_queues["Spatial_Spec"].get()
            res_queues["Symbolic_Spec"].get()

    # 4. Phase 2: Test Phase (Tasks 11-20)
    print("\n--- PHASE 2: TEST PHASE (Tasks 11-20) ---")
    test_set = tasks[10:20]
    solved = 0
    transfer_events = 0
    total_obs = 0

    for task_idx, task in enumerate(test_set):
        print(f"\n  Testing Task {task_idx+11} ({task['id']}):")
        
        # --- PHASE 1: INDUCTION (Collect Top-K with Transfer Tracking) ---
        train_examples_winners = []
        historical_deltas = []
        
        for ex in task["train"]:
            v = ex["vec"]
            historical_deltas.append(v[S_SLICE])
            total_obs += 1
            
            queues["Spatial_Spec"].put({"cmd": "OBSERVE", "x": v})
            queues["Symbolic_Spec"].put({"cmd": "OBSERVE", "x": v})
            
            ex_winners = []
            # Spatial
            r_s = res_queues["Spatial_Spec"].get()
            if r_s.get("competent"):
                for w in r_s["winners"]:
                    ex_winners.append(w["id"])
                    # Track if this winner is a 'leaf_' node (learned)
                    if "leaf_" in w["id"]:
                        transfer_events += 1
            # Symbolic
            r_m = res_queues["Symbolic_Spec"].get()
            if r_m.get("competent"):
                for w in r_m["winners"]:
                    ex_winners.append(w["id"])
                    if "leaf_" in w["id"]:
                        transfer_events += 1
            
            train_examples_winners.append(ex_winners)

        # Rule = Intersection
        if not train_examples_winners: continue
        shared_rules = set(train_examples_winners[0])
        for winners in train_examples_winners[1:]: shared_rules &= set(winners)
        
        hypotheses = list(shared_rules)
        print(f"    Generated {len(hypotheses)} Rule Hypotheses.")

        # --- PHASE 2: THINKING (Verification) ---
        valid_hypothesis = None
        for hyp_id in hypotheses:
            sim_ex = task["train"][0]
            hyp_obj = spatial_registry.get(hyp_id) or math_base.get(hyp_id)
            if not hyp_obj: continue # Skip if it was a dynamically learned leaf (for now)
            
            goal_sim = HFN(mu=sim_ex["vec"][S_SLICE], sigma=np.ones(900)*5.0, id="goal_sim")
            goal_sim.add_edge(goal_sim, hyp_obj, "MUST_SATISFY")
            
            queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal_sim})
            dec_res = res_queues["Spatial_Decoder"].get()["result"]
            
            if isinstance(dec_res, list) and dec_res:
                sim_delta = dec_res[0].mu
                sim_output = reconstruct_grid(sim_ex["input"], sim_delta)
                if np.array_equal(sim_output, sim_ex["output"]):
                    valid_hypothesis = hyp_id
                    break

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
                    print("    [SUCCESS] Puzzle Solved!")
                    solved += 1
                else:
                    print("    [FAIL] Test output mismatch.")
        else:
            print("    [FAIL] No valid hypotheses found.")

    print(f"\n--- SP29 Meta-Transfer Report ---")
    print(f"  Study Tasks processed: 10")
    print(f"  Test Tasks attempted:  10")
    print(f"  Tasks Solved:          {solved}/10")
    print(f"  Transfer Events:       {transfer_events} (Reuse of learned structural motifs)")

    for w in workers.values(): queues[w.config.name].put(None); w.join()

if __name__ == "__main__":
    run_experiment()
