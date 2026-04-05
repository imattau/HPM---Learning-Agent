"""
SP29: Sovereign Study-and-Test Experiment (Mastery-Driven).

Workflow:
1. Build Models (30x30).
2. Spawn Persistent Workers.
3. Phase 1 (Study): Process 10 tasks. 
   - REQUIREMENT: Must solve 6/10 tasks before taking the test.
   - Loop back to failed tasks iteratively.
   - Keep learned nodes (positive and negative) across attempts.
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
    load_sovereign_tasks, COMMON_D, S_SLICE, M_SLICE, C_SLICE, S_DIM
)
from hpm_fractal_node.arc.arc_prior_forest import build_prior_forest
from hpm_fractal_node.math.math_world_model import build_math_world_model

# Reuse components from experiment_thinking_arc_solver import WorkerConfig, SovereignARCWorker, reconstruct_grid, predict_shape
from hpm_fractal_node.experiments.experiment_thinking_arc_solver import (
    WorkerConfig, SovereignARCWorker, reconstruct_grid, predict_shape
)

SEED = 42

def solve_task(task, queues, res_queues, spatial_registry, math_base):
    """Attempt to solve a single ARC task using Stereo Thinking loop."""
    # --- PHASE 1: INDUCTION ---
    train_examples_winners = []
    historical_deltas = []
    
    for i, ex in enumerate(task["train"]):
        v = ex["vec"]
        historical_deltas.append(v[S_SLICE])
        queues["Spatial_Spec"].put({"cmd": "OBSERVE", "x": v})
        queues["Symbolic_Spec"].put({"cmd": "OBSERVE", "x": v})
        
        ex_winners = []
        r_s = res_queues["Spatial_Spec"].get()
        if r_s.get("competent"):
            for w in r_s["winners"]: ex_winners.append(("Spatial_Spec", w["id"]))
        r_m = res_queues["Symbolic_Spec"].get()
        if r_m.get("competent"):
            for w in r_m["winners"]: ex_winners.append(("Symbolic_Spec", w["id"]))
        train_examples_winners.append(ex_winners)

    if not train_examples_winners: return False
    
    # Define roots to ignore
    ROOTS = {
        "prior_grid", "prior_extent", "prior_density", "prior_structure", 
        "prior_spatial_organisation", "prior_colour", "prior_grid_transform", 
        "prior_transformation"
    }

    # Extract candidate lists per example
    example_spatial_candidates = []
    example_symbolic_candidates = []
    for w_list in train_examples_winners:
        spatial = [nid for src, nid in w_list if src == "Spatial_Spec" and nid not in ROOTS]
        symbolic = [nid for src, nid in w_list if src == "Symbolic_Spec" and nid not in ROOTS]
        example_spatial_candidates.append(spatial)
        example_symbolic_candidates.append(symbolic)

    # Intersection of Spatial candidates
    if not example_spatial_candidates or not example_spatial_candidates[0]: return False
    shared_spatial = set(example_spatial_candidates[0])
    for c_list in example_spatial_candidates[1:]: 
        if c_list: shared_spatial &= set(c_list)
    
    # Intersection of Symbolic candidates
    shared_symbolic = set(example_symbolic_candidates[0]) if example_symbolic_candidates and example_symbolic_candidates[0] else set()
    for c_list in example_symbolic_candidates[1:]:
        if c_list: shared_symbolic &= set(c_list)
    
    if not shared_spatial:
        flat_spatial = [nid for sublist in example_spatial_candidates for nid in sublist]
        counts = Counter(flat_spatial)
        shared_spatial = {nid for nid, count in counts.most_common(3)}
    
    hypotheses = list(product(list(shared_spatial), list(shared_symbolic) if shared_symbolic else [None]))
    
    # --- PHASE 2: THINKING ---
    valid_hypothesis = None
    for hyp_spatial_id, hyp_symbolic_id in hypotheses:
        sim_ex = task["train"][0]
        
        # 1. Resolve Spatial Rule
        hyp_spatial_obj = spatial_registry.get(hyp_spatial_id)
        if not hyp_spatial_obj:
            queues["Spatial_Spec"].put({"cmd": "GET_NODE", "id": hyp_spatial_id})
            hyp_spatial_obj = res_queues["Spatial_Spec"].get()["node"]
            if hyp_spatial_obj:
                queues["Spatial_Decoder"].put({"cmd": "REGISTER_NODE", "node": hyp_spatial_obj})
                res_queues["Spatial_Decoder"].get()
        
        if not hyp_spatial_obj: continue
        
        # Simulation
        goal_sim = HFN(mu=sim_ex["vec"][S_SLICE], sigma=np.ones(S_DIM)*5.0, id="goal_sim", use_diag=True)
        goal_sim.add_edge(goal_sim, hyp_spatial_obj, "MUST_SATISFY")
        
        queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal_sim})
        dec_res_dict = res_queues["Spatial_Decoder"].get()
        dec_res = dec_res_dict["result"]
        
        # Handle Curiosity
        if isinstance(dec_res, ResolutionRequest):
            queues["Spatial_Decoder"].put({
                "cmd": "LEARN_FROM_BUFFER", 
                "buffer": historical_deltas, 
                "mu": dec_res.missing_mu, 
                "edges": dec_res.required_edges
            })
            if res_queues["Spatial_Decoder"].get()["success"]:
                queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal_sim})
                dec_res = res_queues["Spatial_Decoder"].get()["result"]

        if isinstance(dec_res, list) and dec_res:
            sim_delta = dec_res[0].mu
            sim_delta_rounded = np.round(sim_delta * 9.0) / 9.0
            # USE TARGET SHAPE OVERRIDE for simulation
            sim_output = reconstruct_grid(sim_ex["input"], sim_delta_rounded, rule_node=None, target_shape=sim_ex["output"].shape)
            if np.array_equal(sim_output, sim_ex["output"]):
                valid_hypothesis = (hyp_spatial_id, hyp_symbolic_id)
                break
    
    # --- PHASE 3: FINAL SOLVE ---
    if valid_hypothesis:
        hyp_spatial_id, hyp_symbolic_id = valid_hypothesis
        test_ex = task["test"][0]
        
        # Resolve object
        hyp_spatial_obj = spatial_registry.get(hyp_spatial_id)
        if not hyp_spatial_obj:
            queues["Spatial_Spec"].put({"cmd": "GET_NODE", "id": hyp_spatial_id})
            hyp_spatial_obj = res_queues["Spatial_Spec"].get()["node"]
            
        target_shape = predict_shape(task)

        goal_test = HFN(mu=np.zeros(S_DIM), sigma=np.ones(S_DIM)*10.0, id="goal_test", use_diag=True)
        goal_test.add_edge(goal_test, hyp_spatial_obj, "MUST_SATISFY")
        queues["Spatial_Decoder"].put({"cmd": "DECODE", "goal": goal_test})
        dec_res = res_queues["Spatial_Decoder"].get()["result"]
        
        if isinstance(dec_res, list) and dec_res:
            final_delta = dec_res[0].mu
            final_delta_rounded = np.round(final_delta * 9.0) / 9.0
            final_output = reconstruct_grid(test_ex["input"], final_delta_rounded, rule_node=None, target_shape=target_shape)
            if test_ex["output"] is not None and np.array_equal(final_output, test_ex["output"]):
                return True
    return False

def run_experiment():
    print("SP29: Sovereign Study-and-Test Experiment (Mastery-Driven)\n")
    mp.set_start_method("spawn", force=True)

    print("Building Baseline World Models (30x30)...")
    tasks = load_sovereign_tasks()
    from hpm_fractal_node.arc.arc_functional_priors import build_functional_spatial_priors, build_functional_symbolic_priors
    
    spatial_nodes = build_functional_spatial_priors(30, 30)
    spatial_priors = {n.id for n in spatial_nodes}
    spatial_registry = {n.id: n for n in spatial_nodes}
    
    attr_nodes = build_functional_symbolic_priors()
    attr_priors = {n.id for n in attr_nodes}
    
    # For compatibility
    math_base = Forest(D=109)

    configs = [
        WorkerConfig("Spatial_Spec", "s_study", Path("data/study_s"), "OBSERVER", common_d=900, source_nodes=spatial_nodes, source_prior_ids=spatial_priors),
        WorkerConfig("Symbolic_Spec", "m_study", Path("data/study_m"), "OBSERVER", common_d=30, competence_threshold=0.0, source_nodes=attr_nodes, source_prior_ids=attr_priors),
        WorkerConfig("Spatial_Decoder", "d_study", Path("data/study_d"), "DECODER", common_d=900, sigma_threshold=0.01, source_nodes=spatial_nodes)
    ]
    
    queues = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    workers = {c.name: SovereignARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
    for w in workers.values(): w.start()

    for name, q in queues.items(): 
        q.put({"cmd": "STATS"})
        res_queues[name].get()
    print("  [PASS] Cluster Communication Heartbeat.")

    print("\n--- PHASE 1: STUDY PHASE (Target: 6/10 Correct) ---")
    study_set = tasks[:10]
    solved_ids = set()
    iteration = 1
    
    while len(solved_ids) < 6:
        print(f"\n  Study Iteration {iteration} (Solved: {len(solved_ids)}/10)...")
        new_solves = 0
        for task in study_set:
            if task["id"] in solved_ids: continue
            if solve_task(task, queues, res_queues, spatial_registry, math_base):
                print(f"      [SUCCESS] Task Solved and Mastered!")
                solved_ids.add(task["id"])
                new_solves += 1
        if new_solves == 0 and len(solved_ids) < 6:
            if iteration >= 5: break
        iteration += 1

    print("\n--- PHASE 2: TEST PHASE (Tasks 11-20) ---")
    test_set = tasks[10:20]
    final_solved = 0
    for task_idx, task in enumerate(test_set):
        print(f"\n  Testing Task {task_idx+11} ({task['id']}):")
        if solve_task(task, queues, res_queues, spatial_registry, math_base):
            print("    [SUCCESS] Test Task Solved!")
            final_solved += 1
        else:
            print("    [FAIL] Test Task Mismatch.")

    print(f"\n--- SP29 Mastery Report ---")
    print(f"  Study Iterations:      {iteration-1}")
    print(f"  Study Mastery:         {len(solved_ids)}/10")
    print(f"  Test Solve Rate:       {final_solved}/10")

    for w in workers.values(): queues[w.config.name].put(None); w.join()

if __name__ == "__main__":
    run_experiment()
