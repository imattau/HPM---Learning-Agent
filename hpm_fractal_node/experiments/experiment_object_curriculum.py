"""
Object-Level Curriculum Experiment — 420D Object Space.

Mirrors the SP30 structured curriculum but uses 420D object-level encoding
instead of 1850D pixel-space sovereign manifold. Each grid is decomposed into
connected components; the HFN pipeline reasons about object-level transformations.

Feature space: 420D
  [0-199]   Input objects  (K=10, 10×20D each)
  [200-399] Output objects (K=10, 10×20D each)
  [400-419] Rule summary   (20D)

WorkerConfig uses common_d=420.
"""
from __future__ import annotations

import multiprocessing as mp
import numpy as np
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn import Evaluator
from hpm_fractal_node.arc.arc_sovereign_loader import load_sovereign_tasks
from hpm_fractal_node.arc.arc_object_encoder import (
    task_pair_to_vec,
    test_input_to_vec,
    grid_to_objects,
    TOTAL_DIM,
    OUT_SLICE,
    D_OBJ,
    K,
)
from hpm_fractal_node.arc.arc_object_priors import build_object_level_priors
from hpm_fractal_node.experiments.experiment_thinking_arc_solver import (
    WorkerConfig,
    SovereignARCWorker,
)

OBJ_COMMON_D = TOTAL_DIM  # 420


# ── Object-space validator ────────────────────────────────────────────────────

def _objects_to_grid(out_objs: np.ndarray, ref_shape: tuple) -> np.ndarray:
    """
    Reconstruct a grid from K×20D output object descriptors.

    Places each object as a single pixel at its encoded (row, col) center,
    using its encoded color. This is a coarse reconstruction suitable for
    exact-match checks on simple ARC puzzles.
    """
    H, W = max(ref_shape[0], 1), max(ref_shape[1], 1)
    grid = np.zeros((H, W), dtype=int)

    for slot in range(K):
        obj = out_objs[slot]
        color_norm = obj[0]
        if color_norm <= 0:
            continue  # Empty slot

        color = int(round(color_norm * 9.0))
        if color <= 0:
            continue

        row = int(round(obj[1] * max(H - 1, 1)))
        col = int(round(obj[2] * max(W - 1, 1)))
        row = int(np.clip(row, 0, H - 1))
        col = int(np.clip(col, 0, W - 1))
        grid[row, col] = color

    return grid


def arc_object_validator(raw_ex: dict, predicted_vector: np.ndarray) -> bool:
    """
    Validate predicted 420D vector against actual output grid.

    Extracts output_objects slice (dims 200-399), reconstructs a grid by
    placing each object at its encoded position/color, and compares with
    raw_ex["output"].
    """
    if raw_ex.get("output") is None:
        return False

    out_obj_flat = predicted_vector[OUT_SLICE]  # 200D
    out_objs = out_obj_flat.reshape(K, D_OBJ)
    ref_shape = raw_ex["output"].shape
    pred_grid = _objects_to_grid(out_objs, ref_shape)
    return bool(np.array_equal(pred_grid, raw_ex["output"]))


# ── Object-space rule applier ─────────────────────────────────────────────────

def arc_object_rule_applier(rule_id: str, test_input: np.ndarray) -> np.ndarray | None:
    """
    Apply a named object-level rule to a 420D test input vector.

    Supported rules:
      prior_identity_obj — copy input_objects to output_objects
      prior_recolor       — copy position/shape dims, leave color as-is (returns None,
                            let solver use decoder)
      others              — return None (let CognitiveSolver use decoder/fallback)

    Args:
        rule_id: str, prior node ID
        test_input: np.ndarray(420,) — test input vector (output slice = zeros)

    Returns:
        np.ndarray(420,) with output objects filled in, or None.
    """
    rid = rule_id.lower()
    in_objs_flat = test_input[0:200]   # dims 0-199

    if "prior_identity_obj" in rid:
        # Output objects = input objects (no transformation)
        result = test_input.copy()
        result[OUT_SLICE] = in_objs_flat.copy()
        # Rule summary: identity
        result[400] = 0.0
        result[404] = 1.0   # bucket 4 = delta-color 0
        result[409] = 0.0   # no position change
        result[410] = 0.0   # no size change
        result[411] = 1.0   # count ratio = 1
        return result

    if "prior_recolor" in rid:
        # Copy position+shape dims from input, but leave color uncertain → return None
        # (let solver use the decoder to estimate target color)
        return None

    # All other rules: let CognitiveSolver handle via decoder
    return None


# ── Complexity ranking ────────────────────────────────────────────────────────

def calculate_object_complexity(task: dict) -> float:
    """
    Rank tasks by object-space complexity using Evaluator.task_complexity
    on 420D object vectors derived from train examples.
    """
    evaluator = Evaluator()
    vecs = [
        task_pair_to_vec(ex["input"], ex["output"])
        for ex in task["train"]
    ]
    return evaluator.task_complexity(vecs)


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment():
    print("Object-Level Curriculum Experiment — 420D Object Space\n")

    # Clear worker data directories
    for d in ["data/obj_s", "data/obj_d"]:
        if Path(d).exists():
            shutil.rmtree(Path(d))

    print("Loading tasks...")
    all_tasks = load_sovereign_tasks()
    print(f"Loaded {len(all_tasks)} tasks.\n")

    # Build object-level priors
    print("Building 420D object-level priors...")
    obj_prior_nodes = build_object_level_priors()
    obj_prior_ids = {n.id for n in obj_prior_nodes}
    print(f"  {len(obj_prior_nodes)} prior nodes built.\n")

    # Rank tasks by object-space complexity
    print(f"Ranking {len(all_tasks)} tasks by object-space complexity...")
    ranked = sorted(all_tasks, key=calculate_object_complexity)
    study_set = ranked[:10]
    test_set  = ranked[-10:]
    print(f"Study Set IDs: {[t['id'] for t in study_set]}")
    print(f"Test Set IDs:  {[t['id'] for t in test_set]}\n")

    # Worker configs — single 420D worker (Observer + solver)
    mp.set_start_method("spawn", force=True)

    configs = [
        WorkerConfig(
            "Object_Spec",
            "obj_spec",
            Path("data/obj_s"),
            "OBSERVER",
            common_d=OBJ_COMMON_D,
            competence_threshold=0.0,
            source_nodes=obj_prior_nodes,
            source_prior_ids=obj_prior_ids,
            sigma_threshold=2.0,
            compression_cooccurrence_threshold=10000,
        ),
        WorkerConfig(
            "Object_Decoder",
            "obj_dec",
            Path("data/obj_d"),
            "DECODER",
            common_d=OBJ_COMMON_D,
            sigma_threshold=0.1,
            source_nodes=obj_prior_nodes,
        ),
    ]

    queues     = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    workers    = {c.name: SovereignARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
    for w in workers.values():
        w.start()

    # Heartbeat
    for name, q in queues.items():
        q.put({"cmd": "STATS"})
        res_queues[name].get()
    print("  [PASS] Worker heartbeat OK.\n")

    # ── PHASE 1: STUDY ───────────────────────────────────────────────────────
    print("--- PHASE 1: OBJECT-SPACE STUDY (Target: 6/10 Correct) ---")
    solved_ids = set()

    for iteration in range(1, 11):
        if len(solved_ids) >= 6:
            break
        print(f"\n  Iteration {iteration} (Mastery: {len(solved_ids)}/10)...")

        for task in study_set:
            if task["id"] in solved_ids:
                continue

            print(f"    Studying Task {task['id']}...")

            # Build 420D training vectors
            history_vecs = [
                task_pair_to_vec(ex["input"], ex["output"])
                for ex in task["train"]
            ]
            history_raw = task["train"]

            test_ex = task["test"][0]
            test_vec = test_input_to_vec(test_ex["input"])

            for attempt in range(1, 6):
                # Observe training examples
                for vec in history_vecs:
                    queues["Object_Spec"].put({"cmd": "OBSERVE", "x": vec})
                    res_queues["Object_Spec"].get()  # drain

                # Solve
                queues["Object_Spec"].put({
                    "cmd": "SOLVE",
                    "history": history_vecs,
                    "test_input": test_vec,
                    "history_raw": history_raw,
                    "test_input_raw": test_ex,
                })
                res = res_queues["Object_Spec"].get()["result"]

                if res is not None:
                    # Use object-space validator
                    fake_raw = {"input": test_ex["input"], "output": test_ex.get("output")}
                    if arc_object_validator(fake_raw, res):
                        print(f"      [SUCCESS] Mastered {task['id']} on attempt {attempt}")
                        solved_ids.add(task["id"])
                        break
                    else:
                        print(f"      [FAIL] Attempt {attempt} output mismatch.")
                else:
                    # Fallback: try identity rule manually
                    identity_pred = arc_object_rule_applier("prior_identity_obj", test_vec)
                    if identity_pred is not None:
                        fake_raw = {"input": test_ex["input"], "output": test_ex.get("output")}
                        if arc_object_validator(fake_raw, identity_pred):
                            print(f"      [SUCCESS] Mastered {task['id']} via identity fallback")
                            solved_ids.add(task["id"])
                            break
                    print(f"      [FAIL] Attempt {attempt} no resolution.")

    # ── PHASE 2: TRANSFER TEST ───────────────────────────────────────────────
    print("\n--- PHASE 2: TRANSFER TEST (Complex Tasks) ---")
    final_solved = 0

    for task_idx, task in enumerate(test_set):
        print(f"\n  Testing Task {task_idx+1} ({task['id']}):")

        history_vecs = [
            task_pair_to_vec(ex["input"], ex["output"])
            for ex in task["train"]
        ]
        history_raw = task["train"]
        test_ex = task["test"][0]
        test_vec = test_input_to_vec(test_ex["input"])

        queues["Object_Spec"].put({
            "cmd": "SOLVE",
            "history": history_vecs,
            "test_input": test_vec,
            "history_raw": history_raw,
            "test_input_raw": test_ex,
        })
        res = res_queues["Object_Spec"].get()["result"]

        fake_raw = {"input": test_ex["input"], "output": test_ex.get("output")}
        if res is not None and arc_object_validator(fake_raw, res):
            print("    [SUCCESS] Solved!")
            final_solved += 1
        else:
            # Fallback: identity rule
            identity_pred = arc_object_rule_applier("prior_identity_obj", test_vec)
            if identity_pred is not None and arc_object_validator(fake_raw, identity_pred):
                print("    [SUCCESS] Solved via identity fallback!")
                final_solved += 1
            else:
                print("    [FAIL] No resolution.")

    print(f"\n--- Object Curriculum Report ---")
    print(f"  Study Mastery: {len(solved_ids)}/10")
    print(f"  Test Solved:   {final_solved}/10")

    for w in workers.values():
        queues[w.config.name].put(None)
        w.join()


if __name__ == "__main__":
    run_experiment()
