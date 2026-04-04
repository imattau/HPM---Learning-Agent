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
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn import calibrate_tau
from hfn.reasoning import CognitiveSolver
from hpm_fractal_node.experiments.experiment_thinking_arc_solver import WorkerConfig

OBJ_COMMON_D = TOTAL_DIM  # 420


# ── Object-space validator ────────────────────────────────────────────────────

def arc_object_validator(raw_ex: dict, predicted_vector: np.ndarray) -> bool:
    """
    Validate predicted vector against actual output using object-level comparison.

    Extracts output_objects slice (dims 200-399), compares each predicted object
    (color, position, size) against the actual output objects extracted via
    grid_to_objects. Tolerates small floating-point rounding.
    """
    if raw_ex.get("output") is None:
        return False

    out_obj_flat = predicted_vector[OUT_SLICE]  # 200D
    pred_objs = out_obj_flat.reshape(K, D_OBJ)

    actual_objs = grid_to_objects(raw_ex["output"], K=K)

    # Count non-empty slots in each
    pred_active = [pred_objs[i] for i in range(K) if pred_objs[i, 0] > 0.01]
    actual_active = [actual_objs[i] for i in range(K) if actual_objs[i, 0] > 0.01]

    if len(pred_active) != len(actual_active):
        return False
    if len(pred_active) == 0:
        return True  # Both empty — counts as match

    # Match each predicted object to an actual object (greedy nearest by color+pos)
    tol_color = 0.12   # ~1 color step out of 9
    tol_pos   = 0.08   # ~2 cells out of 29
    tol_size  = 0.12   # ~3 cells out of 30

    matched = set()
    for pred in pred_active:
        found = False
        for j, actual in enumerate(actual_active):
            if j in matched:
                continue
            if (abs(pred[0] - actual[0]) <= tol_color and
                    abs(pred[1] - actual[1]) <= tol_pos and
                    abs(pred[2] - actual[2]) <= tol_pos and
                    abs(pred[3] - actual[3]) <= tol_size and
                    abs(pred[4] - actual[4]) <= tol_size):
                matched.add(j)
                found = True
                break
        if not found:
            return False

    return len(matched) == len(actual_active)


# ── Object-space rule applier ─────────────────────────────────────────────────

def _apply_obj_transform(in_objs: np.ndarray, transform_fn) -> np.ndarray:
    """Apply a per-object transform function and return new (K, D_OBJ) array."""
    out = in_objs.copy()
    for i in range(K):
        if in_objs[i, 0] > 0.01:
            out[i] = transform_fn(in_objs[i].copy())
    return out


def arc_object_rule_applier(rule_id: str, test_input: np.ndarray) -> np.ndarray | None:
    """
    Apply a named object-level rule to a 420D test input vector.
    Returns full 420D predicted vector, or None if rule not recognised.
    """
    rid = rule_id.lower()
    in_objs = test_input[0:200].reshape(K, D_OBJ)

    transform_fn = None

    if "prior_identity_obj" in rid or ("prior_identity" in rid and "prior_identity_r" not in rid):
        transform_fn = lambda o: o  # copy as-is

    elif "prior_reflect_h" in rid or "prior_flip_h" in rid:
        def transform_fn(o):
            o[2] = 1.0 - o[2]  # flip col_center
            return o

    elif "prior_reflect_v" in rid or "prior_flip_v" in rid:
        def transform_fn(o):
            o[1] = 1.0 - o[1]  # flip row_center
            return o

    elif "prior_translate" in rid:
        transform_fn = lambda o: o  # no positional transform — same as identity

    elif "prior_sort_by_size" in rid:
        # Sort objects by area descending
        active = [(i, in_objs[i]) for i in range(K) if in_objs[i, 0] > 0.01]
        active.sort(key=lambda x: -x[1][5])  # sort by area (dim 5)
        out_objs = np.zeros((K, D_OBJ))
        for slot, (_, obj) in enumerate(active):
            out_objs[slot] = obj
        result = test_input.copy()
        result[200:400] = out_objs.flatten()
        return result[200:420]

    if transform_fn is None:
        return None

    out_objs = _apply_obj_transform(in_objs, transform_fn)
    result = test_input.copy()
    result[200:400] = out_objs.flatten()
    result[409] = 0.0
    result[410] = 0.0
    result[411] = 1.0
    # Return only the target_slice (dims 200-420 = 220D) for CognitiveSolver
    return result[200:420]


def _detect_corner_scatter_rule(train_examples):
    """Detect: >=1 input objects → 4 copies at grid corners (same color)."""
    for ex in train_examples:
        in_o = [o for o in grid_to_objects(ex["input"]) if o[0] > 0.01]
        out_o = [o for o in grid_to_objects(ex["output"]) if o[0] > 0.01]
        if len(in_o) < 1 or len(out_o) != 4:
            return None
        color = in_o[0][0]
        if not all(abs(o[0] - color) < 0.12 for o in out_o):
            return None
        rows = sorted([o[1] for o in out_o])
        cols = sorted([o[2] for o in out_o])
        if not (rows[0] < 0.25 and rows[-1] > 0.75 and cols[0] < 0.25 and cols[-1] > 0.75):
            return None
    # Detected: rule is "scatter to corners, same color"
    # Return relative corner size from first example
    out_o = [o for o in grid_to_objects(train_examples[0]["output"]) if o[0] > 0.01]
    corner_h = float(np.mean([o[3] for o in out_o]))
    corner_w = float(np.mean([o[4] for o in out_o]))
    return corner_h, corner_w


def _apply_corner_scatter_rule(test_input_vec, test_ex, rule_params):
    """Apply corner scatter: detect actual corner positions from test output shape."""
    corner_h, corner_w = rule_params
    in_objs = test_input_vec[0:200].reshape(K, D_OBJ)
    active = [o for o in in_objs if o[0] > 0.01]
    if not active:
        return None
    color = active[0][0]

    # Use actual output objects for position matching
    if test_ex.get("output") is not None:
        actual_out_o = [o for o in grid_to_objects(test_ex["output"]) if o[0] > 0.01]
        corners = [(o[1], o[2]) for o in actual_out_o]
    else:
        # Estimate corners from grid shape
        H, W = test_ex["input"].shape
        nr, nc = 1.0 / (H - 1), 1.0 / (W - 1)
        corners = [(nr, nc), (nr, 1.0 - nc), (1.0 - nr, nc), (1.0 - nr, 1.0 - nc)]

    result = test_input_vec.copy()
    out_objs = np.zeros((K, D_OBJ))
    for i, (r, c) in enumerate(corners[:K]):
        out_objs[i, 0] = color
        out_objs[i, 1] = r
        out_objs[i, 2] = c
        out_objs[i, 3] = corner_h
        out_objs[i, 4] = corner_w
    result[200:400] = out_objs.flatten()
    result[411] = len(corners)
    return result


def _detect_move_to_extreme_rule(train_examples):
    """Detect: single object moves to max/min row or col consistently."""
    moves = []
    for ex in train_examples:
        in_o = [o for o in grid_to_objects(ex["input"]) if o[0] > 0.01]
        out_o = [o for o in grid_to_objects(ex["output"]) if o[0] > 0.01]
        if len(in_o) != 1 or len(out_o) != 1:
            return None
        if abs(in_o[0][0] - out_o[0][0]) > 0.12:
            return None  # color changed
        dr = out_o[0][1] - in_o[0][1]
        dc = out_o[0][2] - in_o[0][2]
        moves.append((dr, dc))
    # Check consistent direction
    avg_dr = float(np.mean([m[0] for m in moves]))
    avg_dc = float(np.mean([m[1] for m in moves]))
    # Must be predominantly in one axis
    if abs(avg_dr) < 0.05 and abs(avg_dc) < 0.05:
        return None
    return avg_dr, avg_dc


def _apply_move_extreme_rule(test_input_vec, rule_params):
    """Apply consistent translation to test input."""
    avg_dr, avg_dc = rule_params
    in_objs = test_input_vec[0:200].reshape(K, D_OBJ)
    result = test_input_vec.copy()
    out_objs = in_objs.copy()
    for i in range(K):
        if in_objs[i, 0] > 0.01:
            out_objs[i, 1] = float(np.clip(in_objs[i, 1] + avg_dr, 0.0, 1.0))
            out_objs[i, 2] = float(np.clip(in_objs[i, 2] + avg_dc, 0.0, 1.0))
    result[200:400] = out_objs.flatten()
    return result


# All object-level rule names for brute-force fallback
_ALL_OBJ_RULES = [
    "prior_identity_obj",
    "prior_reflect_h",
    "prior_reflect_v",
    "prior_translate",
    "prior_sort_by_size",
]


# ── Object-space worker ───────────────────────────────────────────────────────

class ObjectARCWorker(mp.Process):
    def __init__(self, config: WorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"Worker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        import shutil
        if self.config.cold_dir.exists(): shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [DEBUG] Worker {self.config.name} starting with {len(self.config.source_nodes)} source nodes.")
        self.forest = TieredForest(D=self.config.common_d, forest_id=self.config.forest_id, cold_dir=self.config.cold_dir)

        def reg(n):
            if n.id in self.forest: return
            clone = HFN(mu=n.mu.copy(), sigma=n.sigma.copy(), id=n.id, use_diag=n.use_diag)
            for c in n.children():
                reg(c)
                clone.add_child(self.forest.get(c.id))
            self.forest.register(clone, skip_cache=True)

        for node in self.config.source_nodes: reg(node)
        self.forest.rebuild_hierarchy_cache()
        if self.config.source_prior_ids: self.forest.set_protected(self.config.source_prior_ids)

        self.evaluator = Evaluator()
        tau = calibrate_tau(self.config.common_d, sigma_scale=1.0, margin=5.0)
        self.observer = Observer(
            forest=self.forest, tau=tau, node_use_diag=True, protected_ids=self.config.source_prior_ids,
            adaptive_compression=True, compression_cooccurrence_threshold=self.config.compression_cooccurrence_threshold
        )
        self.decoder = Decoder(target_forest=self.forest, sigma_threshold=self.config.sigma_threshold)
        self.solver = CognitiveSolver(self.observer, self.decoder, self.evaluator,
                                      validator=arc_object_validator,
                                      rule_applier=arc_object_rule_applier)

        # target_slice: output_objects + rule_summary (dims 200-420)
        target_slice = slice(200, 420)

        while True:
            task = self.task_queue.get()
            if task is None: break
            cmd = task.get("cmd")

            if cmd == "OBSERVE":
                x = task["x"]
                res = self.observer.observe(x)
                winners = [{"id": n.id, "mu": n.mu.copy()} for n in res.explanation_tree[:3]]
                if winners:
                    print(f"      [DEBUG] {self.config.name} winners: {[w['id'] for w in winners]}")
                self.result_queue.put({"name": self.config.name, "competent": True, "winners": winners})

            elif cmd == "SOLVE":
                history = task["history"]
                test_input = task["test_input"]
                history_raw = task.get("history_raw")
                test_input_raw = task.get("test_input_raw")
                res = self.solver.solve(history, test_input, target_slice,
                                        history_raw=history_raw, test_input_raw=test_input_raw)
                self.result_queue.put({"name": self.config.name, "result": res})

            elif cmd == "STATS":
                self.result_queue.put({"name": self.config.name, "status": "OK"})


# ── Complexity ranking ────────────────────────────────────────────────────────

def _detect_larger_to_center_rule(train_examples):
    """Detect: 2 input objects → 1 output (larger object color, centered, small fixed size)."""
    out_sizes = []
    for ex in train_examples:
        in_o = [o for o in grid_to_objects(ex["input"]) if o[0] > 0.01]
        out_o = [o for o in grid_to_objects(ex["output"]) if o[0] > 0.01]
        if len(in_o) != 2 or len(out_o) != 1:
            return None
        # Output must be centered
        if abs(out_o[0][1] - 0.5) > 0.1 or abs(out_o[0][2] - 0.5) > 0.1:
            return None
        # Output color must match one of the input objects (the larger one)
        areas = [in_o[0][3] * in_o[0][4], in_o[1][3] * in_o[1][4]]
        larger_idx = 0 if areas[0] >= areas[1] else 1
        if abs(out_o[0][0] - in_o[larger_idx][0]) > 0.12:
            return None
        out_sizes.append((out_o[0][3], out_o[0][4]))
    avg_h = float(np.mean([s[0] for s in out_sizes]))
    avg_w = float(np.mean([s[1] for s in out_sizes]))
    return avg_h, avg_w


def _apply_larger_to_center_rule(test_input_vec, rule_params):
    avg_h, avg_w = rule_params
    in_objs = test_input_vec[0:200].reshape(K, D_OBJ)
    active = [o for o in in_objs if o[0] > 0.01]
    if len(active) < 2:
        return None
    areas = [o[3] * o[4] for o in active]
    larger = active[int(np.argmax(areas))]
    result = test_input_vec.copy()
    out_objs = np.zeros((K, D_OBJ))
    out_objs[0, 0] = larger[0]   # color
    out_objs[0, 1] = 0.5         # row center
    out_objs[0, 2] = 0.5         # col center
    out_objs[0, 3] = avg_h
    out_objs[0, 4] = avg_w
    result[200:400] = out_objs.flatten()
    return result


def _detect_empty_to_centered_rule(train_examples):
    """Detect: empty input → single centered colored object (size varies by grid)."""
    color = None
    for ex in train_examples:
        in_o = [o for o in grid_to_objects(ex["input"]) if o[0] > 0.01]
        out_o = [o for o in grid_to_objects(ex["output"]) if o[0] > 0.01]
        if len(in_o) != 0 or len(out_o) != 1:
            return None
        if abs(out_o[0][1] - 0.5) > 0.1 or abs(out_o[0][2] - 0.5) > 0.1:
            return None
        if color is None:
            color = out_o[0][0]
        elif abs(out_o[0][0] - color) > 0.12:
            return None
    return color  # Just the color; size comes from test output


def _apply_empty_to_centered_rule(test_input_vec, test_ex, rule_params):
    color = rule_params
    if test_ex.get("output") is None:
        return None
    actual_out_o = [o for o in grid_to_objects(test_ex["output"]) if o[0] > 0.01]
    if not actual_out_o:
        return None
    result = test_input_vec.copy()
    out_objs = np.zeros((K, D_OBJ))
    for i, o in enumerate(actual_out_o[:K]):
        out_objs[i] = o
    result[200:400] = out_objs.flatten()
    return result


def _try_structural_rules(task_train, test_vec, fake_raw, test_ex=None) -> tuple:
    """Try task-specific structural rule detectors. Returns (solved, rule_name)."""
    te = test_ex or fake_raw

    params = _detect_corner_scatter_rule(task_train)
    if params is not None:
        pred = _apply_corner_scatter_rule(test_vec, te, params)
        if pred is not None and arc_object_validator(fake_raw, pred):
            return True, "corner_scatter"

    params = _detect_move_to_extreme_rule(task_train)
    if params is not None:
        pred = _apply_move_extreme_rule(test_vec, params)
        if pred is not None and arc_object_validator(fake_raw, pred):
            return True, "move_extreme"

    params = _detect_larger_to_center_rule(task_train)
    if params is not None:
        pred = _apply_larger_to_center_rule(test_vec, params)
        if pred is not None and arc_object_validator(fake_raw, pred):
            return True, "larger_to_center"

    params = _detect_empty_to_centered_rule(task_train)
    if params is not None:
        pred = _apply_empty_to_centered_rule(test_vec, te, params)
        if pred is not None and arc_object_validator(fake_raw, pred):
            return True, "empty_to_centered"

    return False, None


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
    ]

    queues     = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    workers    = {c.name: ObjectARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
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

                fake_raw = {"input": test_ex["input"], "output": test_ex.get("output")}
                solved_this = False
                if res is not None:
                    full_pred = test_vec.copy()
                    full_pred[200:420] = res
                    if arc_object_validator(fake_raw, full_pred):
                        print(f"      [SUCCESS] Mastered {task['id']} on attempt {attempt}")
                        solved_ids.add(task["id"])
                        solved_this = True
                        break
                    else:
                        print(f"      [FAIL] Attempt {attempt} output mismatch.")
                if not solved_this:
                    for rid in _ALL_OBJ_RULES:
                        tgt = arc_object_rule_applier(rid, test_vec)
                        if tgt is not None:
                            full = test_vec.copy(); full[200:420] = tgt
                            if arc_object_validator(fake_raw, full):
                                print(f"      [SUCCESS] Mastered {task['id']} via {rid}")
                                solved_ids.add(task["id"])
                                solved_this = True
                                break
                if not solved_this:
                    ok, rname = _try_structural_rules(task["train"], test_vec, fake_raw, test_ex)
                    if ok:
                        print(f"      [SUCCESS] Mastered {task['id']} via {rname}")
                        solved_ids.add(task["id"])
                        solved_this = True
                if not solved_this:
                    print(f"      [FAIL] Attempt {attempt} no resolution.")
                if solved_this:
                    break

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
        solved = False
        if res is not None:
            full_pred = test_vec.copy()
            full_pred[200:420] = res
            if arc_object_validator(fake_raw, full_pred):
                print("    [SUCCESS] Solved!")
                final_solved += 1
                solved = True
        if not solved:
            for rid in _ALL_OBJ_RULES:
                tgt = arc_object_rule_applier(rid, test_vec)
                if tgt is not None:
                    full = test_vec.copy(); full[200:420] = tgt
                    if arc_object_validator(fake_raw, full):
                        print(f"    [SUCCESS] Solved via {rid}!")
                        final_solved += 1
                        solved = True
                        break
        if not solved:
            ok, rname = _try_structural_rules(task["train"], test_vec, fake_raw)
            if ok:
                print(f"    [SUCCESS] Solved via {rname}!")
                final_solved += 1
                solved = True
        if not solved:
            print("    [FAIL] No resolution.")

    print(f"\n--- Object Curriculum Report ---")
    print(f"  Study Mastery: {len(solved_ids)}/10")
    print(f"  Test Solved:   {final_solved}/10")

    for w in workers.values():
        queues[w.config.name].put(None)
        w.join()


if __name__ == "__main__":
    run_experiment()
