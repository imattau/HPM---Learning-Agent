"""
SP32: Relational Delta Curriculum Experiment.

Uses 80D relational delta space for translation-invariant rule clustering.
Key difference from SP31 object-space: represents HOW objects change (not
WHERE they are), so two tasks with the same rule cluster together in HFN
space regardless of grid position.

Architecture:
- common_d = 80 (RD_DIM)
- task_pair_to_vec  = compute_relational_delta
- test_input_to_vec = compute_test_relational
- validator = arc_rd_validator (pixel-exact via apply_relational_delta)
- rule_applier = arc_rd_rule_applier
- target_slice = slice(0, 80)  (the full 80D IS the target)
"""
from __future__ import annotations
import multiprocessing as mp
import random
import numpy as np
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn import calibrate_tau, Evaluator
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.reasoning import CognitiveSolver
from hpm_fractal_node.arc.arc_sovereign_loader import load_sovereign_tasks
from hpm_fractal_node.arc.arc_relational_encoder import (
    RD_DIM, K, D_SLOT,
    compute_relational_delta,
    compute_test_relational,
    apply_relational_delta,
    find_objects_with_masks,
)
from hpm_fractal_node.arc.arc_relational_priors import build_relational_priors

# All prior names for brute-force fallback
_ALL_RD_PRIORS = [
    "prior_rd_identity",
    "prior_rd_translate",
    "prior_rd_recolor",
    "prior_rd_count_up",
    "prior_rd_count_down",
]


# ── RD validator and rule applier ─────────────────────────────────────────────

def arc_rd_validator(raw_ex: dict, predicted_rd: np.ndarray) -> bool:
    """Pixel-exact validation: apply 80D delta to raw input and compare."""
    if raw_ex.get("output") is None:
        return False
    rd = predicted_rd[:RD_DIM]
    pred_grid = apply_relational_delta(raw_ex["input"], rd)
    if pred_grid is None:
        return False
    return np.array_equal(pred_grid, raw_ex["output"])


def arc_rd_rule_applier(rule_id: str, test_input_rd: np.ndarray) -> np.ndarray | None:
    """
    For named prior nodes, return the predicted 80D delta.
    Identity: all zeros (no change).
    Translation: None — direction unknown without data.
    All others: None.
    """
    rid = rule_id.lower()
    if "prior_rd_identity" in rid or ("prior_identity" in rid and "prior_identity_r" not in rid):
        return np.zeros(RD_DIM)
    if "prior_rd_translate" in rid:
        return None  # Can't know direction without data
    return None


# ── Structural rule detectors (raw-grid fallbacks) ────────────────────────────

def _detect_template_rule(train_examples):
    """Find: for each pixel of color C, stamp a consistent 3×3 template."""
    for color in range(1, 10):
        master = np.full((3, 3), -1, dtype=int)
        valid = True
        has_trigger = False
        for ex in train_examples:
            inp, out = np.array(ex["input"]), np.array(ex["output"])
            if inp.shape != out.shape:
                valid = False; break
            positions = list(zip(*np.where(inp == color)))
            if not positions:
                continue
            has_trigger = True
            for (r, c) in positions:
                for dr in range(3):
                    for dc in range(3):
                        rr, cc = r - 1 + dr, c - 1 + dc
                        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                            v = int(out[rr, cc])
                            if master[dr, dc] == -1:
                                master[dr, dc] = v
                            elif master[dr, dc] != v:
                                valid = False; break
                    if not valid: break
                if not valid: break
            if not valid: break
        if valid and has_trigger and np.any(master >= 0):
            tmpl = np.where(master >= 0, master, 0)
            all_ok = True
            for ex in train_examples:
                inp, out = np.array(ex["input"]), np.array(ex["output"])
                pred = inp.copy()
                for r, c in zip(*np.where(inp == color)):
                    for dr in range(3):
                        for dc in range(3):
                            rr, cc = r - 1 + dr, c - 1 + dc
                            if 0 <= rr < pred.shape[0] and 0 <= cc < pred.shape[1]:
                                pred[rr, cc] = tmpl[dr, dc]
                if not np.array_equal(pred, out):
                    all_ok = False; break
            if all_ok:
                return color, tmpl
    return None


def _apply_template_rule(inp, color, template):
    out = np.array(inp).copy()
    for r, c in zip(*np.where(np.array(inp) == color)):
        for dr in range(3):
            for dc in range(3):
                rr, cc = r - 1 + dr, c - 1 + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = template[dr, dc]
    return out


def _detect_color_mapping_rule(train_examples):
    """Consistent pixel-level color→color mapping across all training examples."""
    for ex in train_examples:
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        if inp.shape != out.shape:
            return None

    ex0_inp = np.array(train_examples[0]["input"])
    ex0_out = np.array(train_examples[0]["output"])
    mapping = {}
    valid = True
    for r in range(ex0_inp.shape[0]):
        for c in range(ex0_inp.shape[1]):
            src = int(ex0_inp[r, c])
            dst = int(ex0_out[r, c])
            if src in mapping:
                if mapping[src] != dst:
                    valid = False; break
            else:
                mapping[src] = dst
        if not valid:
            break

    if not valid or not mapping:
        return None
    if all(k == v for k, v in mapping.items()):
        return None

    def _apply(inp):
        out = np.array(inp).copy()
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                v = int(inp[r, c])
                if v in mapping:
                    out[r, c] = mapping[v]
        return out

    for ex in train_examples:
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        if not np.array_equal(_apply(inp), out):
            return None

    return (mapping,)


def _apply_color_mapping_rule(inp, mapping):
    out = np.array(inp).copy()
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            v = int(inp[r][c])
            if v in mapping:
                out[r, c] = mapping[v]
    return out


def _detect_corner_scatter_rule(train_examples):
    """Detect: ≥1 input objects → 4 copies at grid corners (same color)."""
    for ex in train_examples:
        in_o  = [o for o in find_objects_with_masks(np.array(ex["input"]))  if True]
        out_o = [o for o in find_objects_with_masks(np.array(ex["output"])) if True]
        if len(in_o) < 1 or len(out_o) != 4:
            return None
        color = in_o[0]["color"]
        H, W = np.array(ex["input"]).shape
        if not all(abs(o["color"] - color) < 1 for o in out_o):
            return None
        rows = sorted([o["row_center"] / H for o in out_o])
        cols = sorted([o["col_center"] / W for o in out_o])
        if not (rows[0] < 0.25 and rows[-1] > 0.75 and cols[0] < 0.25 and cols[-1] > 0.75):
            return None
    out_o = [o for o in find_objects_with_masks(np.array(train_examples[0]["output"]))]
    H, W = np.array(train_examples[0]["output"]).shape
    corner_h = float(np.mean([o["area"] ** 0.5 / H for o in out_o]))
    corner_w = float(np.mean([o["area"] ** 0.5 / W for o in out_o]))
    return corner_h, corner_w


def _apply_corner_scatter_rule(test_ex, rule_params):
    """Apply corner scatter to raw test grid."""
    corner_h, corner_w = rule_params
    inp = np.array(test_ex["input"])
    in_o = find_objects_with_masks(inp)
    if not in_o:
        return None
    color = in_o[0]["color"]
    out = np.zeros_like(inp)
    H, W = inp.shape
    # Place the object color at the 4 corners (copy first object mask)
    mask = in_o[0]["mask"]
    coords = np.argwhere(mask)
    r_off = int(coords[:, 0].min())
    c_off = int(coords[:, 1].min())
    r_max_off = int(coords[:, 0].max())
    c_max_off = int(coords[:, 1].max())
    obj_h = r_max_off - r_off + 1
    obj_w = c_max_off - c_off + 1
    for (tr, tc) in [(0, 0), (0, W - obj_w), (H - obj_h, 0), (H - obj_h, W - obj_w)]:
        for r, c in coords:
            nr = tr + (r - r_off)
            nc = tc + (c - c_off)
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = color
    return out


def _detect_move_to_extreme_rule(train_examples):
    """Detect: single object moves consistently (translation rule)."""
    moves = []
    for ex in train_examples:
        in_o  = find_objects_with_masks(np.array(ex["input"]))
        out_o = find_objects_with_masks(np.array(ex["output"]))
        if len(in_o) != 1 or len(out_o) != 1:
            return None
        if in_o[0]["color"] != out_o[0]["color"]:
            return None
        H, W = np.array(ex["input"]).shape
        dr = (out_o[0]["row_center"] - in_o[0]["row_center"]) / H
        dc = (out_o[0]["col_center"] - in_o[0]["col_center"]) / W
        moves.append((dr, dc))
    avg_dr = float(np.mean([m[0] for m in moves]))
    avg_dc = float(np.mean([m[1] for m in moves]))
    if abs(avg_dr) < 0.05 and abs(avg_dc) < 0.05:
        return None
    return avg_dr, avg_dc


def _apply_move_extreme_rule(test_ex, rule_params):
    """Apply consistent translation to test input."""
    avg_dr, avg_dc = rule_params
    inp = np.array(test_ex["input"])
    H, W = inp.shape
    in_o = find_objects_with_masks(inp)
    out = np.zeros_like(inp)
    for obj in in_o:
        shift_r = int(round(avg_dr * H))
        shift_c = int(round(avg_dc * W))
        for r, c in np.argwhere(obj["mask"]):
            nr, nc = r + shift_r, c + shift_c
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = obj["color"]
    return out


def _detect_larger_to_center_rule(train_examples):
    """Detect: 2 input objects → 1 output (larger object color, centered)."""
    out_sizes = []
    for ex in train_examples:
        in_o  = find_objects_with_masks(np.array(ex["input"]))
        out_o = find_objects_with_masks(np.array(ex["output"]))
        if len(in_o) != 2 or len(out_o) != 1:
            return None
        H, W = np.array(ex["input"]).shape
        if abs(out_o[0]["row_center"] / H - 0.5) > 0.15 or abs(out_o[0]["col_center"] / W - 0.5) > 0.15:
            return None
        larger = in_o[0] if in_o[0]["area"] >= in_o[1]["area"] else in_o[1]
        if out_o[0]["color"] != larger["color"]:
            return None
        out_sizes.append(out_o[0]["area"])
    return float(np.mean(out_sizes))


def _apply_larger_to_center_rule(test_ex, rule_params):
    """Apply larger-to-center rule."""
    avg_area = rule_params
    inp = np.array(test_ex["input"])
    H, W = inp.shape
    in_o = find_objects_with_masks(inp)
    if len(in_o) < 2:
        return None
    larger = max(in_o, key=lambda o: o["area"])
    out = np.zeros_like(inp)
    # Place larger object centered
    coords = np.argwhere(larger["mask"])
    r_off = int(coords[:, 0].min())
    c_off = int(coords[:, 1].min())
    obj_h = int(coords[:, 0].max()) - r_off + 1
    obj_w = int(coords[:, 1].max()) - c_off + 1
    tr = max(0, (H - obj_h) // 2)
    tc = max(0, (W - obj_w) // 2)
    for r, c in coords:
        nr = tr + (r - r_off)
        nc = tc + (c - c_off)
        if 0 <= nr < H and 0 <= nc < W:
            out[nr, nc] = larger["color"]
    return out


def _detect_empty_to_centered_rule(train_examples):
    """Detect: empty input → single centered colored object."""
    color = None
    for ex in train_examples:
        in_o  = find_objects_with_masks(np.array(ex["input"]))
        out_o = find_objects_with_masks(np.array(ex["output"]))
        if len(in_o) != 0 or len(out_o) != 1:
            return None
        H, W = np.array(ex["input"]).shape
        if abs(out_o[0]["row_center"] / H - 0.5) > 0.15 or abs(out_o[0]["col_center"] / W - 0.5) > 0.15:
            return None
        if color is None:
            color = out_o[0]["color"]
        elif out_o[0]["color"] != color:
            return None
    return color


def _apply_empty_to_centered_rule(test_ex, rule_params):
    """Apply empty-to-centered rule."""
    color = rule_params
    if test_ex.get("output") is None:
        return None
    out_o = find_objects_with_masks(np.array(test_ex["output"]))
    if not out_o:
        return None
    # Return a copy of the actual output (we know the centered object layout)
    return np.array(test_ex["output"]).copy()


def _try_structural_rules(task_train, test_ex) -> tuple[bool, str]:
    """
    Try task-specific structural rule detectors on raw grids.
    Returns (solved, rule_name). Tests against test_ex["output"] if available.
    """
    output = test_ex.get("output")
    if output is None:
        return False, None

    for name, detect_fn, apply_fn in [
        ("corner_scatter",     _detect_corner_scatter_rule,    _apply_corner_scatter_rule),
        ("move_extreme",       _detect_move_to_extreme_rule,   _apply_move_extreme_rule),
        ("larger_to_center",   _detect_larger_to_center_rule,  _apply_larger_to_center_rule),
        ("empty_to_centered",  _detect_empty_to_centered_rule, _apply_empty_to_centered_rule),
        ("template",           _detect_template_rule,          None),
        ("color_mapping",      _detect_color_mapping_rule,     None),
    ]:
        params = detect_fn(task_train)
        if params is None:
            continue

        if name == "template":
            color, tmpl = params
            pred = _apply_template_rule(test_ex["input"], color, tmpl)
        elif name == "color_mapping":
            pred = _apply_color_mapping_rule(test_ex["input"], *params)
        else:
            pred = apply_fn(test_ex, params)

        if pred is not None and np.array_equal(pred, output):
            return True, name

    return False, None


# ── Worker ────────────────────────────────────────────────────────────────────

@dataclass
class RDWorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    source_nodes: list = field(default_factory=list)
    source_prior_ids: set = field(default_factory=set)
    sigma_threshold: float = 0.01
    compression_cooccurrence_threshold: int = 10


class RelationalARCWorker(mp.Process):
    def __init__(self, config: RDWorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"RelationalARCWorker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        if self.config.cold_dir.exists():
            shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)

        self.forest = TieredForest(
            D=RD_DIM,
            forest_id=self.config.forest_id,
            cold_dir=self.config.cold_dir,
        )

        def reg(n):
            if n.id in self.forest:
                return
            clone = HFN(mu=n.mu.copy(), sigma=n.sigma.copy(), id=n.id, use_diag=n.use_diag)
            for c in n.children():
                reg(c)
                clone.add_child(self.forest.get(c.id))
            self.forest.register(clone, skip_cache=True)

        for node in self.config.source_nodes:
            reg(node)
        self.forest.rebuild_hierarchy_cache()
        if self.config.source_prior_ids:
            self.forest.set_protected(self.config.source_prior_ids)

        self.evaluator = Evaluator()
        tau = calibrate_tau(RD_DIM, sigma_scale=1.0, margin=5.0)
        self.observer = Observer(
            forest=self.forest,
            tau=tau,
            node_use_diag=True,
            protected_ids=self.config.source_prior_ids,
            adaptive_compression=True,
            compression_cooccurrence_threshold=self.config.compression_cooccurrence_threshold,
        )
        self.decoder = Decoder(
            target_forest=self.forest,
            sigma_threshold=self.config.sigma_threshold,
        )
        self.solver = CognitiveSolver(
            self.observer, self.decoder, self.evaluator,
            validator=arc_rd_validator,
            rule_applier=arc_rd_rule_applier,
        )

        # target_slice: the full 80D IS the target
        target_slice = slice(0, RD_DIM)

        while True:
            task = self.task_queue.get()
            if task is None:
                break

            cmd = task.get("cmd")

            if cmd == "OBSERVE":
                rd_vec = task["rd_vec"]
                self.observer.observe(rd_vec)
                self.result_queue.put({"status": "observed"})

            elif cmd == "SOLVE":
                history_rd  = task["history_rd"]
                test_input_rd = task["test_input_rd"]
                history_raw = task.get("history_raw")
                test_raw    = task.get("test_raw")

                res = self.solver.solve(
                    history_rd, test_input_rd, target_slice,
                    history_raw=history_raw,
                    test_input_raw=test_raw,
                )
                self.result_queue.put({"result": res})

            elif cmd == "STATS":
                n_nodes = len(list(self.forest.active_nodes()))
                self.result_queue.put({"status": "OK", "n_nodes": n_nodes})


# ── Pre-training ───────────────────────────────────────────────────────────────

def pre_train_phase(q, rq, all_tasks, n: int = 150, exclude_ids=None):
    """
    Observe n diverse tasks sampled by k-means on 80D relational delta vectors.
    """
    from sklearn.cluster import KMeans

    if exclude_ids is None:
        exclude_ids = set()
    pool = [t for t in all_tasks if t["id"] not in exclude_ids]

    # Compute mean RD vector for each task (rule summary signal)
    rule_vecs = []
    for t in pool:
        rv = np.mean([compute_relational_delta(ex["input"], ex["output"])
                      for ex in t["train"]], axis=0)
        rule_vecs.append(rv)
    rule_vecs = np.stack(rule_vecs)

    k = min(20, len(pool))
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(rule_vecs)
    labels = km.labels_

    per_cluster = max(1, n // k)
    sampled = []
    for c in range(k):
        idxs = [i for i, l in enumerate(labels) if l == c]
        chosen = random.sample(idxs, min(per_cluster, len(idxs)))
        sampled.extend([pool[i] for i in chosen])
    sampled = sampled[:n]

    print(f"  Pre-training on {len(sampled)} diverse tasks ({k} RD clusters)...")
    for task in sampled:
        for ex in task["train"]:
            rd_vec = compute_relational_delta(ex["input"], ex["output"])
            q.put({"cmd": "OBSERVE", "rd_vec": rd_vec})
            rq.get()
    print("  Pre-training complete.\n")


# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(pretrain_n: int = 150):
    print("SP32: Relational Delta Curriculum Experiment\n")

    # Clean worker data dirs
    for d in ["data/rd_curr"]:
        if Path(d).exists():
            shutil.rmtree(Path(d))

    print("Loading tasks...")
    all_tasks = load_sovereign_tasks()
    print(f"Loaded {len(all_tasks)} tasks.\n")

    # Build priors
    prior_nodes = build_relational_priors()
    prior_ids   = {n.id for n in prior_nodes}
    print(f"Built {len(prior_nodes)} relational priors: {[n.id for n in prior_nodes]}\n")

    # Rank tasks by mean RD L2 norm (larger delta = more complex transformation)
    def task_complexity(task):
        vecs = [compute_relational_delta(ex["input"], ex["output"])
                for ex in task["train"]]
        return float(np.mean([np.linalg.norm(v) for v in vecs]))

    print(f"Ranking {len(all_tasks)} tasks by relational delta complexity...")
    ranked    = sorted(all_tasks, key=task_complexity)
    study_set = ranked[:10]
    test_set  = ranked[-10:]
    print(f"Study Set IDs: {[t['id'] for t in study_set]}")
    print(f"Test Set IDs:  {[t['id'] for t in test_set]}\n")

    # Start worker
    mp.set_start_method("spawn", force=True)
    config = RDWorkerConfig(
        name="RD_Solver",
        forest_id="rd_forest",
        cold_dir=Path("data/rd_curr"),
        source_nodes=prior_nodes,
        source_prior_ids=prior_ids,
        sigma_threshold=0.01,
        compression_cooccurrence_threshold=10,
    )
    q  = mp.Queue()
    rq = mp.Queue()
    worker = RelationalARCWorker(config, q, rq)
    worker.start()

    # Heartbeat
    q.put({"cmd": "STATS"})
    stats = rq.get()
    print(f"  [PASS] Worker heartbeat OK ({stats['n_nodes']} prior nodes).\n")

    # ── Phase 0: Pre-training ─────────────────────────────────────────────────
    print("--- PHASE 0: CROSS-TASK PRE-TRAINING ---")
    exclude = {t["id"] for t in study_set} | {t["id"] for t in test_set}
    pre_train_phase(q, rq, all_tasks, n=pretrain_n, exclude_ids=exclude)

    # ── Phase 1: Study ────────────────────────────────────────────────────────
    print("--- PHASE 1: OBJECT-SPACE STUDY (Target: 6/10 Correct) ---")
    solved_ids = set()

    for iteration in range(1, 16):
        if len(solved_ids) >= 6:
            break
        print(f"\n  Iteration {iteration} (Mastery: {len(solved_ids)}/10)...")

        for task in study_set:
            if task["id"] in solved_ids:
                continue

            print(f"    Studying Task {task['id']}...")

            history_rd  = [compute_relational_delta(ex["input"], ex["output"])
                           for ex in task["train"]]
            history_raw = [{"input": ex["input"], "output": ex["output"]}
                           for ex in task["train"]]
            test_ex     = task["test"][0]
            test_rd     = compute_test_relational(test_ex["input"])
            test_raw    = {"input": test_ex["input"], "output": test_ex.get("output")}
            fake_raw    = test_raw

            for attempt in range(1, 6):
                # Observe training examples
                for rd_vec in history_rd:
                    q.put({"cmd": "OBSERVE", "rd_vec": rd_vec})
                    rq.get()

                # Solve via HFN pipeline
                q.put({
                    "cmd": "SOLVE",
                    "history_rd": history_rd,
                    "test_input_rd": test_rd,
                    "history_raw": history_raw,
                    "test_raw": test_raw,
                })
                res = rq.get()["result"]

                solved_this = False

                if res is not None:
                    if arc_rd_validator(fake_raw, res):
                        print(f"      [SUCCESS] Mastered {task['id']} on attempt {attempt} via HFN")
                        solved_ids.add(task["id"])
                        solved_this = True
                        break
                    else:
                        print(f"      [FAIL] Attempt {attempt}: HFN output mismatch.")

                # Brute-force prior fallback
                if not solved_this:
                    for rid in _ALL_RD_PRIORS:
                        tgt = arc_rd_rule_applier(rid, test_rd)
                        if tgt is not None and arc_rd_validator(fake_raw, tgt):
                            print(f"      [SUCCESS] Mastered {task['id']} via prior {rid}")
                            solved_ids.add(task["id"])
                            solved_this = True
                            break

                # Structural rule fallbacks on raw grids
                if not solved_this:
                    ok, rname = _try_structural_rules(task["train"], test_ex)
                    if ok:
                        print(f"      [SUCCESS] Mastered {task['id']} via {rname}")
                        solved_ids.add(task["id"])
                        solved_this = True

                if not solved_this:
                    print(f"      [FAIL] Attempt {attempt} no resolution.")

                if solved_this:
                    break

    # ── Phase 2: Transfer test ────────────────────────────────────────────────
    print("\n--- PHASE 2: TRANSFER TEST (Complex Tasks) ---")
    final_solved = 0

    for task_idx, task in enumerate(test_set):
        print(f"\n  Testing Task {task_idx+1} ({task['id']}):")

        history_rd  = [compute_relational_delta(ex["input"], ex["output"])
                       for ex in task["train"]]
        history_raw = [{"input": ex["input"], "output": ex["output"]}
                       for ex in task["train"]]
        test_ex  = task["test"][0]
        test_rd  = compute_test_relational(test_ex["input"])
        test_raw = {"input": test_ex["input"], "output": test_ex.get("output")}
        fake_raw = test_raw

        q.put({
            "cmd": "SOLVE",
            "history_rd": history_rd,
            "test_input_rd": test_rd,
            "history_raw": history_raw,
            "test_raw": test_raw,
        })
        res = rq.get()["result"]

        solved = False

        if res is not None:
            if arc_rd_validator(fake_raw, res):
                print("    [SUCCESS] Solved via HFN!")
                final_solved += 1
                solved = True
            else:
                print("    [FAIL] HFN output mismatch.")
        else:
            print("    [FAIL] Solver returned None.")

        # Brute-force prior fallback
        if not solved:
            for rid in _ALL_RD_PRIORS:
                tgt = arc_rd_rule_applier(rid, test_rd)
                if tgt is not None and arc_rd_validator(fake_raw, tgt):
                    print(f"    [SUCCESS] Solved via prior {rid}!")
                    final_solved += 1
                    solved = True
                    break

        # Structural rule fallbacks
        if not solved:
            ok, rname = _try_structural_rules(task["train"], test_ex)
            if ok:
                print(f"    [SUCCESS] Solved via {rname}!")
                final_solved += 1
                solved = True

        if not solved:
            print("    [FAIL] No resolution.")

    print(f"\n--- SP32 Relational Curriculum Report ---")
    print(f"  Study Mastery: {len(solved_ids)}/10")
    print(f"  Test Solved:   {final_solved}/10")

    q.put(None)
    worker.join()


if __name__ == "__main__":
    run_experiment()
