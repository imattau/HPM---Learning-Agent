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
import re
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
    load_sovereign_tasks, COMMON_D, S_SLICE, M_SLICE, C_SLICE, S_DIM, D_SLICE, I_SLICE
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
    compression_cooccurrence_threshold: int = 10

class SovereignARCWorker(mp.Process):
    def __init__(self, config: WorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"Worker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
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
        
        def arc_validator(raw_ex: dict, predicted_vector: np.ndarray) -> bool:
            # predicted_vector is full manifold (1850D) or target_slice (930D).
            # reconstruct_grid expects [delta(900), attrs(30+)]; slice from offset 900.
            offset = 900 if predicted_vector.shape[0] > 930 else 0
            delta_attrs = predicted_vector[offset:offset+930].copy()
            # delta_attrs[920]/[921] = predicted output origin from symbolic slice — do NOT override.
            pred_grid = reconstruct_grid(raw_ex["input"], delta_attrs, target_shape=raw_ex["output"].shape)
            return np.array_equal(pred_grid, raw_ex["output"])

        def arc_rule_applier(rule_id: str, test_input: np.ndarray) -> np.ndarray | None:
            """Apply a named geometric rule to compute delta for the test input.
            Handles both functional prior IDs (prior_flip_v, prior_rot_180, prior_identity)
            and compressed/abbreviated IDs (prior_fl, prior_ro, prior_id) from arc_prior_forest.
            """
            input_norm = test_input[0:900]
            input_30x30 = (input_norm.reshape(30, 30) * 9.0)
            rid = rule_id.lower()

            # Determine transform — check both full names and abbreviations
            # identity: 'prior_identity', 'prior_id', but NOT 'prior_identity_rule'
            is_identity = ('prior_identity' in rid and 'prior_identity_r' not in rid) or \
                          (re.search(r'\bprior_id\b|prior_id[,)]', rid) is not None)
            is_flip_v   = 'prior_flip_v' in rid or 'prior_fl_v' in rid or \
                          re.search(r'\bprior_fv\b|prior_fv[,)]', rid) is not None
            is_flip_h   = 'prior_flip_h' in rid or 'prior_fl_h' in rid or \
                          re.search(r'\bprior_fh\b|prior_fh[,)]', rid) is not None
            # prior_fl without _v or _h suffix — treat as flip_h by convention
            is_flip_gen = (re.search(r'\bprior_fl\b|prior_fl[,)]', rid) is not None) and \
                          'prior_fl_v' not in rid and 'prior_fl_h' not in rid and \
                          'prior_flip' not in rid
            is_rot_180  = 'rot_180' in rid or \
                          re.search(r'\bprior_ro\b|prior_ro[,)]', rid) is not None
            is_rot_90   = 'rot_90' in rid and 'rot_270' not in rid and 'rot_180' not in rid
            is_rot_270  = 'rot_270' in rid

            if is_identity:
                transform = lambda g: g.copy()
            elif is_flip_v:
                transform = lambda g: np.flip(g, axis=0).copy()
            elif is_flip_h or is_flip_gen:
                transform = lambda g: np.flip(g, axis=1).copy()
            elif is_rot_180:
                transform = lambda g: np.rot90(g, k=2).copy()
            elif is_rot_90:
                transform = lambda g: np.rot90(g, k=1).copy()
            elif is_rot_270:
                transform = lambda g: np.rot90(g, k=3).copy()
            else:
                return None
            output_30x30 = transform(input_30x30)
            delta = output_30x30.flatten() / 9.0 - input_norm
            result = np.zeros(930)
            result[:900] = delta
            return result

        from hfn.reasoning import CognitiveSolver
        self.solver = CognitiveSolver(self.observer, self.decoder, self.evaluator,
                                      validator=arc_validator, rule_applier=arc_rule_applier)

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE":
                x_full = task["x"]
                if self.config.name == "Spatial_Spec": x = x_full[S_SLICE]
                elif self.config.name == "Symbolic_Spec": x = x_full[M_SLICE]
                else: x = x_full

                acc = self.evaluator.accuracy(x, self.forest)
                if acc < self.config.competence_threshold:
                    self.result_queue.put({"name": self.config.name, "competent": False})
                else:
                    res = self.observer.observe(x)
                    self.forest._on_observe()
                    winners = [{"id": n.id, "mu": n.mu.copy()} for n in res.explanation_tree[:3]]
                    if winners:
                        print(f"      [DEBUG] {self.config.name} winners: {[w['id'] for w in winners]}")
                    self.result_queue.put({"name": self.config.name, "competent": True, "winners": winners})
            
            elif cmd == "SOLVE":
                history_full = task["history"]
                test_input_full = task["test_input"]
                history_raw = task.get("history_raw")
                test_input_raw = task.get("test_input_raw")
                
                # Slicing
                if self.config.name == "Spatial_Spec":
                    # Manifold: [Input(900), Delta(900)]
                    history = [h[S_SLICE] for h in history_full]
                    test_input = test_input_full[S_SLICE]
                    # Target is the DELTA portion (900:1800)
                    target_slice = slice(900, 1800)
                elif self.config.name == "Symbolic_Spec":
                    history = [h[M_SLICE] for h in history_full]
                    test_input = test_input_full[M_SLICE]
                    target_slice = slice(None)
                else:
                    history = history_full
                    test_input = test_input_full
                    # Explorer targets Delta (900:1800) + Symbolic (1800:1830)
                    target_slice = slice(900, 1830) 
                
                res = self.solver.solve(history, test_input, target_slice, history_raw=history_raw, test_input_raw=test_input_raw)
                self.result_queue.put({"name": self.config.name, "result": res})

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
                    # Tighter threshold for ARC precision
                    if np.linalg.norm(obs - target_mu) < 0.1:
                        leaf = HFN(mu=obs, sigma=np.ones(self.config.common_d)*0.001, id=f"discovered_{int(np.sum(obs*100))}", use_diag=True)
                        for e in edges: leaf.add_edge(leaf, e.target, e.relation)
                        self.forest.register(leaf)
                        found = True; break
                self.result_queue.put({"name": self.config.name, "success": found})

            
            elif cmd == "STATS":
                self.result_queue.put({"name": self.config.name, "status": "OK"})
            
            elif cmd == "GET_NODE":
                nid = task["id"]
                node = self.forest.get(nid)
                self.result_queue.put({"name": self.config.name, "node": node})
            
            elif cmd == "REGISTER_NODE":
                node = task["node"]
                # Deep register
                def reg_recursive(n):
                    if n.id in self.forest: return
                    # DIMENSION CHECK
                    if n.mu.shape[0] != self.config.common_d:
                        return # Skip cross-domain nodes for now
                    
                    clone = HFN(mu=n.mu.copy(), sigma=n.sigma.copy(), id=n.id, use_diag=n.use_diag)
                    for c in n.children():
                        reg_recursive(c)
                        # Only add child if it was successfully cloned (matched dim)
                        child_node = self.forest.get(c.id)
                        if child_node:
                            clone.add_child(child_node)
                    self.forest.register(clone)
                reg_recursive(node)
                self.result_queue.put({"name": self.config.name, "status": "OK"})
def predict_shape(task: dict) -> tuple[int, int]:
    """Look at train examples to predict output shape."""
    # Logic: If all train examples have the same shape ratio or delta, use it.
    # For now, simplest: if input_shape == output_shape for all, return test_input.shape
    ratios = []
    for ex in task["train"]:
        in_r, in_c = ex["input"].shape
        out_r, out_c = ex["output"].shape
        ratios.append((out_r / in_r, out_c / in_c))

    if len(set(ratios)) == 1:
        # Consistent ratio found
        rr, rc = ratios[0]
        test_in_r, test_in_c = task["test"][0]["input"].shape
        return int(np.round(test_in_r * rr)), int(np.round(test_in_c * rc))

    return task["test"][0]["input"].shape

def reconstruct_grid(input_grid: np.ndarray, delta_900d: np.ndarray, rule_node: HFN | None = None, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    # 1. Build canonical input (centered 30x30), matching grid_to_vec encoding
    in_coords = np.argwhere(input_grid > 0)
    canon_in = np.zeros((30, 30))
    if in_coords.size > 0:
        iy0, ix0 = in_coords.min(axis=0)
        iy1, ix1 = in_coords.max(axis=0) + 1
        content = input_grid[iy0:iy1, ix0:ix1]
        cr, cc = content.shape
        r_start = (30 - cr) // 2
        c_start = (30 - cc) // 2
        r_size, c_size = min(cr, 30), min(cc, 30)
        canon_in[r_start:r_start+r_size, c_start:c_start+c_size] = content[:r_size, :c_size]

    # 2. Apply delta in canonical space
    delta_2d = (delta_900d[:900] * 9.0).reshape(30, 30)
    canon_out = np.clip(np.rint(canon_in + delta_2d), 0, 9).astype(int)

    # 3. Find output content bounding box in canonical canvas
    out_coords = np.argwhere(canon_out > 0)
    if out_coords.size == 0:
        out_r, out_c = target_shape if target_shape else input_grid.shape
        return np.zeros((max(1, out_r), max(1, out_c)), dtype=int)

    cy0, cx0 = out_coords.min(axis=0)
    cy1, cx1 = out_coords.max(axis=0) + 1
    out_content = canon_out[cy0:cy1, cx0:cx1]

    # 4. Determine world output origin.
    # The canonical output content is centered around the same position as the
    # canonical input content (both centered in the 30x30 canvas). The canonical
    # center offset of the output content maps directly to a world offset from the
    # input content center. This is input-position-independent.
    #
    # World output top-left = world input center + (canonical output top-left - canonical center 14.5)
    out_r, out_c = target_shape if target_shape else input_grid.shape
    out_r, out_c = max(1, min(30, out_r)), max(1, min(30, out_c))

    # Canonical center of the 30x30 canvas
    canon_center = 14.5

    if in_coords.size > 0:
        # World input center (float)
        w_in_center_r = (in_coords[:, 0].min() + in_coords[:, 0].max()) / 2.0
        w_in_center_c = (in_coords[:, 1].min() + in_coords[:, 1].max()) / 2.0
    else:
        w_in_center_r, w_in_center_c = 0.0, 0.0

    # Canonical output top-left offset from canvas center
    can_out_offset_r = cy0 - canon_center
    can_out_offset_c = cx0 - canon_center

    wy0 = int(np.rint(w_in_center_r + can_out_offset_r))
    wx0 = int(np.rint(w_in_center_c + can_out_offset_c))

    # 5. Place output content at world origin in final grid
    final = np.zeros((out_r, out_c), dtype=int)
    wy0 = max(0, min(out_r - 1, wy0))
    wx0 = max(0, min(out_c - 1, wx0))
    r_fit = min(out_content.shape[0], out_r - wy0)
    c_fit = min(out_content.shape[1], out_c - wx0)
    if r_fit > 0 and c_fit > 0:
        final[wy0:wy0+r_fit, wx0:wx0+c_fit] = out_content[:r_fit, :c_fit]

    return final

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
        WorkerConfig("Spatial_Spec", "s_think_sp28", Path("data/think_s_sp28"), "OBSERVER", common_d=1800, competence_threshold=0.0, source_nodes=list(spatial_forest.active_nodes()), source_prior_ids=spatial_priors),
        WorkerConfig("Symbolic_Spec", "m_think_sp28", Path("data/think_m_sp28"), "OBSERVER", common_d=109, competence_threshold=0.0, source_nodes=list(math_base.active_nodes()), source_prior_ids=math_priors),
        WorkerConfig("Spatial_Decoder", "d_think_sp28", Path("data/think_d_sp28"), "DECODER", common_d=1800, sigma_threshold=0.1)
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
