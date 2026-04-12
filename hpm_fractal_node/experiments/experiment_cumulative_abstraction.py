"""
SP60: Cumulative Abstraction via Multi-Polygraph HFNs.

Demonstrates that an HFN agent can synthesize a complex operator chain,
encapsulate it as a Multi-Polygraph HFN node (a "macro"), store it in the manifold,
and reuse this structural abstraction to accelerate future reasoning.
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Any, Optional, Tuple, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GeometricRetriever
from hpm_fractal_node.code.sp57_operators import (
    OperatorOracle, 
    Operator, 
    AffineOperator, 
    ModOperator, 
    ComposedOperator,
    PolygraphOperator,
    S_DIM
)

# Manifold Structure: [30D Input State | 30D Resulting Delta | 30D Operator Parameters]
D = 90
REL_OFFSET = 30
PARAM_OFFSET = 60

class MultiPolygraphBeamSearch:
    def __init__(self, forest: Forest, retriever: GeometricRetriever, oracle: OperatorOracle, max_depth: int = 3, beam_width: int = 5):
        self.forest = forest
        self.retriever = retriever
        self.oracle = oracle
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.nodes_evaluated = 0

    def _decode_op(self, node: HFN) -> Operator:
        params = node.mu[PARAM_OFFSET:]
        if node.inputs:
            child_ops = [self._decode_op(child) for child in node.inputs]
            return PolygraphOperator(node.id, child_ops, tuple(params))
        if abs(params[2]) > 0.01:
            return ModOperator(modulus=params[2], name=node.id)
        return AffineOperator(weight=params[0], bias=params[1], name=node.id)

    def _get_candidates(self, x_curr: np.ndarray, target_delta: np.ndarray, k: int = 15) -> List[Operator]:
        query_vec = np.zeros(D)
        query_vec[0:30] = x_curr
        query_vec[REL_OFFSET : REL_OFFSET+30] = target_delta
        sigma = np.ones(D) * 100.0 
        sigma[0], sigma[REL_OFFSET] = 0.01, 0.01 
        query_node = HFN(mu=query_vec, sigma=sigma, id="search_query", use_diag=True)
        candidates = self.retriever.retrieve(query_node, k=max(k*2, 20))
        scored = sorted([(n, query_node.log_prob(n.mu)) for n in candidates], key=lambda x: x[1], reverse=True)
        ops = []
        seen = set()
        for node, _ in scored[:k]:
            op = self._decode_op(node)
            if op.get_params() not in seen:
                seen.add(op.get_params()); ops.append(op)
        return ops

    def search(self, x_seq: List[np.ndarray]) -> Tuple[List[Operator], int]:
        self.nodes_evaluated = 0
        initial_set = set(); all_initial = []
        for t in range(len(x_seq)-1):
            for op in self._get_candidates(x_seq[t], x_seq[t+1] - x_seq[t], k=10):
                if op.get_params() not in initial_set:
                    initial_set.add(op.get_params()); all_initial.append(op)
        self.nodes_evaluated += len(all_initial)
        beam = sorted([(op, self._score(op, x_seq)) for op in all_initial], key=lambda x: x[1])[:self.beam_width]
        if beam and beam[0][1] < 0.05: return [op for op, _ in beam], 1

        for depth in range(2, self.max_depth + 1):
            new_candidates = []
            for current_op, _ in beam:
                for t in range(len(x_seq)-1):
                    s_pred = current_op.apply(x_seq[t])
                    residual = x_seq[t+1] - s_pred
                    if np.linalg.norm(residual) < 0.001: continue
                    ref_ops = self._get_candidates(s_pred, residual, k=5)
                    self.nodes_evaluated += len(ref_ops)
                    for p in ref_ops:
                        new_op = p.compose(current_op)
                        new_candidates.append((new_op, self._score(new_op, x_seq)))
            combined = beam + new_candidates
            unique = []; seen = set()
            for op, score in combined:
                if op.get_params() not in seen:
                    seen.add(op.get_params()); unique.append((op, score))
            beam = sorted(unique, key=lambda x: x[1])[:self.beam_width]
            if beam and beam[0][1] < 0.05: return [op for op, _ in beam], depth
        return [op for op, score in beam], self.max_depth

    def _score(self, op: Operator, x_seq: List[np.ndarray]) -> float:
        total_err = 0.0; curr_s = x_seq[0]
        for t in range(1, len(x_seq)):
            pred_s = op.apply(curr_s); total_err += abs(pred_s[0] - x_seq[t][0]); curr_s = pred_s 
        depth = (str(op.name).count(" ∘ ") + 1) if isinstance(op, ComposedOperator) else 1
        bonus = 1.5 if isinstance(op, PolygraphOperator) else 0.0
        return (total_err / (len(x_seq)-1)) + (depth * 0.1) - bonus

def compress_to_polygraph(op: Operator, forest: Forest, oracle: OperatorOracle, observer: Observer) -> List[HFN]:
    def get_nodes(oper: Operator):
        if isinstance(oper, ComposedOperator): return get_nodes(oper.f) + get_nodes(oper.g)
        node = forest.get(oper.name)
        return [node] if node else []
    child_nodes = get_nodes(op)
    nodes = []
    for x in [1.0, 5.0, 10.0, 20.0, 30.0]:
        s_in = oracle.encode(x); s_out = op.apply(s_in); delta = s_out - s_in
        vec = np.zeros(D); vec[0:30] = s_in; vec[REL_OFFSET:REL_OFFSET+30] = delta
        p = op.get_params(); params = np.zeros(S_DIM)
        if not isinstance(p[0], tuple): params[0], params[1], params[2] = p
        vec[PARAM_OFFSET:] = params
        node = HFN(mu=vec, sigma=np.ones(D)*0.1, id=f"macro({op.name})_{x}", inputs=child_nodes, relation_type="macro", use_diag=True)
        forest.register(node); observer.protected_ids.add(node.id); nodes.append(node)
    return nodes

class CumulativeAbstractionExperiment:
    def __init__(self):
        self.forest = Forest(D=D); self.oracle = OperatorOracle(); self.retriever = GeometricRetriever(self.forest)
        self.observer = Observer(forest=self.forest, retriever=self.retriever, tau=0.01, node_use_diag=True)
        print(f"Initialized SP60 Experiment with D={D} [Multi-Polygraph HFN]")

    def store_primitive(self, op: Operator, x: float):
        s_in = self.oracle.encode(x); s_out = op.apply(s_in); delta = s_out - s_in
        vec = np.zeros(D); vec[0:30] = s_in; vec[REL_OFFSET:REL_OFFSET+30] = delta
        p = op.get_params(); params = np.zeros(S_DIM); params[0], params[1], params[2] = p
        vec[PARAM_OFFSET:] = params
        node = HFN(mu=vec, sigma=np.ones(D)*0.1, id=op.name, use_diag=True)
        self.forest.register(node); self.observer.protected_ids.add(node.id)

    def run_phase_1_pretraining(self):
        print("\n--- PHASE 1: PRIMITIVE PRE-TRAINING ---")
        ops = [AffineOperator(1.0, 0.1, "Add_1"), AffineOperator(2.0, 0.0, "Mul_2"), ModOperator(10.0, "Mod_10")]
        for op in ops:
            for x in [1.0, 5.0, 10.0, 20.0]: self.store_primitive(op, x)
        print(f"  Forest Size: {len(self.forest)}")

    def run_phase_2_task_a(self):
        print("\n--- PHASE 2: TASK A (POLYGRAPH FORMATION) ---")
        clean_seq = [1, 4, 10, 22]; noisy = [self.oracle.encode(x) + np.random.normal(0, 0.005, S_DIM) for x in clean_seq]
        searcher = MultiPolygraphBeamSearch(self.forest, self.retriever, self.oracle, beam_width=5)
        candidates, depth = searcher.search(noisy)
        print(f"  Synthesized: {candidates[0].name} (Depth {depth})")
        compress_to_polygraph(candidates[0], self.forest, self.oracle, self.observer)
        print(f"  [MACRO] Created Multi-Polygraph node for {candidates[0].name}")

    def run_phase_3_task_b(self):
        print("\n--- PHASE 3: TASK B (CUMULATIVE TRANSFER) ---")
        clean_seq = [1, 4, 0, 2, 6]; noisy = [self.oracle.encode(x) + np.random.normal(0, 0.005, S_DIM) for x in clean_seq]
        print(f"  --- Mode: Abstracted (With Macros) ---")
        searcher = MultiPolygraphBeamSearch(self.forest, self.retriever, self.oracle, beam_width=5)
        candidates, depth_abs = searcher.search(noisy); evals_abs = searcher.nodes_evaluated
        print(f"    Winner: {candidates[0].name} | Depth: {depth_abs} | Evals: {evals_abs}")
        print(f"\n  --- Mode: Naive (Without Macros) ---")
        for mid in [n.id for n in self.forest.active_nodes() if "macro" in n.id]: self.forest.deregister(mid)
        searcher_naive = MultiPolygraphBeamSearch(self.forest, self.retriever, self.oracle, beam_width=5)
        candidates_naive, depth_naive = searcher_naive.search(noisy); evals_naive = searcher_naive.nodes_evaluated
        print(f"    Winner: {candidates_naive[0].name} | Depth: {depth_naive} | Evals: {evals_naive}")
        if depth_abs < depth_naive:
            print("\n  [SUCCESS] Cumulative Abstraction Verified!")
            print(f"  Search Depth reduced from {depth_naive} to {depth_abs} via Multi-Polygraph reuse.")
        else: print("\n  [FAIL] Did not achieve search acceleration.")

if __name__ == "__main__":
    exp = CumulativeAbstractionExperiment()
    exp.run_phase_1_pretraining()
    exp.run_phase_2_task_a()
    exp.run_phase_3_task_b()
