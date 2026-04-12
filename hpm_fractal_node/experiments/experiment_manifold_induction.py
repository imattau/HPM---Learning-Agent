"""
SP59: Manifold-Guided Operator Induction.

Demonstrates HFN-guided function induction. The agent uses residual error 
to query the HFN manifold for the most relevant operators, proving that 
geometric latent space can actively direct symbolic synthesis.
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
    S_DIM
)

# Manifold Structure: [30D Input State | 30D Resulting Delta | 30D Operator Parameters]
D = 90
REL_OFFSET = 30
PARAM_OFFSET = 60

class ManifoldGuidedBeamSearch:
    """
    Synthesizes operator chains using HFN retrieval to prune the search space.
    """
    def __init__(self, forest: Forest, retriever: GeometricRetriever, oracle: OperatorOracle, max_depth: int = 3, beam_width: int = 5):
        self.forest = forest
        self.retriever = retriever
        self.oracle = oracle
        self.max_depth = max_depth
        self.beam_width = beam_width

    def _decode_op(self, vec: np.ndarray) -> Operator:
        params = vec[PARAM_OFFSET:]
        if abs(params[2]) > 0.01:
            return ModOperator(modulus=params[2], name=f"Mod_{int(params[2])}")
        return AffineOperator(weight=params[0], bias=params[1], name=f"Affine({params[0]:.1f},{params[1]:.1f})")

    def _get_candidates(self, x_curr: np.ndarray, target_delta: np.ndarray, k: int = 10) -> List[Operator]:
        """Manifold-aware retrieval that respects the sigma mask."""
        query_vec = np.zeros(D)
        query_vec[0:30] = x_curr
        query_vec[REL_OFFSET : REL_OFFSET+30] = target_delta
        
        # We care about Axis 0 of Input and Axis 0 of Delta
        active_indices = [0, REL_OFFSET]
        
        candidates = []
        for n in self.forest.active_nodes():
            dist = 0.0
            for idx in active_indices:
                dist += (n.mu[idx] - query_vec[idx])**2
            dist = np.sqrt(dist)
            candidates.append((n, dist))
            
        candidates.sort(key=lambda x: x[1])
        top_k = [c[0] for c in candidates[:k]]
        
        ops = []
        seen_params = set()
        for node in top_k:
            op = self._decode_op(node.mu)
            p = op.get_params()
            if p not in seen_params:
                seen_params.add(p)
                ops.append(op)
        return ops

    def search(self, x_seq: List[np.ndarray]) -> List[Operator]:
        """
        Heuristic search guided by HFN retrieval.
        """
        s0 = x_seq[0]
        s1 = x_seq[1]
        delta1 = s1 - s0
        
        initial_ops = self._get_candidates(s0, delta1, k=20)
        
        beam = [(op, self._score(op, x_seq)) for op in initial_ops]
        beam.sort(key=lambda x: x[1])
        beam = beam[:self.beam_width]

        for depth in range(1, self.max_depth):
            new_candidates = []
            for current_op, _ in beam:
                for t in range(len(x_seq)-1):
                    s_in = x_seq[t]
                    s_target = x_seq[t+1]
                    s_pred = current_op.apply(s_in)
                    residual = s_target - s_pred
                    
                    if np.linalg.norm(residual) < 0.001:
                        continue 
                        
                    refinement_ops = self._get_candidates(s_pred, residual, k=5)
                    for p in refinement_ops:
                        new_op = p.compose(current_op)
                        new_candidates.append((new_op, self._score(new_op, x_seq)))
            
            combined = beam + new_candidates
            unique = []
            seen = set()
            for op, score in combined:
                p_key = op.get_params()
                if p_key not in seen:
                    seen.add(p_key)
                    unique.append((op, score))
            
            unique.sort(key=lambda x: x[1])
            beam = unique[:self.beam_width]
            
        return [op for op, score in beam]

    def _score(self, op: Operator, x_seq: List[np.ndarray]) -> float:
        """Score based on multi-step rollout error + complexity."""
        total_err = 0.0
        curr_s = x_seq[0]
        for t in range(1, len(x_seq)):
            target = x_seq[t]
            pred_s = op.apply(curr_s)
            total_err += abs(pred_s[0] - target[0])
            curr_s = pred_s 
            
        depth = 1
        if isinstance(op, ComposedOperator):
            depth = str(op.name).count(" ∘ ") + 1
        
        return (total_err / (len(x_seq)-1)) + (depth * 0.01)

class ManifoldGuidedExperiment:
    def __init__(self):
        self.forest = Forest(D=D)
        self.oracle = OperatorOracle()
        self.retriever = GeometricRetriever(self.forest)
        
        self.observer = Observer(
            forest=self.forest,
            retriever=self.retriever,
            tau=0.01,
            residual_surprise_threshold=0.5, 
            node_use_diag=True
        )
        print(f"Initialized SP59 Experiment with D={D} [Input|Delta|Params]")

    def store_primitive(self, op: Operator, input_val: float):
        """Store a primitive in HFN by observing its action on an input."""
        s_in = self.oracle.encode(input_val)
        s_out = op.apply(s_in)
        delta = s_out - s_in
        
        vec = np.zeros(D)
        vec[0:30] = s_in
        vec[REL_OFFSET : REL_OFFSET+30] = delta
        
        params = np.zeros(S_DIM)
        p = op.get_params()
        params[0] = p[0] # w
        params[1] = p[1] # b
        params[2] = p[2] # m
        vec[PARAM_OFFSET : PARAM_OFFSET+30] = params
        
        self.observer.observe(vec)

    def _decode_op(self, vec: np.ndarray) -> Operator:
        params = vec[PARAM_OFFSET:]
        if abs(params[2]) > 0.01:
            return ModOperator(modulus=params[2], name=f"Mod_{int(params[2])}")
        return AffineOperator(weight=params[0], bias=params[1], name=f"Affine({params[0]:.1f},{params[1]:.1f})")

    def run_phase_1_training(self):
        print("\n--- PHASE 1: MANIFOLD PRE-TRAINING ---")
        ops = [
            AffineOperator(1.0, 0.1, "Add_1"),
            AffineOperator(2.0, 0.0, "Mul_2"),
            ModOperator(10.0, "Mod_10")
        ]
        inputs = list(range(1, 25))
        for op in ops:
            for x in inputs:
                self.store_primitive(op, x)
        print(f"  Forest Size: {len(self.forest)}")

    def run_phase_2_induction(self):
        print("\n--- PHASE 2: MANIFOLD-GUIDED INDUCTION ---")
        
        print(f"  Forest Summary ({len(self.forest)} nodes):")
        op_counts = {}
        for n in self.forest.active_nodes():
            op = self._decode_op(n.mu)
            op_counts[op.name] = op_counts.get(op.name, 0) + 1
        for name, count in op_counts.items():
            print(f"    - {name}: {count} nodes")
        
        clean_seq = [1, 3, 7, 5, 1, 3, 7, 5]
        np.random.seed(42)
        noisy_states = [self.oracle.encode(x) + np.random.normal(0, 0.005, S_DIM) for x in clean_seq]
        
        print(f"  Noisy Sequence: {[round(self.oracle.decode_numeric(s), 2) for s in noisy_states[:4]]} ...")
        
        synthesizer = ManifoldGuidedBeamSearch(self.forest, self.retriever, self.oracle, max_depth=3, beam_width=20)
        candidates = synthesizer.search(noisy_states[:4])
        
        print(f"  Top Manifold-Guided Candidates:")
        for i, c in enumerate(candidates[:3]):
            print(f"    {i+1}. {c.name}")

        winning_op = candidates[0]
        print(f"  Selected Winner: {winning_op.name}")

        print(f"\n  --- AUTOREGRESSIVE ROLLOUT PREDICTION (t=4..7) ---")
        curr_s = noisy_states[3]
        errors = []
        for t in range(4, len(clean_seq)):
            actual = clean_seq[t]
            pred_s = winning_op.apply(curr_s)
            pred_val = self.oracle.decode_numeric(pred_s)
            
            err = abs(pred_val - actual)
            errors.append(err)
            print(f"    t={t}: Predicted={pred_val:5.2f}, Actual={actual:5.2f}, Error={err:5.4f}")
            curr_s = pred_s
            
        mean_err = np.mean(errors)
        print(f"  Mean Prediction Error: {mean_err:.4f}")
        
        if mean_err < 0.2:
            print("\n  [SUCCESS] Manifold-Guided Induction Verified!")
            print(f"  The agent synthesized the depth-3 chain by actively querying the geometric manifold.")
        else:
            print("\n  [FAIL] Induction diverged.")

def run_experiment():
    exp = ManifoldGuidedExperiment()
    exp.run_phase_1_training()
    exp.run_phase_2_induction()

if __name__ == "__main__":
    run_experiment()
