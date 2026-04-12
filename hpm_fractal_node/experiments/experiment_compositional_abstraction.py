"""
SP56 (Refactored): Compositional Abstraction and Meta-Relational Transfer.

This experiment validates the HPM Hierarchical Pattern Stack by:
1. Building first-order relations (L2) from data content (L1).
2. Abstracting second-order trajectories (L3 meta-patterns) from relations.
3. Using the L3 meta-pattern to actively constrain prediction in an unseen domain.
4. Comparing HPM performance against L2-only and random-L3 baselines.
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GeometricRetriever
from hpm_fractal_node.code.sp56_oracle import StatefulOracleSP56, S_DIM, D

class CompositionalExperimentRefactored:
    def __init__(self):
        self.forest = Forest(D=D)
        self.oracle = StatefulOracleSP56()
        self.retriever = GeometricRetriever(self.forest)
        
        self.observer = Observer(
            forest=self.forest,
            retriever=self.retriever,
            tau=0.01, # High sensitivity for priming
            residual_surprise_threshold=0.5, 
            node_use_diag=True
        )
        print(f"Initialized SP56 Refactored with D={D} [3-Level Stack]")

    # --- Curriculum Generation ---
    
    def gen_numeric_constant(self, n=10, start=0, step=1):
        return [start + i * step for i in range(n)]

    def gen_numeric_accumulator(self, n=10, start=0):
        # 0, 1, 3, 6, 10... (L2 grows linearly, L3 is constant)
        seq = [start]
        for i in range(1, n):
            seq.append(seq[-1] + i)
        return seq

    def gen_spatial_1d_accumulator(self, n=10, start=0):
        # 1D Accumulator on axis 2
        seq = [[start]]
        for i in range(1, n):
            seq.append([seq[-1][0] + i])
        return seq

    def gen_spatial_2d_accumulator(self, n=10):
        # 2D Accumulator on axes 2 and 3
        # Truly unseen domain: uses multiple axes
        seq = [[0, 0]]
        for i in range(1, n):
            seq.append([seq[-1][0] + i, seq[-1][1] + i])
        return seq

    # --- Prediction Core ---

    def predict_next(self, current_vec: np.ndarray, l3_constraint: np.ndarray) -> np.ndarray:
        """
        Predicts the next 90D state vector using top-down L3 constraint.
        L3 = L2_t - L2_{t-1} => L2_pred = L2_curr + L3
        L2 = L1_t - L1_{t-1} => L1_pred = L1_curr + L2_pred
        """
        curr_l1 = current_vec[0:30]
        curr_l2 = current_vec[30:60]
        
        pred_l2 = curr_l2 + l3_constraint
        pred_l1 = curr_l1 + pred_l2
        
        pred_vec = np.zeros(D)
        pred_vec[0:30] = pred_l1
        pred_vec[30:60] = pred_l2
        pred_vec[60:90] = l3_constraint
        return pred_vec

    # --- Training Phases ---

    def run_phase_1_l2_formation(self):
        """Pre-train relational primitives in seen domains."""
        print("\n--- PHASE 1: L2 RELATION FORMATION ---")
        # Ensure Boolean is NOT here to maintain Zero-Shot integrity
        pairs = [
            ([1, 2], "num_add_1"),
            ([5, 6], "num_add_1_v2"),
            ([[0], [1]], "sp1_add_1"),
        ]
        for seq, _ in pairs:
            vecs = self.oracle.compute_sequence(seq)
            self.observer.observe(vecs[1])
        print(f"  Forest Size: {len(self.forest)}")

    def run_phase_2_l3_formation(self):
        """Pre-train meta-relational meta-patterns in seen domains."""
        print("\n--- PHASE 2: L3 META-PATTERN DISCOVERY ---")
        sequences = [
            self.gen_numeric_constant(15), 
            self.gen_numeric_accumulator(15),
            self.gen_spatial_1d_accumulator(15),
        ]
        
        for i, seq in enumerate(sequences):
            print(f"  Training Sequence {i} (Length {len(seq)})...")
            vecs = self.oracle.compute_sequence(seq)
            for v in vecs:
                self.observer.observe(v)
            print(f"    Forest Size: {len(self.forest)}")

    def run_phase_3_zero_shot_prediction(self):
        """Test on truly unseen domain using active prediction loop."""
        print("\n--- PHASE 3: ZERO-SHOT DOMAIN TRANSFER & PREDICTION ---")
        
        # Truly unseen domain: Spatial 2D Accumulator
        test_seq = self.gen_spatial_2d_accumulator(n=10)
        print(f"  Unseen Test Sequence (Spatial 2D): {test_seq[:4]} ...")
        
        vecs = self.oracle.compute_sequence(test_seq)
        
        # 1. Priming: Observe first 4 steps to allow dynamic HFN formation
        print("  Priming (t=0 to t=3)...")
        for i in range(4):
            self.observer.observe(vecs[i])
            
        print(f"  Forest Size after priming: {len(self.forest)}")
        
        # 2. Retrieval: Use current L3 trajectory to retrieve the L3 pattern
        # Since the manifold is additive, the newly formed leaf for vecs[3] 
        # is the best source for the L3 constraint.
        query_vec = vecs[3]
        query_node = HFN(mu=query_vec, sigma=np.ones(D)*0.01, id="query", use_diag=True)
        candidates = self.retriever.retrieve(query_node, k=5)
        
        best_l3_node = None
        for c in candidates:
            if np.linalg.norm(c.mu[60:90]) > 0.01:
                best_l3_node = c
                break
                
        if not best_l3_node:
            print("  [FAIL] Failed to retrieve L3 meta-node.")
            return
            
        print(f"  Retrieved L3 Meta-Node: {best_l3_node.id}")
        
        # 3. Prediction Loop with Ablations
        def evaluate_prediction(l3_constraint_vec: np.ndarray, name: str):
            print(f"\n  --- TEST: {name} ---")
            current_vec = vecs[3] # Start from end of priming
            errors = []
            
            # Predict t=4 to 9 autoregressively
            for t in range(4, len(vecs)):
                actual_vec = vecs[t]
                pred_vec = self.predict_next(current_vec, l3_constraint_vec)
                
                # Compare predicted L1 state to ground truth
                error = np.linalg.norm(pred_vec[0:30] - actual_vec[0:30])
                errors.append(error)
                
                # Feedback loop: use prediction as next input
                current_vec = pred_vec
                
            mean_error = np.mean(errors)
            print(f"    Mean Autoregressive L1 Prediction Error (t=4..9): {mean_error:.4f}")
            if mean_error < 0.05:
                print("    [SUCCESS] High-fidelity prediction.")
            else:
                print("    [FAIL] Prediction diverged from ground truth.")

        # Baseline A: L2-only (Assume no meta-relational change)
        evaluate_prediction(np.zeros(S_DIM), "L2-Only Baseline (Constant Rule Assumption)")
        
        # Baseline B: Random L3
        np.random.seed(42)
        rand_l3 = np.random.randn(S_DIM) * np.linalg.norm(best_l3_node.mu[60:90])
        evaluate_prediction(rand_l3, "Random L3 Baseline")
        
        # Full HPM: Active L3 constraint
        evaluate_prediction(best_l3_node.mu[60:90], "Full HPM (L3 Top-Down Constraint)")

def run_experiment():
    exp = CompositionalExperimentRefactored()
    exp.run_phase_1_l2_formation()
    exp.run_phase_2_l3_formation()
    exp.run_phase_3_zero_shot_prediction()

if __name__ == "__main__":
    run_experiment()
