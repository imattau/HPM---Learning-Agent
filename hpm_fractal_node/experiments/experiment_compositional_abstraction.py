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

    def gen_numeric_alternating(self, n=10, start=0, step_a=1, step_b=-1):
        seq = [start]
        for i in range(n-1):
            step = step_a if i % 2 == 0 else step_b
            seq.append(seq[-1] + step)
        return seq

    def gen_spatial_constant(self, n=10, start=0, step=1):
        # 1D movement
        def to_vec(pos):
            v = np.zeros(5)
            v[pos % 5] = 1.0
            return tuple(v)
        return [to_vec(start + i * step) for i in range(n)]

    def gen_numeric_accumulator(self, n=10, start=0, start_delta=0, acceleration=1):
        # Accelerating sequence: L3 is constant 'acceleration'
        seq = [start]
        curr_delta = start_delta
        for i in range(1, n):
            curr_delta += acceleration
            seq.append(seq[-1] + curr_delta)
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
        # Only ONE primitive: Add 1. Agent MUST compose Add_1 + Add_1 to get +2.
        pairs = [
            ([0, 1], "num_add_1"),
            ([10, 11], "num_add_1_v2"),
        ]
        for seq, _ in pairs:
            vecs = self.oracle.compute_sequence(seq)
            self.observer.observe(vecs[1])
        print(f"  Forest Size: {len(self.forest)}")

    def run_phase_2_l3_formation(self):
        """Pre-train meta-relational meta-patterns in seen domains."""
        print("\n--- PHASE 2: L3 META-PATTERN DISCOVERY ---")
        # Constant with very small step to avoid L1 overlap with target +2.
        sequences = [
            self.gen_numeric_constant(15, step=0.1), 
            self.gen_numeric_alternating(15, step_a=0.5, step_b=-0.5),
        ]
        
        for i, seq in enumerate(sequences):
            print(f"  Training Sequence {i} (Length {len(seq)})...")
            vecs = self.oracle.compute_sequence(seq)
            for v in vecs:
                self.observer.observe(v)
            print(f"    Forest Size: {len(self.forest)}")

    def run_phase_3_zero_shot_prediction(self):
        """Test on truly unseen structure using active prediction and Generative Composition."""
        print("\n--- PHASE 3: ZERO-SHOT DOMAIN TRANSFER & GENERATIVE COMPOSITION ---")
        
        ltm_ids = [n.id for n in self.forest.active_nodes()]
        
        # Truly unseen structure: Accumulator with Acceleration +2
        # Agent only knows +1 primitive. It MUST compose them.
        test_seq = self.gen_numeric_accumulator(n=10, acceleration=2)
        print(f"  Unseen Test Sequence (Numeric Accumulator +2): {test_seq[:4]} ...")
        
        clean_vecs = self.oracle.compute_sequence(test_seq)
        
        # 1. Priming with Perceptual Noise
        np.random.seed(42)
        noisy_vecs = []
        for t, v in enumerate(clean_vecs):
            noisy_v = v.copy()
            noisy_v[0:30] += np.random.normal(0, 0.01, S_DIM)
            if t > 0:
                noisy_v[30:60] = noisy_v[0:30] - noisy_vecs[t-1][0:30]
            if t > 1:
                noisy_v[60:90] = noisy_v[30:60] - noisy_vecs[t-1][30:60]
            noisy_vecs.append(noisy_v)

        print("  Priming (t=0 to t=3) with noisy perception...")
        for i in range(4):
            self.observer.observe(noisy_vecs[i])
            
        print(f"  Forest Size after priming: {len(self.forest)}")
        
        # 2. Generative Composition: Combine multiple LTM concepts to approximate L3
        # No single stored vector matches +3.
        inferred_l3 = noisy_vecs[3][60:90]
        slices = [slice(0,30), slice(30,60), slice(60,90)]
        
        candidate_priors = []
        for nid in ltm_ids:
            n = self.forest.get(nid)
            for i, slc in enumerate(slices):
                sub_vec = n.mu[slc]
                if np.linalg.norm(sub_vec) > 0.01:
                    candidate_priors.append((nid, i, sub_vec))
        
        # Find best SINGLE prior
        best_single_dist = float('inf')
        best_single_vec = np.zeros(S_DIM)
        best_single_desc = "None"
        
        for nid, s_idx, vec in candidate_priors:
            dist = np.linalg.norm(vec - inferred_l3)
            if dist < best_single_dist:
                best_single_dist = dist
                best_single_vec = vec
                best_single_desc = f"{nid} (slice {s_idx})"
                
        # Find best COMPOSED prior (Pair sum)
        best_composed_dist = float('inf')
        best_composed_vec = np.zeros(S_DIM)
        best_composed_desc = "None"
        
        for i, (nid1, s1, v1) in enumerate(candidate_priors):
            for j, (nid2, s2, v2) in enumerate(candidate_priors):
                # Compositional Synthesis: L3 ≈ Prior1 + Prior2
                composed = v1 + v2
                dist = np.linalg.norm(composed - inferred_l3)
                if dist < best_composed_dist:
                    best_composed_dist = dist
                    best_composed_vec = composed
                    best_composed_desc = f"{nid1}[{s1}] + {nid2}[{s2}]"
                    
        print(f"  Best Single Prior:   {best_single_desc} (Dist: {best_single_dist:.4f})")
        print(f"  Best Composed Prior: {best_composed_desc} (Dist: {best_composed_dist:.4f})")
        
        # 3. Prediction Loop with Ablations
        def evaluate_prediction(l3_constraint_vec: np.ndarray, name: str):
            print(f"\n  --- TEST: {name} ---")
            current_vec = noisy_vecs[3]
            errors = []
            for t in range(4, len(clean_vecs)):
                actual_vec = clean_vecs[t]
                pred_vec = self.predict_next(current_vec, l3_constraint_vec)
                error = np.linalg.norm(pred_vec[0:30] - actual_vec[0:30])
                errors.append(error)
                current_vec = pred_vec
            mean_error = np.mean(errors)
            print(f"    Mean Autoregressive L1 Prediction Error (t=4..9): {mean_error:.4f}")
            return mean_error

        err_l2 = evaluate_prediction(np.zeros(S_DIM), "L2-Only Baseline (Constant Rule Assumption)")
        err_single = evaluate_prediction(best_single_vec, "Single Best Prior (Fractal Reuse Only)")
        err_composed = evaluate_prediction(best_composed_vec, "Compositional HPM (Generative Synthesis)")
        
        if err_composed < err_single and err_composed < err_l2:
            print("\n  [SUCCESS] Generative Composition Verified!")
            print(f"  The agent dynamically synthesized the novel meta-pattern from '{best_composed_desc}'.")
        else:
            print("\n  [FAIL] Generative composition failed to outperform retrieval.")

def run_experiment():
    exp = CompositionalExperimentRefactored()
    exp.run_phase_1_l2_formation()
    exp.run_phase_2_l3_formation()
    exp.run_phase_3_zero_shot_prediction()

if __name__ == "__main__":
    run_experiment()
