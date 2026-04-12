"""
SP57: Operator-Level Compositional Abstraction.

Demonstrates synthesizing non-linear dynamics (x -> 2x + 1) by composing
primitive functional operators (Add, Mul).
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Any, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GeometricRetriever
from hpm_fractal_node.code.sp57_operators import OperatorOracle, Operator, S_DIM

D = 60 # [30D Content | 30D Operator Parameters]
PARAM_OFFSET = 30

class OperatorCompositionExperiment:
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
        print(f"Initialized SP57 Experiment with D={D} [Operator Manifold]")

    def encode_op_node(self, op: Operator, state: np.ndarray) -> np.ndarray:
        """Encode an operator acting on a state into a 60D vector."""
        vec = np.zeros(D)
        vec[0:30] = state
        # Parameterize Op: [weight on axis 0 | bias on axis 0 | ...]
        vec[PARAM_OFFSET] = op.weight
        vec[PARAM_OFFSET + 1] = op.bias
        return vec

    def decode_op_from_vec(self, vec: np.ndarray, name: str = "decoded") -> Operator:
        return Operator(weight=vec[PARAM_OFFSET], bias=vec[PARAM_OFFSET+1], name=name)

    # --- Training Phases ---

    def run_phase_1_primitives(self):
        print("\n--- PHASE 1: OPERATOR PRIMITIVE FORMATION ---")
        # 1. Add_1 Primitive
        op_add = Operator(weight=1.0, bias=0.1, name="Add_1") # 0.1 in manifold = 1.0 numeric
        s1 = self.oracle.encode(10)
        v1 = self.encode_op_node(op_add, s1)
        self.observer.observe(v1)
        
        # 2. Mul_2 Primitive
        op_mul = Operator(weight=2.0, bias=0.0, name="Mul_2")
        s2 = self.oracle.encode(5)
        v2 = self.encode_op_node(op_mul, s2)
        self.observer.observe(v2)
        
        print(f"  Forest Size: {len(self.forest)}")

    def run_phase_2_stabilization(self):
        print("\n--- PHASE 2: OPERATOR STABILIZATION ---")
        # Constant Add sequence
        seq_add = [1, 2, 3, 4, 5]
        op = Operator(weight=1.0, bias=0.1, name="Add_1")
        for x in seq_add:
            v = self.encode_op_node(op, self.oracle.encode(x))
            self.observer.observe(v)
            
        # Constant Mul sequence
        seq_mul = [1, 2, 4, 8, 16]
        op = Operator(weight=2.0, bias=0.0, name="Mul_2")
        for x in seq_mul:
            v = self.encode_op_node(op, self.oracle.encode(x))
            self.observer.observe(v)
            
        print(f"  Forest Size: {len(self.forest)}")

    def run_phase_3_generative_composition(self):
        print("\n--- PHASE 3: GENERATIVE OPERATOR COMPOSITION ---")
        
        # Test Sequence: x -> 2x + 1
        # 1 -> 3 -> 7 -> 15 -> 31
        test_data = [1, 3, 7, 15, 31, 63, 127, 255]
        print(f"  Test Sequence: {test_data[:4]} ...")
        
        # 1. Infer Operators from history
        # We'll use the first two transitions to infer noisy local operators
        s0 = self.oracle.encode(test_data[0])
        s1 = self.oracle.encode(test_data[1])
        s2 = self.oracle.encode(test_data[2])
        
        # Local deltas
        d1 = s1[0] - s0[0]
        d2 = s2[0] - s1[0]
        
        print(f"  Inferred Local Deltas: {d1*10:.1f}, {d2*10:.1f}")
        
        # 2. Generative Search for Operator Chain
        # LTM contains Add_1 and Mul_2
        ltm_ops = []
        for n in self.forest.active_nodes():
            if np.linalg.norm(n.mu[PARAM_OFFSET:]) > 0.01:
                ltm_ops.append(self.decode_op_from_vec(n.mu, name=n.id))
        
        # Deduplicate LTM ops by params
        unique_ltm = []
        seen = set()
        for op in ltm_ops:
            key = (round(op.weight, 2), round(op.bias, 2))
            if key not in seen:
                seen.add(key)
                unique_ltm.append(op)
        
        print(f"  LTM Primitives: {[op.name for op in unique_ltm]}")
        
        # Combinatorial Search (Chain length 1 and 2)
        best_chain = None
        min_error = float('inf')
        
        chains = []
        # Length 1
        for op in unique_ltm: chains.append([op])
        # Length 2
        for op1 in unique_ltm:
            for op2 in unique_ltm:
                chains.append([op1, op2])
                
        for chain in chains:
            # Compose: f(g(x))
            composed = chain[0]
            desc = chain[0].name
            if len(chain) > 1:
                composed = chain[0].compose(chain[1])
                desc = f"{chain[0].name} ∘ {chain[1].name}"
            
            # Check invariance on Axis 0
            pred_s1 = composed.apply(s0)
            err1 = abs(pred_s1[0] - s1[0])
            
            pred_s2 = composed.apply(s1)
            err2 = abs(pred_s2[0] - s2[0])
            
            error = (err1 + err2) / 2.0
            print(f"    [SEARCH] Testing {desc:20} | Result: {composed} | Error: {error:.4f}")
            
            if error < min_error:
                min_error = error
                best_chain = composed
                
        print(f"  Selected Composed Operator: {best_chain}")
        
        # 3. Prediction Loop
        def evaluate(op: Operator, name: str):
            print(f"\n  --- TEST: {name} ---")
            curr_val = test_data[3]
            curr_s = self.oracle.encode(curr_val)
            errors = []
            for t in range(4, len(test_data)):
                actual = test_data[t]
                pred_s = op.apply(curr_s)
                pred_val = self.oracle.decode_numeric(pred_s)
                
                error = abs(pred_val - actual)
                errors.append(error)
                curr_s = pred_s
            
            mean_err = np.mean(errors)
            print(f"    Mean Prediction Error (t=4..7): {mean_err:.4f}")
            return mean_err

        # Baseline: Add_1 only
        evaluate(Operator(1.0, 0.1, "Add_1"), "Baseline: Constant Addition")
        # Baseline: Mul_2 only
        evaluate(Operator(2.0, 0.0, "Mul_2"), "Baseline: Constant Multiplication")
        # Full HPM
        err_hpm = evaluate(best_chain, "SP57: Composed Operator x -> 2x+1")
        
        if err_hpm < 0.1:
            print("\n  [SUCCESS] Operator Composition Verified!")
            print("  The agent synthesized the non-linear rule by chaining Add and Mul primitives.")
        else:
            print("\n  [FAIL] Synthesis failed to reach near-zero error.")

def run_experiment():
    exp = OperatorCompositionExperiment()
    exp.run_phase_1_primitives()
    exp.run_phase_2_stabilization()
    exp.run_phase_3_generative_composition()

if __name__ == "__main__":
    run_experiment()
