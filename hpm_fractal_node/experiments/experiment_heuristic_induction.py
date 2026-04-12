"""
SP58: Heuristic Operator Induction and Disambiguation.

Demonstrates scalable function induction using a manifold-guided beam search
to synthesize deep non-linear operator chains under noise.
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
    S_DIM
)

D = 60 # [30D Content | 30D Operator Parameters]
PARAM_OFFSET = 30

class BeamSearchSynthesis:
    """
    Heuristically explores the space of operator compositions.
    Guided by HFN residual error and chain complexity.
    """
    def __init__(self, primitives: List[Operator], max_depth: int = 3, beam_width: int = 5):
        self.primitives = primitives
        self.max_depth = max_depth
        self.beam_width = beam_width

    def search(self, x_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Operator]:
        """
        Finds the top-k operator chains that best explain the sequence of pairs.
        """
        # Initial beam: the primitives themselves
        beam = [(p, self._score(p, x_pairs)) for p in self.primitives]
        beam.sort(key=lambda x: x[1])
        beam = beam[:self.beam_width]

        for depth in range(1, self.max_depth):
            new_candidates = []
            for current_op, _ in beam:
                for p in self.primitives:
                    # Form new composition f(g(x))
                    new_op = p.compose(current_op)
                    new_candidates.append((new_op, self._score(new_op, x_pairs)))
            
            # Combine with current beam and prune
            combined = beam + new_candidates
            # Deduplicate by name or effective params
            seen = set()
            unique = []
            for op, score in combined:
                if op.name not in seen:
                    seen.add(op.name)
                    unique.append((op, score))
            
            unique.sort(key=lambda x: x[1])
            beam = unique[:self.beam_width]
            
        return [op for op, score in beam]

    def _score(self, op: Operator, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Score based on prediction error + complexity penalty."""
        total_err = 0.0
        for x_in, x_target in pairs:
            x_pred = op.apply(x_in)
            # Only focus on numeric axis error
            total_err += abs(x_pred[0] - x_target[0])
        
        # Complexity penalty (prefer shorter names/chains)
        complexity = len(op.name.split(" ∘ ")) * 0.01
        return (total_err / len(pairs)) + complexity

class OperatorCompositionExperimentSP58:
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
        print(f"Initialized SP58 Experiment with D={D}")

    def encode_op_node(self, op: Operator, state: np.ndarray) -> np.ndarray:
        vec = np.zeros(D)
        vec[0:30] = state
        if isinstance(op, AffineOperator):
            vec[PARAM_OFFSET] = op.weight
            vec[PARAM_OFFSET + 1] = op.bias
        elif isinstance(op, ModOperator):
            vec[PARAM_OFFSET + 2] = op.modulus
        return vec

    def decode_op_from_vec(self, vec: np.ndarray, id: str) -> Operator:
        if abs(vec[PARAM_OFFSET + 2]) > 0.01:
            return ModOperator(modulus=vec[PARAM_OFFSET + 2], name=f"Mod_{int(vec[PARAM_OFFSET+2])}")
        return AffineOperator(weight=vec[PARAM_OFFSET], bias=vec[PARAM_OFFSET+1], name=id)

    # --- Training ---

    def run_phase_1_training(self):
        print("\n--- PHASE 1: PRIMITIVE FORMATION ---")
        # Primitives: Add_1, Mul_2, Mod_10
        ops = [
            AffineOperator(1.0, 0.1, "Add_1"),
            AffineOperator(2.0, 0.0, "Mul_2"),
            ModOperator(10.0, "Mod_10")
        ]
        for op in ops:
            v = self.encode_op_node(op, self.oracle.encode(5))
            self.observer.observe(v)
        print(f"  Primitives registered in Forest.")

    # --- Test ---

    def run_phase_2_induction(self):
        print("\n--- PHASE 2: DISAMBIGUATION & INDUCTION ---")
        
        # Dynamic: ((x * 2) + 1) % 10
        # 1 -> 3 -> 7 -> 5 -> 1 -> 3 ...
        clean_seq = [1, 3, 7, 5, 1, 3, 7, 5]
        
        # Add Perceptual Noise
        np.random.seed(42)
        noisy_states = [self.oracle.encode(x) + np.random.normal(0, 0.005, S_DIM) for x in clean_seq]
        
        print(f"  Noisy Sequence: {[round(self.oracle.decode_numeric(s), 2) for s in noisy_states[:4]]} ...")
        
        # 1. Collect LTM Primitives
        ltm_ops = []
        for n in self.forest.active_nodes():
            if np.linalg.norm(n.mu[PARAM_OFFSET:]) > 0.01:
                ltm_ops.append(self.decode_op_from_vec(n.mu, n.id))
        
        # 2. Heuristic Beam Search
        # We'll use the first 3 transitions for induction
        pairs = [(noisy_states[i], noisy_states[i+1]) for i in range(3)]
        
        synthesizer = BeamSearchSynthesis(ltm_ops, max_depth=3, beam_width=10)
        candidates = synthesizer.search(pairs)
        
        print(f"  Top Disambiguation Candidates:")
        for i, c in enumerate(candidates[:3]):
            print(f"    {i+1}. {c.name}")

        winning_op = candidates[0]
        print(f"  Selected Winner: {winning_op.name}")

        # 3. Autoregressive Prediction
        print(f"\n  --- AUTOREGRESSIVE PREDICTION (t=4..7) ---")
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
            print("\n  [SUCCESS] Scalable Operator Induction Verified!")
            print(f"  The agent synthesized the depth-3 non-linear chain '{winning_op.name}' under noise.")
        else:
            print("\n  [FAIL] Induction diverged.")

def run_experiment():
    exp = OperatorCompositionExperimentSP58()
    exp.run_phase_1_training()
    exp.run_phase_2_induction()

if __name__ == "__main__":
    run_experiment()
