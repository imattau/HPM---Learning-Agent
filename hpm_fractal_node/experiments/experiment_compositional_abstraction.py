"""
SP56: Compositional Abstraction and Meta-Relational Transfer.

Tests the HPM Framework's Hierarchical Pattern Stack:
1. Form L2 (Relation) from L1 (Content).
2. Form L3 (Meta-Relation) from L2 sequence.
3. Zero-shot transfer L3 pattern to novel L1 domain.
"""

import numpy as np
import os
import sys
import copy
from pathlib import Path
from typing import List, Any, Optional, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GeometricRetriever
from hfn.evaluator import Evaluator
from hpm_fractal_node.code.sp56_oracle import StatefulOracleSP56, S_DIM, D

class CompositionalExperiment:
    def __init__(self):
        self.forest = Forest(D=D)
        self.oracle = StatefulOracleSP56()
        self.retriever = GeometricRetriever(self.forest)
        self.evaluator = Evaluator()
        
        # We set a low residual surprise threshold to trigger L3 node formation
        # when L2 sequences become complex (variant).
        self.observer = Observer(
            forest=self.forest,
            retriever=self.retriever,
            tau=0.1,
            residual_surprise_threshold=0.5, 
            node_use_diag=True
        )
        print(f"Initialized SP56 Experiment with D={D} [3-Level Stack]")

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
        # 1D movement: [0,0,1,0] -> [0,0,0,1]
        def to_vec(pos):
            v = np.zeros(5)
            v[pos % 5] = 1.0
            return tuple(v)
        return [to_vec(start + i * step) for i in range(n)]

    def gen_boolean_alternating(self, n=10, start=True):
        seq = [start]
        for i in range(n-1):
            seq.append(not seq[-1])
        return seq

    # --- Training Phases ---

    def run_phase_1_l2_formation(self):
        """Build a dictionary of basic relational rules at Level 2."""
        print("\n--- PHASE 1: L2 RELATION FORMATION ---")
        # Just simple pairs to register the deltas
        pairs = [
            ([1, 2], "add_1"),
            ([10, 11], "add_1_v2"),
            ([5, 4], "sub_1"),
            ([True, False], "negate"),
        ]
        for seq, label in pairs:
            vecs = self.oracle.compute_sequence(seq)
            # Only observe the transition
            self.observer.observe(vecs[1])
        print(f"  Forest Size: {len(self.forest)}")

    def run_phase_2_l3_formation(self):
        """Build meta-relational nodes from prolonged sequences."""
        print("\n--- PHASE 2: L3 META-PATTERN DISCOVERY ---")
        sequences = [
            self.gen_numeric_constant(15, start=0, step=1),   # Constant
            self.gen_numeric_alternating(15, start=0, step_a=5, step_b=-5), # Oscillator
            self.gen_spatial_constant(15, start=0, step=1),   # Constant (Spatial)
        ]
        
        for i, seq in enumerate(sequences):
            print(f"  Training Sequence {i} (Length {len(seq)})...")
            vecs = self.oracle.compute_sequence(seq)
            for v in vecs:
                self.observer.observe(v)
            print(f"    Forest Size: {len(self.forest)}")

    def run_phase_3_zero_shot(self):
        """Test on novel domain (Boolean) using L3 Top-Down Constraint."""
        print("\n--- PHASE 3: ZERO-SHOT TRANSFER (Boolean) ---")
        
        # Unseen domain: Boolean oscillation
        # True -> False -> True -> False ...
        # L1: Boolean
        # L2: Negation
        # L3: Oscillation (Abstract Pattern from Phase 2)
        
        test_seq = self.gen_boolean_alternating(n=10)
        print(f"  Test Sequence: {test_seq[:4]} ...")
        
        vecs = self.oracle.compute_sequence(test_seq)
        
        # Present first 4 steps to 'prime' the manifolds
        print("  Priming...")
        for i in range(4):
            self.observer.observe(vecs[i])
            
        # Retrieval Test:
        # At t=4, does the retriever identify the L3 Oscillator node?
        # Target: Delta at L3 [60:90]
        query_vec = vecs[4]
        # Mask L1 and L2 to find the L3 meta-pattern
        query_mu = np.zeros(D)
        query_mu[60:90] = query_vec[60:90]
        
        query_node = HFN(mu=query_mu, sigma=np.ones(D)*0.1, id="meta_query", use_diag=True)
        candidates = self.retriever.retrieve(query_node, k=5)
        
        print(f"  Current Forest Size: {len(self.forest)}")
        print("  Top-Down Candidates (based on L3 Meta-Pattern):")
        for c in candidates:
            # Check if this node has energy in L3 slice
            l3_energy = np.linalg.norm(c.mu[60:90])
            l2_energy = np.linalg.norm(c.mu[30:60])
            print(f"    - {c.id:20} | L3 Energy: {l3_energy:.2f} | L2 Energy: {l2_energy:.2f}")

        # Prediction Accuracy (Simulation)
        # In a full synthesis loop, L3 would pin L2. 
        # Here we verify the 'matching' probability.
        best_match = candidates[0]
        is_l3 = np.linalg.norm(best_match.mu[60:90]) > 0.5
        
        if is_l3:
            print("  [SUCCESS] Level 3 Meta-Pattern Recognized!")
        else:
            print("  [FAIL] Failed to identify L3 pattern in novel domain.")

    def analyze_manifold(self):
        print("\n--- MANIFOLD ANALYSIS ---")
        # Group nodes by their dominant slice energy
        l1_nodes, l2_nodes, l3_nodes = [], [], []
        for n in self.forest.active_nodes():
            e1 = np.linalg.norm(n.mu[0:30])
            e2 = np.linalg.norm(n.mu[30:60])
            e3 = np.linalg.norm(n.mu[60:90])
            
            if e3 > e2 and e3 > e1: l3_nodes.append(n.id)
            elif e2 > e1: l2_nodes.append(n.id)
            else: l1_nodes.append(n.id)
            
        print(f"  L1 (Content) Nodes: {len(l1_nodes)}")
        print(f"  L2 (Relational) Nodes: {len(l2_nodes)}")
        print(f"  L3 (Meta) Nodes: {len(l3_nodes)}")
        if l3_nodes:
            print(f"  Discovered Meta-Patterns: {l3_nodes[:5]}")

def run_experiment():
    exp = CompositionalExperiment()
    exp.run_phase_1_l2_formation()
    exp.run_phase_2_l3_formation()
    exp.analyze_manifold()
    exp.run_phase_3_zero_shot()

if __name__ == "__main__":
    run_experiment()
