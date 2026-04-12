"""
SP55: Experiment 45 (Redesign) — HPM-Native Library Discovery

Focus: Emergent discovery via utility competition, probabilistic retrieval,
and compositional task solving.
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
from hpm_fractal_node.code.library_query import (
    BehavioralOracle, 
    LibraryScannerQuery, 
    LibraryProbingConverter,
    S_DIM
)

# Configuration
D = 60 # [25D Pre-state | 10D Concept | 25D Delta]
CONCEPT_OFFSET = 25
DELTA_OFFSET = 35

class HPMDiscoveryExperiment:
    def __init__(self):
        self.forest = Forest(D=D)
        self.oracle = BehavioralOracle()
        self.retriever = GeometricRetriever(self.forest)
        
        # Setup Discovery Stack
        lib_path = str(Path(__file__).parent.parent / "code" / "mock_tool_lib.py")
        self.query = LibraryScannerQuery("mock_tool_lib", lib_path)
        self.converter = LibraryProbingConverter(self.oracle)
        
        self.observer = Observer(
            forest=self.forest,
            retriever=self.retriever,
            query=self.query,
            converter=self.converter,
            gap_query_threshold=0.6,
            residual_surprise_threshold=2.0, # Default threshold allows local learning
            node_use_diag=True
        )
        
        self.evaluator = Evaluator()
        
        # Inject structural priors (e.g. Identity transform)
        # These will compete with library discovery
        prior = HFN(mu=np.zeros(D), sigma=np.ones(D)*0.5, id="prior_identity", use_diag=True)
        # Identity means Delta = 0
        self.observer.register(prior, protected=True)
        
        print(f"Initialized HPM-Native Discovery Experiment with D={D}")

    def run_phase_1_discovery(self):
        """
        Triggers discovery through iterative task presentation.
        Discovery should emerge when 'prior_identity' fails to explain the signals.
        """
        print("\n--- PHASE 1: EMERGENT DISCOVERY ---")
        
        # Present tasks that cannot be explained by Identity
        tasks = [
            ([1, 2, 3], [(1, 2), (1, 3), (2, 3)]), # Pairing
            ([1, 1, 2], [1, 2]), # Uniquify
        ]
        
        for inp, out in tasks:
            inp_s = self.oracle.compute(inp)
            out_s = self.oracle.compute(out)
            observed_delta = out_s - inp_s
            
            x = np.zeros(D)
            x[:S_DIM] = inp_s
            x[DELTA_OFFSET:] = observed_delta
            
            print(f"Presenting signal: {inp} -> {out}")
            res = self.observer.observe(x)
            print(f"  Residual Surprise: {res.residual_surprise:.2f} | Nodes: {len(self.forest)}")

        print(f"Final Forest Size: {len(self.forest)}")

    def solve_probabilistic(self, input_data: Any, target_output: Any) -> Optional[HFN]:
        """Finds the best tool using purely probabilistic scoring."""
        inp_s = self.oracle.compute(input_data)
        out_s = self.oracle.compute(target_output)
        req_delta = out_s - inp_s
        
        goal_vec = np.zeros(D)
        goal_vec[:S_DIM] = inp_s
        goal_vec[DELTA_OFFSET:] = req_delta
        
        query_node = HFN(mu=goal_vec, sigma=np.ones(D)*0.1, id="task_query", use_diag=True)
        candidates = self.retriever.retrieve(query_node, k=20)
        
        if not candidates:
            return None
            
        # Rank by log_prob (Probabilistic Selection)
        scored = [(n, n.log_prob(goal_vec)) for n in candidates if n.id != "prior_identity"]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[0][0] if scored else None

    def run_phase_2_validation(self):
        print("\n--- PHASE 2: PROBABILISTIC RECOGNITION ---")
        
        test_cases = [
            ("Pairing", [10, 20, 30], [(10, 20), (10, 30), (20, 30)]),
            ("Uniquify", [5, 5, 6, 7], [5, 6, 7]),
            ("Inversion", [[1, 2], [3, 4]], [[4, 3], [2, 1]])
        ]
        
        for name, inp, out in test_cases:
            tool = self.solve_probabilistic(inp, out)
            if tool:
                print(f"Task: {name:10} | Selected: {tool.id:12} | LogProb: {tool.log_prob(np.zeros(D)):.2f}")
            else:
                print(f"Task: {name:10} | Selected: NONE")

    def run_phase_3_composition(self):
        """
        Tests if the agent can solve a multi-step task by combining deltas.
        Task: Uniquify then Invert.
        """
        print("\n--- PHASE 3: COMPOSITIONAL REASONING ---")
        
        # [1, 1, [2, 3], [4, 5]] -> Uniquify -> [1, [2, 3], [4, 5]] -> Invert -> [[5, 4], [3, 2], 1]
        inp = [1, 1, [2, 3], [4, 5]]
        target = [[5, 4], [3, 2], 1]
        
        print(f"Task: Uniquify + Invert")
        print(f"  Input: {inp}")
        print(f"  Goal:  {target}")
        
        # 1. Solve iteratively
        # In a full HierarchicalOrchestrator, this would be a search. 
        # Here we verify the 'path' exists in the manifold.
        
        # Step 1: Find tool for Inp -> Mid
        mid_data = [1, [2, 3], [4, 5]] # Hand-coded intermediate for verification
        tool_1 = self.solve_probabilistic(inp, mid_data)
        
        # Step 2: Find tool for Mid -> Goal
        tool_2 = self.solve_probabilistic(mid_data, target)
        
        if tool_1 and tool_2:
            print(f"  [PATH FOUND]")
            print(f"  Step 1: {tool_1.id}")
            print(f"  Step 2: {tool_2.id}")
            
            # Verify they are actually different behavioral tools
            if tool_1.id != tool_2.id:
                print("  [SUCCESS] Compositional chain verified.")
            else:
                print("  [WARNING] Chain used the same tool twice - check manifold separation.")
        else:
            print("  [FAIL] Could not find compositional path.")

def run_experiment():
    exp = HPMDiscoveryExperiment()
    exp.run_phase_1_discovery()
    exp.run_phase_2_validation()
    exp.run_phase_3_composition()

if __name__ == "__main__":
    run_experiment()
