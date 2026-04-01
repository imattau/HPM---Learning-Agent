"""
Experiment: Agnostic Decoder (SP22).
Tests 1D block-stacking via Variance Collapse.
"""
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.decoder import Decoder

def run_experiment():
    print("SP22: Agnostic Decoder Experiment (1D Block Stacking)\n")
    
    # --- 1. Manifolds ---
    # Hand: Specific locations (Sigma=0)
    hand = Forest(D=1, forest_id="hand")
    x1 = HFN(mu=np.array([1.0]), sigma=np.array([0.0001]), id="pos_1.0", use_diag=True)
    x2 = HFN(mu=np.array([2.0]), sigma=np.array([0.0001]), id="pos_2.0", use_diag=True)
    x5 = HFN(mu=np.array([5.0]), sigma=np.array([0.0001]), id="pos_5.0", use_diag=True)
    for p in [x1, x2, x5]: hand.register(p)

    # Add topological constraints
    color_red = HFN(mu=np.array([0.0]), sigma=np.array([0.0]), id="COLOR_RED")
    x2.add_edge(x2, color_red, "HAS_COLOR") # x2 is where red usually is

    # --- 2. Decoder Instance ---
    decoder = Decoder(target_forest=hand, sigma_threshold=0.01)

    # --- 3. Test Cases ---
    
    # Test 1: Explicit Expansion (A sequence goal)
    print("Test 1: Explicit Expansion (Sequence Goal)")
    goal_seq = HFN(mu=np.array([0.0]), sigma=np.array([1.0]), id="goal_seq")
    goal_seq.add_child(x1)
    goal_seq.add_child(x5)
    
    output1 = decoder.decode(goal_seq)
    print(f"  Goal: [pos_1.0, pos_5.0]")
    print(f"  Result: {[n.id for n in output1]}\n")

    # Test 2: Implicit Resolution (Searching by constraint)
    print("Test 2: Implicit Resolution (Search by Edge Constraint)")
    # Goal: An abstract 'RED' thing at X=2.1. No children.
    goal_find = HFN(mu=np.array([2.1]), sigma=np.array([0.5]), id="goal_find")
    goal_find.add_edge(goal_find, color_red, "HAS_COLOR")
    
    output2 = decoder.decode(goal_find)
    print(f"  Goal: Find RED near X=2.1")
    print(f"  Result: {[n.id for n in output2]} (Expect: pos_2.0)\n")

    # Test 3: Constraint Rejection
    print("Test 3: Constraint Rejection")
    # Goal: Find RED near X=5.0 (where blue is). Should reject blue and find red if in range.
    # Note: Target manifold retrieve will return nearest nodes.
    goal_wrong = HFN(mu=np.array([4.9]), sigma=np.array([0.5]), id="goal_wrong")
    goal_wrong.add_edge(goal_wrong, color_red, "HAS_COLOR")
    
    output3 = decoder.decode(goal_wrong)
    print(f"  Goal: Find RED near X=4.9")
    # It should pick pos_2.0 because it has the COLOR_RED edge, even though pos_5.0 is closer.
    print(f"  Result: {[n.id for n in output3]} (Expect: pos_2.0)\n")

if __name__ == "__main__":
    run_experiment()
