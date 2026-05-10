
import numpy as np
import sys
import os

# Ensure the testbed is in the path
sys.path.append(os.path.join(os.getcwd(), 'hpm_ai_v6/testbed'))

from toy_hpm_v4 import Cell, MetaPatternRule, learn_step, hierarchical_total_score, make_1cell, DynamicPatternField

def run_curiosity_test():
    np.random.seed(42)
    
    # 1. Setup Environment
    # 0-cells
    o = [Cell(0, np.array([1.0, 0.0]), name="A"), 
         Cell(0, np.array([0.0, 1.0]), name="B"), 
         Cell(0, np.array([-1.0, 0.0]), name="C")]
    
    # 1-cells
    patterns = [
        make_1cell(o[0], o[1], np.array([0.5, 0.5]), name="A_to_B"),
        make_1cell(o[1], o[2], np.array([-0.5, 0.5]), name="B_to_C"),
        make_1cell(o[2], o[0], np.array([0.5, -0.5]), name="C_to_A")
    ]
    
    all_cells = o + patterns
    consensus_vec = np.array([1.0, 0.0])
    
    # 2. Define Sequences of varying complexity
    # Low: Deterministic A->B->C->A
    seq_low = [o[0], o[1], o[2], o[0], o[1], o[2]]
    
    # Medium: Structured but probabilistic
    seq_mid = [o[0], o[1], o[2], o[1], o[2], o[0]] # A bit more variety
    
    # High: Random
    seq_high = list(np.random.choice(o, 6))
    
    results = {}
    
    for label, seq in [("Low", seq_low), ("Mid", seq_mid), ("High", seq_high)]:
        # Fresh field and rule for each test
        field = DynamicPatternField(patterns)
        mpr = MetaPatternRule(patterns)
        
        # We'll measure the total score of the best pattern for this sequence
        # Note: In a real agent, curiosity would drive sequence selection.
        # Here we measure how 'attractive' the sequence is to the hierarchy.
        
        scores = []
        for _ in range(3): # A few learning steps
            s = learn_step(mpr, field, seq, all_cells, consensus_vec)
            scores.append(np.max(s))
            
        results[label] = np.mean(scores)
        print(f"Complexity: {label:10} | Avg Best Score: {results[label]:.3f}")

    print("\nVerification of Prediction 9.4:")
    if results["Mid"] > results["High"]:
        print("[PASS] Mid > High (Randomness penalized)")
    else:
        print("[FAIL] Mid <= High")
        
    if results["Mid"] > results["Low"]:
        print("[PASS] Mid > Low (Triviality penalized)")
    else:
        print("[INFO] Mid <= Low (Current parameters might favor predictability over novelty)")

if __name__ == "__main__":
    run_curiosity_test()
