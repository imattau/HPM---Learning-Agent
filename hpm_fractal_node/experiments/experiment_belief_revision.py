"""
SP37: Experiment 13 — Competing Hypotheses (Belief Revision)

Tests if the HFN system can revise its beliefs when faced with shifting evidence.
Setup:
1. Phase 1 (Ambiguity): Observations support Rule A OR Rule B equally.
2. Phase 2 (Disambiguation): New evidence falsifies Rule A, supports only Rule B.
3. Observe: Does the system shift its dominance from A to B?

Metrics:
- Weight Trajectory: Weight of Rule A vs Rule B over time.
- Falsification Speed: Epochs to penalize the incorrect rule.
- Survival: Does Rule A get pruned or just suppressed?
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.evaluator import Evaluator

def generate_competing_curriculum(dim=10):
    """
    Phase 1: Both 'Add 2' and 'Set to 5' work.
    Phase 2: Only 'Set to 5' works.
    """
    p1 = []
    # Both work if input is 3
    for _ in range(10):
        vec = np.zeros(dim)
        vec[0] = 3.0 # Input
        vec[dim//2] = 2.0 # Delta (Rule A: Add 2, Rule B: Set to 5)
        p1.append(vec)
        
    p2 = []
    # Only Rule B works if input is 8
    for _ in range(10):
        vec = np.zeros(dim)
        vec[0] = 8.0 # Input
        # Target remains 5. 
        # Rule A (Add 2) would predict Delta=2 (Input+2=10)
        # Rule B (Set to 5) predicts Delta=-3 (Input-3=5)
        vec[dim//2] = -3.0 # The actual observed delta
        p2.append(vec)
        
    return p1, p2

def run_experiment():
    print("--- SP37: Experiment 13 — Competing Hypotheses (Belief Revision) ---\n")
    
    DIM = 10
    forest = Forest()
    observer = Observer(
        forest=forest,
        tau=1.0,
        residual_surprise_threshold=1.5,
        alpha_gain=0.1,
        beta_loss=0.1, # Symmetrical gain/loss
        node_use_diag=True
    )
    
    # Pre-seed the two hypotheses to track them easily
    # Rule A: "Always Add 2" (Vague context, precise delta)
    mu_a = np.zeros(DIM); mu_a[DIM//2] = 2.0
    sig_a = np.ones(DIM)*5.0; sig_a[DIM//2] = 0.01
    rule_a = HFN(mu=mu_a, sigma=sig_a, id="rule_add_2", use_diag=True)
    forest.register(rule_a)
    
    # Rule B: "Always Set to 5" (This is harder to represent agnostically, 
    # so we'll let it emerge or use a placeholder that matches the Phase 2 delta)
    # For this experiment, we'll let Rule B be learned naturally in Phase 2.
    
    p1, p2 = generate_competing_curriculum(DIM)
    
    stats = []
    print(f"{'Step':<6} | {'Rule A Weight':<15} | {'Rule B Weight':<15} | {'Active Nodes'}")
    print("-" * 65)
    
    # 1. PHASE 1: Ambiguity
    for i, x in enumerate(p1):
        observer.observe(x)
        w_a = observer.get_weight("rule_add_2")
        
        # Look for any other rule that might be Rule B
        w_b = 0.0
        other_nodes = [n for n in forest.active_nodes() if n.id != "rule_add_2"]
        if other_nodes:
            w_b = max([observer.get_weight(n.id) for n in other_nodes])
            
        stats.append({"a": w_a, "b": w_b})
        if i % 2 == 0:
            print(f"{i+1:<6} | {w_a:<15.4f} | {w_b:<15.4f} | {len(list(forest.active_nodes()))}")

    print("\n--- SHIFT TO PHASE 2 (Disambiguation) ---")
    
    # 2. PHASE 2: Falsification
    for i, x in enumerate(p2):
        observer.observe(x)
        w_a = observer.get_weight("rule_add_2")
        
        # Identify the new winner (Rule B)
        w_b = 0.0
        other_nodes = [n for n in forest.active_nodes() if n.id != "rule_add_2"]
        if other_nodes:
            # We track the highest weight node that isn't Rule A
            w_b = max([observer.get_weight(n.id) for n in other_nodes])
            
        stats.append({"a": w_a, "b": w_b})
        if i % 2 == 0:
            print(f"{len(p1)+i+1:<6} | {w_a:<15.4f} | {w_b:<15.4f} | {len(list(forest.active_nodes()))}")

    # 3. Final Analysis
    print("\n--- RESULTS ANALYSIS ---")
    final_a = stats[-1]["a"]
    final_b = stats[-1]["b"]
    
    peak_a = max([s["a"] for s in stats])
    
    print(f"Rule A Peak Weight: {peak_a:.4f}")
    print(f"Rule A Final Weight: {final_a:.4f}")
    print(f"Rule B Final Weight: {final_b:.4f}")
    
    if final_b > final_a and final_a < peak_a:
        print("\n[SUCCESS] Belief Revision confirmed.")
        print("The system successfully penalized the falsified Rule A and promoted Rule B.")
    elif final_a >= peak_a:
        print("\n[FAIL] Confirmation Bias detected. The system stuck with Rule A despite contradictory evidence.")
    else:
        print("\n[PARTIAL] The system detected the shift but failed to fully promote the alternative.")

if __name__ == "__main__":
    run_experiment()
