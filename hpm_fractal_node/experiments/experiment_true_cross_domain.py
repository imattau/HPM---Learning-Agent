"""
SP35: Experiment 11 — True Cross-Domain Transfer (Structural Equivalence)

Validates whether HFN can learn abstract structure independently of surface geometry.
Setup:
1. Domain A: Manifold [0:10]. Train heavily on a set of rules.
2. Domain B: Manifold [10:20]. Same rules, different surface.
3. Rosetta Phase: A few joint observations (A+B) to bridge domains.
4. Probe B: Test if system reuses Domain A rules for Domain B tasks.

Metrics:
- Reuse Rate: % of Domain A nodes activated during Domain B probe.
- Creation Rate: Does the system build redundant Domain B rules?
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.evaluator import Evaluator

def generate_curriculum(dim=20, n_rules=3, n_examples_per_rule=10):
    """
    Creates cross-domain data with a Shared Rule Subspace.
    Domain A Surface: Dims 0-5
    Domain B Surface: Dims 5-10
    Shared Structure: Dims 10-15
    """
    rng = np.random.RandomState(42)
    
    domain_a_train = []
    domain_b_train = [] 
    rosetta_stone = []
    
    for r in range(n_rules):
        rule_val = (r + 1) * 2.0  # The shared structural concept
        
        base_inp_a = r * 10.0
        base_inp_b = 100.0 + (r * 10.0)
        
        # 1. Domain A
        for _ in range(n_examples_per_rule):
            vec_a = np.zeros(dim)
            vec_a[0] = rng.normal(base_inp_a, 0.5) # Surface A
            vec_a[10] = rule_val                   # Shared Structure
            domain_a_train.append(vec_a)
            
        # 2. Domain B
        for _ in range(n_examples_per_rule):
            vec_b = np.zeros(dim)
            vec_b[5] = rng.normal(base_inp_b, 0.5) # Surface B
            vec_b[10] = rule_val                   # Shared Structure
            domain_b_train.append(vec_b)
            
        # 3. Rosetta Stone - bridging A and B
        for _ in range(5): 
            vec_rosetta = np.zeros(dim)
            vec_rosetta[0] = rng.normal(base_inp_a, 0.5)
            vec_rosetta[5] = rng.normal(base_inp_b, 0.5)
            vec_rosetta[10] = rule_val             # Shared Structure
            rosetta_stone.append(vec_rosetta)
            
    return domain_a_train, rosetta_stone, domain_b_train

def run_experiment():
    print("--- SP35: Experiment 11 — True Cross-Domain Transfer ---\n")
    
    DIM = 20
    N_RULES = 3
    EXAMPLES = 15
    
    train_a, rosetta, probe_b = generate_curriculum(DIM, N_RULES, EXAMPLES)
    
    # 1. Setup Observer
    forest = Forest()
    decoder = Decoder(target_forest=forest, sigma_threshold=1.5)
    observer = Observer(
        forest=forest,
        tau=1.0,
        node_use_diag=True,
        budget=8
    )

    
    print("PHASE 1: Training Domain A (Math Subspace)...")
    np.random.shuffle(train_a)
    for x in train_a:
        observer.observe(x)
        
    nodes_after_a = len(list(forest.active_nodes()))
    ids_a = {n.id for n in forest.active_nodes()}
    print(f"  Forest Size after Domain A: {nodes_after_a}")
    
    print("\nPHASE 2: Rosetta Stone (Joint A+B Observations)...")
    np.random.shuffle(rosetta)
    for x in rosetta:
        observer.observe(x)
        
    nodes_after_rosetta = len(list(forest.active_nodes()))
    print(f"  Forest Size after Rosetta:  {nodes_after_rosetta} (+{nodes_after_rosetta - nodes_after_a} bridging nodes)")
    
    print("\nPHASE 3: Probing Domain B (Symbolic Subspace)...")
    
    success_count = 0
    reuse_count = 0
    
    for idx, x_b in enumerate(probe_b):
        # Create a partial observation: Surface B is known, Structure B is unknown
        query_mu = np.zeros(DIM)
        query_mu[5] = x_b[5] 
        
        # We want the decoder to predict the Rule (dims 10-15)
        goal = HFN(mu=query_mu, sigma=np.ones(DIM)*5.0, id=f"probe_{idx}", use_diag=True)
        goal.sigma[5] = 0.01 # Lock Surface B
        
        # Decode
        dec_res = decoder.decode(goal)
        
        predicted_rule = None
        if isinstance(dec_res, list) and dec_res:
            pred_node = dec_res[0]
            predicted_rule = pred_node.mu[10]
            
            # Check if the predicted node has ancestry in Domain A
            # Trace parents up to see if any are from ids_a
            def has_domain_a_ancestry(n):
                if n.id in ids_a: return True
                for p in forest.get_parents(n.id):
                    if has_domain_a_ancestry(p): return True
                return False
                
            if has_domain_a_ancestry(pred_node):
                reuse_count += 1
                
        if predicted_rule is not None:
            # Check accuracy of prediction
            true_rule = x_b[10]
            if abs(predicted_rule - true_rule) < 0.5:
                success_count += 1
            
    print("\n--- RESULTS ANALYSIS ---")
    total_probes = len(probe_b)
    
    accuracy = success_count / total_probes
    reuse_rate = reuse_count / total_probes
    
    print(f"Domain B Explanation Accuracy: {accuracy*100:.1f}%")
    print(f"Domain A Structural Reuse:     {reuse_rate*100:.1f}%")
    
    print(f"\nFinal Forest Size: {len(list(forest.active_nodes()))}")
    
    if accuracy > 0.8 and reuse_rate > 0.5:
        print("\n[SUCCESS] True Cross-Domain Transfer Achieved!")
        print("The system discovered structural equivalence and reused Domain A's deep structure to solve Domain B without external alignment.")
    else:
        print("\n[FAIL] The system failed to transfer structure across domains.")

if __name__ == "__main__":
    run_experiment()
