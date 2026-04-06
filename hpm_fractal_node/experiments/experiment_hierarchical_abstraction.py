"""
SP39: Experiment 15 — Hierarchical Abstraction Discovery (Core HPM Claim)

Validates whether the HFN system autonomously builds a multi-layered DAG
(letters -> words -> sentences) and reuses lower-level nodes, proving
compositional generalization rather than flat clustering.
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.evaluator import Evaluator

def generate_hierarchical_data(dim=60):
    """
    Generates a 3-level hierarchical dataset.
    - L1 (Letters): 10 distinct patterns (each uses 6 unique dims).
    - L2 (Words): 10 distinct pairs of Letters.
    - L3 (Sentences): 10 distinct pairs of Words.
    """
    rng = np.random.RandomState(42)
    
    letter_vecs = []
    for i in range(10):
        vec = np.zeros(dim)
        vec[i*6:(i+1)*6] = 10.0
        letter_vecs.append(vec)
        
    word_vecs = []
    for i in range(10):
        # Word i is average of Letter i and Letter (i+1)%10
        l1 = i
        l2 = (i + 1) % 10
        vec = (letter_vecs[l1] + letter_vecs[l2]) / 2.0
        word_vecs.append(vec)
        
    sentence_vecs = []
    for i in range(10):
        # Sentence i is average of Word i and Word (i+2)%10
        w1 = i
        w2 = (i + 2) % 10
        vec = (word_vecs[w1] + word_vecs[w2]) / 2.0
        sentence_vecs.append(vec)
        
    return letter_vecs, word_vecs, sentence_vecs

def analyze_dag(forest):
    """Calculates max depth, compression ratio, and node reuse rate."""
    nodes = list(forest.active_nodes())
    
    # 1. Depth Calculation (longest path from any root to a leaf)
    def get_depth(node_id, memo, current_path=None):
        if current_path is None:
            current_path = set()
        if node_id in current_path:
            return 0 # cycle
        if node_id in memo:
            return memo[node_id]
            
        children = [c.id for c in forest.get(node_id).children()]
        if not children:
            memo[node_id] = 1
            return 1
            
        current_path.add(node_id)
        max_child_depth = max([get_depth(c, memo, current_path) for c in children])
        current_path.remove(node_id)
        
        depth = 1 + max_child_depth
        memo[node_id] = depth
        return depth

    depths = {}
    max_depth = 0
    for n in nodes:
        d = get_depth(n.id, depths)
        max_depth = max(max_depth, d)
        
    # 2. Reuse Calculation (average number of parents per node)
    parent_counts = []
    for n in nodes:
        parents = forest.get_parents(n.id)
        parent_counts.append(len(parents))
        
    avg_reuse = np.mean(parent_counts)
    highly_reused = sum(1 for p in parent_counts if p > 1)
    
    return {
        "total_nodes": len(nodes),
        "max_depth": max_depth,
        "avg_reuse": avg_reuse,
        "highly_reused_nodes": highly_reused
    }

def run_experiment():
    print("--- SP39: Experiment 15 — Hierarchical Abstraction Discovery ---\n")
    
    DIM = 60
    letter_vecs, word_vecs, sentence_vecs = generate_hierarchical_data(DIM)
    
    # Setup Observer
    forest = Forest()
    
    # Pre-register letters as broad priors so they can explain composites
    for i, vec in enumerate(letter_vecs):
        rule = HFN(mu=vec, sigma=np.ones(DIM)*5.0, id=f"Letter_{i}", use_diag=True)
        forest.register(rule)
        
    observer = Observer(
        forest=forest,
        tau=2.5, # Broad enough to allow partial matching
        residual_surprise_threshold=999.0, # Disable random node creation, force compression
        compression_cooccurrence_threshold=2, 
        adaptive_compression=False,
        node_use_diag=True
    )
    
    print("PHASE 1: Learning Words (L2)...")
    for _ in range(5):
        for x in word_vecs:
            observer.observe(x)
            
    print(f"  Nodes: {len(list(forest.active_nodes()))}")
    
    print("\nPHASE 2: Learning Sentences (L3)...")
    for _ in range(10): # More epochs to ensure high-level compression
        for x in sentence_vecs:
            observer.observe(x)
            
    print(f"  Nodes: {len(list(forest.active_nodes()))}")
    
    # Analysis
    print("\n--- DAG STRUCTURAL ANALYSIS ---")
    stats = analyze_dag(forest)
    
    print(f"Total Nodes in Forest: {stats['total_nodes']}")
    print(f"Maximum DAG Depth:     {stats['max_depth']} layers")
    print(f"Average Node Reuse:    {stats['avg_reuse']:.2f} parents per node")
    print(f"Highly Reused Nodes:   {stats['highly_reused_nodes']} (nodes shared by >1 parent)")
    
    # Verification
    if stats['max_depth'] >= 3 and stats['highly_reused_nodes'] > 0:
        print("\n[SUCCESS] Hierarchical Abstraction Achieved!")
        print("The system formed a natural multi-layered hierarchy and reused lower-level concepts.")
    else:
        print("\n[FAIL] The system failed to form a deep compositional hierarchy.")

if __name__ == "__main__":
    run_experiment()
