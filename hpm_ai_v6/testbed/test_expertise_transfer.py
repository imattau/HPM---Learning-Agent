"""
Test Prediction 9.2: Expertise as compression and cross‑surface transfer.
An expert (many training episodes) should show:
- Higher compression (structural similarity proxy).
- Better transfer accuracy to surface‑changed versions of the same deep structure.
"""

import numpy as np
import os
import sys
from copy import deepcopy

# Ensure the testbed is in the path
sys.path.append(os.path.join(os.getcwd(), 'hpm_ai_v6/testbed'))

from toy_hpm_v4 import (
    Cell, MetaPatternRule, learn_step, hierarchical_total_score, compression,
    make_1cell
)

# 1. Setup Objects
apple = Cell(0, np.array([1.0, 0.2]), name="apple")
banana = Cell(0, np.array([0.8, 0.9]), name="banana")
cherry = Cell(0, np.array([0.1, 1.2]), name="cherry")
orig_objects = [apple, banana, cherry]

# Aliased objects (Surface change: rotated/shifted embeddings but same identities)
X = Cell(0, np.array([-0.2, 1.0]), name="X")
Y = Cell(0, np.array([-0.9, 0.8]), name="Y")
Z = Cell(0, np.array([-1.2, 0.1]), name="Z")
aliased_objects = [X, Y, Z]

# 2. Environment Generator
def generate_structured_episode(objects_map, length=20):
    """Hidden grammar: A->B, B->C, C->A cycles."""
    seq = []
    cur = objects_map[0]
    for _ in range(length):
        seq.append(cur)
        if cur == objects_map[0]: cur = objects_map[1]
        elif cur == objects_map[1]: cur = objects_map[2]
        else: cur = objects_map[0]
    return seq

# 3. Agent Wrapper (reusing HPMAgent logic)
class HPMAgent:
    def __init__(self, base_patterns, agent_id=0):
        self.patterns = []
        for p in base_patterns:
            noisy_emb = p.emb + np.random.randn(*p.emb.shape) * 0.05
            self.patterns.append(Cell(p.dim, noisy_emb, p.src, p.tgt, p.name))
        self.mpr = MetaPatternRule(self.patterns, learning_rate=0.2, decay=0.99)
        self.mpr.weights = np.random.dirichlet(np.ones(len(self.patterns)))
        self.id = agent_id

    def learn(self, seq, objects_list, n_steps=5):
        from toy_hpm_v4 import DynamicPatternField
        field = DynamicPatternField(self.patterns)
        consensus_vec = np.array([1.0, 0.0])
        for _ in range(n_steps):
            learn_step(self.mpr, field, seq, self.patterns + objects_list, consensus_vec)

def evaluate(agent, objects_map, length=20):
    """Compute average log-likelihood of observations given the weights."""
    seq = generate_structured_episode(objects_map, length)
    total_ll = 0.0
    count = 0
    
    # Simple evaluation: how well does the most weighted pattern predict?
    best_pat = agent.patterns[np.argmax(agent.mpr.weights)]
    population = objects_map
    
    for i in range(len(seq) - 1):
        curr, next_obj = seq[i], seq[i+1]
        # In this toy, we check if the pattern's source matches current object
        # and see the probability of next_obj.
        if best_pat.src.name == curr.name: # name-based matching for aliased transfer
            probs = best_pat.predict_probs(population)
            idx = population.index(next_obj)
            total_ll += np.log(probs[idx] + 1e-9)
            count += 1
    return total_ll / count if count > 0 else -10.0

# 4. Run Experiment
def run_expertise_test():
    np.random.seed(42)
    
    # Initial patterns (weakly trained/random)
    base_pats = [
        Cell(1, np.random.randn(2), source=apple, target=banana, name="P1"),
        Cell(1, np.random.randn(2), source=banana, target=cherry, name="P2"),
        Cell(1, np.random.randn(2), source=cherry, target=apple, name="P3")
    ]
    
    novice = HPMAgent(base_pats, agent_id=0)
    expert = HPMAgent(base_pats, agent_id=1)
    
    # Train Novice briefly
    novice.learn(generate_structured_episode(orig_objects, 10), orig_objects, n_steps=2)
    
    # Train Expert extensively
    expert.learn(generate_structured_episode(orig_objects, 30), orig_objects, n_steps=20)
    
    # Results
    print("=== Expertise and Transfer (Prediction 9.2) ===")
    
    # Compression (proxy: mean compression of patterns)
    novice_comp = np.mean([compression(p) for p in novice.patterns])
    expert_comp = np.mean([compression(p) for p in expert.patterns])
    
    # Transfer Accuracy
    novice_transfer = evaluate(novice, aliased_objects)
    expert_transfer = evaluate(expert, aliased_objects)
    
    print(f"Novice Compression: {novice_comp:.4f} | Transfer LL: {novice_transfer:.4f}")
    print(f"Expert Compression: {expert_comp:.4f} | Transfer LL: {expert_transfer:.4f}")
    
    if expert_comp > novice_comp:
        print("[PASS] Expert has higher structural compression.")
    if expert_transfer > novice_transfer:
        print("[PASS] Expert has better transfer accuracy to aliased environment.")

if __name__ == "__main__":
    run_expertise_test()
