#!/usr/bin/env python3
"""
Prediction 9.1: Hierarchical structure vs surface features.
Trains an HPM agent on a hierarchical grammar, then tests sensitivity to
surface changes vs deep structural changes.
"""

import numpy as np
import os
import sys
from copy import deepcopy

# Ensure the project root is in the path
sys.path.append(os.getcwd())

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.learning_agent import LearningAgent
from hpm_ai_v6.hpm_model.environments.grammar_env import GrammarEnvironment
from hpm_ai_v6.hpm_model.environments.noisy_env import NoisyEnvironment

def run_prediction_9_1_experiment():
    """
    Evaluates sensitivity to deep vs surface changes.
    Baseline: A -> B -> C cycle.
    Surface Change: X -> Y -> Z cycle (isomorphic deep structure).
    Deep Change: A -> C -> B cycle (different deep structure).
    """
    np.random.seed(42)
    
    # 1. Objects
    # Domain A
    apple = Cell(name="apple", dim=0, embedding=np.array([1.0, 0.0]))
    banana = Cell(name="banana", dim=0, embedding=np.array([0.0, 1.0]))
    cherry = Cell(name="cherry", dim=0, embedding=np.array([-1.0, 0.0]))
    domain_a = [apple, banana, cherry]
    
    # Domain B (Surface Aliased)
    # These have different embeddings but we assume the agent can still match them 
    # if it has learned the abstract transition vectors.
    X = Cell(name="X", dim=0, embedding=np.array([0.7, 0.7]))
    Y = Cell(name="Y", dim=0, embedding=np.array([-0.7, 0.7]))
    Z = Cell(name="Z", dim=0, embedding=np.array([-0.7, -0.7]))
    domain_b = [X, Y, Z]

    # 2. Initial Patterns (1-cells)
    # We create patterns for the training grammar
    pats = [
        Cell(name="A_to_B", dim=1, embedding=banana.embedding - apple.embedding, source=apple, target=banana),
        Cell(name="B_to_C", dim=1, embedding=cherry.embedding - banana.embedding, source=banana, target=cherry),
        Cell(name="C_to_A", dim=1, embedding=apple.embedding - cherry.embedding, source=cherry, target=apple)
    ]
    
    agent = LearningAgent(patterns=pats, beta_e=1.0, beta_a=0.0, beta_s=0.0) # Focus on epistemic accuracy
    
    # 3. Training on Baseline (A->B->C)
    env_base = GrammarEnvironment(domain_a)
    print("Training agent on baseline grammar (A->B->C)...")
    for _ in range(20):
        seq = env_base.generate_episode(10)
        agent.perceive(seq, domain_a, context={})

    # 4. Evaluation Function
    def evaluate_ll(agent, objects, seq):
        total_ll = 0.0
        count = 0
        
        for i in range(len(seq) - 1):
            curr, next_obj = seq[i], seq[i+1]
            
            # Find pattern whose source matches curr (using vector similarity)
            best_p = None
            max_sim = -1.0
            for p in agent.patterns:
                sim = p.source.similarity(curr)
                if sim > max_sim:
                    max_sim = sim
                    best_p = p
            
            if best_p:
                probs = best_p.predict_probs(objects)
                try:
                    idx = objects.index(next_obj)
                    total_ll += np.log(probs[idx] + 1e-9)
                    count += 1
                except ValueError:
                    pass
        return total_ll / count if count > 0 else -10.0

    # 5. Tests
    print("\n--- Evaluation ---")
    
    # Baseline Test
    seq_base = env_base.generate_episode(20)
    ll_base = evaluate_ll(agent, domain_a, seq_base)
    print(f"Baseline LL:       {ll_base:.4f}")
    
    # Surface Change Test (X->Y->Z)
    env_surface = GrammarEnvironment(domain_b)
    seq_surface = env_surface.generate_episode(20)
    ll_surface = evaluate_ll(agent, domain_b, seq_surface)
    print(f"Surface Change LL: {ll_surface:.4f}")
    
    # Deep Change Test (A->C->B)
    class ReverseGrammar(GrammarEnvironment):
        def generate_episode(self, length: int) -> list:
            seq = []
            cur_idx = 0
            for _ in range(length):
                seq.append(self.objects[cur_idx])
                cur_idx = (cur_idx - 1) % 3
            return seq
            
    env_deep = ReverseGrammar(domain_a)
    seq_deep = env_deep.generate_episode(20)
    ll_deep = evaluate_ll(agent, domain_a, seq_deep)
    print(f"Deep Change LL:    {ll_deep:.4f}")
    
    # 6. Analysis
    drop_surface = ll_base - ll_surface
    drop_deep = ll_base - ll_deep
    
    print(f"\nDrop (Surface): {drop_surface:.4f}")
    print(f"Drop (Deep):    {drop_deep:.4f}")
    
    if drop_deep > drop_surface:
        print("\n✅ Prediction 9.1 Supported: Agent is more sensitive to deep structure changes.")
    else:
        print("\n⚠️ Prediction 9.1 Not Supported: Check embedding similarity or pattern mapping.")

if __name__ == "__main__":
    run_prediction_9_1_experiment()
