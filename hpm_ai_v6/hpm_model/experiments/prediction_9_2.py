import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.learning_agent import LearningAgent
from hpm_ai_v6.hpm_model.environments.grammar_env import GrammarEnvironment
from hpm_ai_v6.hpm_model.environments.noisy_env import NoisyEnvironment

def run_expertise_transfer_experiment():
    """
    Production implementation of Prediction 9.2: 
    Expertise as compression and cross-surface transfer.
    """
    np.random.seed(42)
    
    # 1. Setup Domains
    # Domain A (Original)
    apple = Cell(name="apple", embedding=np.array([1.0, 0.2]))
    banana = Cell(name="banana", embedding=np.array([0.8, 0.9]))
    cherry = Cell(name="cherry", embedding=np.array([0.1, 1.2]))
    domain_a_objects = [apple, banana, cherry]
    
    # Domain B (Aliased surface, same deep structure)
    X = Cell(name="X", embedding=np.array([-0.2, 1.0]))
    Y = Cell(name="Y", embedding=np.array([-0.9, 0.8]))
    Z = Cell(name="Z", embedding=np.array([-1.2, 0.1]))
    domain_b_objects = [X, Y, Z]
    
    # 2. Setup Patterns (1-cells)
    base_pats = [
        Cell(name="P1", dim=1, embedding=np.random.randn(2), source=apple, target=banana),
        Cell(name="P2", dim=1, embedding=np.random.randn(2), source=banana, target=cherry),
        Cell(name="P3", dim=1, embedding=np.random.randn(2), source=cherry, target=apple)
    ]
    
    # 3. Initialize Agents
    novice = LearningAgent(patterns=deepcopy_patterns(base_pats))
    expert = LearningAgent(patterns=deepcopy_patterns(base_pats))
    
    env_a = GrammarEnvironment(domain_a_objects)
    
    # 4. Training
    print("Training Novice (5 episodes)...")
    for _ in range(5):
        seq = env_a.generate_episode(15)
        novice.perceive(seq, domain_a_objects, context={})
        
    print("Training Expert (50 episodes)...")
    for _ in range(50):
        seq = env_a.generate_episode(15)
        expert.perceive(seq, domain_a_objects, context={})
        
    # 5. Evaluate Transfer to Domain B
    # We must map the expert's patterns to the new domain's objects based on structure
    # For the transfer test, we assume the agent 'tries' its patterns on the new objects.
    
    def evaluate(agent, objects):
        # We simulate a test where the agent predicts sequences in Domain B
        env_b = GrammarEnvironment(objects)
        test_seq = env_b.generate_episode(20)
        
        # We check the log-likelihood of the best pattern
        best_pat = agent.get_best_pattern()
        # Mocking the source for Domain B evaluation
        # In a real transfer, the agent would search for which object 'A' corresponds to in the new domain
        # Here we manually set the source to the corresponding aliased object for the test
        source_map = {apple.name: objects[0], banana.name: objects[1], cherry.name: objects[2]}
        best_pat.source = source_map[best_pat.source.name]
        
        count = 0
        total_ll = 0.0
        for i in range(len(test_seq)-1):
            curr, next_obj = test_seq[i], test_seq[i+1]
            if curr.name == best_pat.source.name:
                probs = best_pat.predict_probs(objects)
                idx = objects.index(next_obj)
                total_ll += np.log(probs[idx] + 1e-9)
                count += 1
        return total_ll / count if count > 0 else -10.0

    novice_ll = evaluate(novice, domain_b_objects)
    expert_ll = evaluate(expert, domain_b_objects)
    
    print(f"\nNovice Transfer LL: {novice_ll:.4f}")
    print(f"Expert Transfer LL: {expert_ll:.4f}")
    
    if expert_ll > novice_ll:
        print("✅ Prediction 9.2 Supported: Expert transfers better to surface-changed domain.")

def deepcopy_patterns(patterns: List[Cell]) -> List[Cell]:
    from copy import deepcopy
    return [Cell(name=p.name, dim=p.dim, embedding=p.embedding.copy(), 
                 source=p.source, target=p.target) for p in patterns]

if __name__ == "__main__":
    run_expertise_transfer_experiment()
