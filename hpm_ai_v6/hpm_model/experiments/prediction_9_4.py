import numpy as np
import matplotlib.pyplot as plt
from typing import List
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.learning_agent import LearningAgent
from hpm_ai_v6.hpm_model.environments.grammar_env import GrammarEnvironment
from hpm_ai_v6.hpm_model.environments.noisy_env import NoisyEnvironment

def run_curiosity_inverted_u_experiment():
    """
    Production implementation of Prediction 9.4: 
    Curiosity peaks at intermediate complexity.
    """
    np.random.seed(42)
    
    # 1. Setup Objects
    objs = [Cell(name=f"o{i}", embedding=np.random.randn(2)) for i in range(5)]
    
    # 2. Setup Patterns
    pats = [Cell(name=f"p{i}", dim=1, embedding=np.random.randn(2), source=objs[i], target=objs[(i+1)%5]) 
            for i in range(5)]
    
    # 3. Environments of varying complexity
    # Low complexity: perfectly deterministic
    env_low = GrammarEnvironment(objs[:3])
    
    # Mid complexity: structured but with 10% noise
    env_mid = NoisyEnvironment(GrammarEnvironment(objs[:3]), noise_level=0.1, all_objects=objs[:3])
    
    # High complexity: pure random noise
    env_high = NoisyEnvironment(GrammarEnvironment(objs[:3]), noise_level=0.9, all_objects=objs[:3])
    
    results = {}
    
    for label, env in [("Low", env_low), ("Mid", env_mid), ("High", env_high)]:
        agent = LearningAgent(patterns=[Cell(name=p.name, dim=p.dim, embedding=p.embedding.copy(), 
                                           source=p.source, target=p.target) for p in pats])
        
        affective_scores = []
        for _ in range(10):
            seq = env.generate_episode(20)
            # We record the affective component of the utility
            scores = agent.perceive(seq, objs[:3], context={"consensus_vec": np.zeros(2)})
            # Average affective score across patterns
            aff_score = np.mean([agent.learner.affective.evaluate(p, {"population": objs[:3]}).score 
                               for p in agent.patterns])
            affective_scores.append(aff_score)
            
        results[label] = np.mean(affective_scores)
        print(f"Complexity: {label:6} | Affective Score (Curiosity): {results[label]:.4f}")

    print("\nVerification of Prediction 9.4:")
    if results["Mid"] > results["Low"] and results["Mid"] > results["High"]:
        print("✅ Prediction 9.4 Supported: Inverted-U Curiosity Curve observed.")
    else:
        print("⚠️ Prediction 9.4 Not Fully Supported. Adjust target entropy ratio.")

if __name__ == "__main__":
    run_curiosity_inverted_u_experiment()
",file_path: