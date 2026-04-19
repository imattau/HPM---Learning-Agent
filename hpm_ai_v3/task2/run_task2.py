import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from task1.regression_pattern import RegressionPattern
from pattern_field import PatternField
from task2_data import generate_task2_data

def main():
    # Setup agents
    print("Initializing Pattern Field agents...")
    field = PatternField(num_agents=5, pattern_factory=lambda: RegressionPattern(input_dim=2))
    
    # Phase 1: Train with spurious correlation
    print("Phase 1: Training with spurious correlation (x_spurious -> y)...")
    for _ in range(50):
        obs = [generate_task2_data(1, spurious_active=True)[0] for _ in range(5)]
        field.step_field([{"input": torch.tensor(o[0]), "target": torch.tensor([o[1]])} for o in obs])
        
    # Phase 2: Spurious correlation breaks (x_spurious is now noise)
    print("Phase 2: Spurious correlation broken. Training on causal-only data...")
    for step in range(50):
        obs = [generate_task2_data(1, spurious_active=False)[0] for _ in range(5)]
        field.step_field([{"input": torch.tensor(o[0]), "target": torch.tensor([o[1]])} for o in obs])
        
        if step % 10 == 0:
            print(f"  Step {step}: Field convergence = {field.get_field_convergence():.3f}")
        
    # Analyze weight reliance on spurious feature (index 1)
    print("\nFinal pattern analysis (weights and feature reliance):")
    for i, agent in enumerate(field.agents):
        tops = agent.get_top_patterns(k=1)
        if not tops: continue
        top = tops[0]
        # In a regression pattern, weights are in the causal structure, 
        # but for simplicity we check the encoder's first-layer coefficients.
        # This is a proxy for feature importance.
        importance = top.fc_x_to_z1.weight.data.abs().mean(dim=0)
        print(f"Agent {i} Top Pattern Importance (Causal: {importance[0]:.3f}, Spurious: {importance[1]:.3f})")

if __name__ == "__main__":
    main()
