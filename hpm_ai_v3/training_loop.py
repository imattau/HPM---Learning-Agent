import torch
import numpy as np
from .causal_pattern import CausalPattern
from .pattern_field import PatternField

def create_random_pattern():
    return CausalPattern(input_dim=2, z1_dim=8, z2_dim=2)

def main():
    field = PatternField(num_agents=5, pattern_factory=create_random_pattern)

    def generate_batch(batch_size=10):
        if np.random.random() > 0.5:
            x = torch.randn(batch_size, 2) * 0.5 + torch.tensor([2.0, 2.0])
        else:
            x = torch.randn(batch_size, 2) * 0.5 + torch.tensor([-2.0, -2.0])
        return {"x": x}

    for step in range(100):
        obs_batch = [generate_batch(1) for _ in range(5)]
        field.step_field(obs_batch)
        
        if step % 20 == 0:
            conv = field.get_field_convergence()
            print(f"Step {step}: Field convergence = {conv:.3f}")

if __name__ == "__main__":
    main()
