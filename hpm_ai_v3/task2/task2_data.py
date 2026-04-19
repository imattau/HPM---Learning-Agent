import numpy as np
import torch

def generate_task2_data(n_samples=1000, spurious_active=True):
    """
    Generate data for the "Monday Effect" spurious correlation task.
    y = x_causal + (0.5 * x_spurious if spurious_active else 0.0) + noise.
    """
    data = []
    for _ in range(n_samples):
        # Structural feature
        x_causal = np.random.uniform(-1, 1)
        # Spurious feature (e.g., 'Monday' indicator: 1 = Monday, 0 = other)
        x_spurious = np.random.choice([0, 1])
        
        # Target with spurious correlation
        y = x_causal + (0.5 * x_spurious if spurious_active else 0.0)
        y += np.random.normal(0, 0.05)
        
        x = np.array([x_causal, x_spurious], dtype=np.float32)
        data.append((x, y))
    return data

def create_torch_dataset(data):
    X = torch.tensor([d[0] for d in data], dtype=torch.float32)
    y = torch.tensor([d[1] for d in data], dtype=torch.float32).unsqueeze(1)
    return X, y
