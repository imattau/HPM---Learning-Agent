import numpy as np
import torch

def generate_task1_data(n_samples=1000, noise_std=0.1):
    data = []
    for _ in range(n_samples):
        a = np.random.uniform(-2, 2)
        b = np.random.uniform(-2, 2)
        color = np.random.choice(['red', 'blue'])
        shape = np.random.choice(['circle', 'square'])
        
        if a > 0:
            y = a * b + np.sin(a)
        else:
            y = a + b
        y += np.random.normal(0, noise_std)
        
        color_red = 1.0 if color == 'red' else 0.0
        color_blue = 1.0 if color == 'blue' else 0.0
        shape_circle = 1.0 if shape == 'circle' else 0.0
        shape_square = 1.0 if shape == 'square' else 0.0
        
        x = np.array([a, b, color_red, color_blue, shape_circle, shape_square], dtype=np.float32)
        data.append((x, y))
    return data

def create_torch_dataset(data):
    X = torch.tensor([d[0] for d in data], dtype=torch.float32)
    y = torch.tensor([d[1] for d in data], dtype=torch.float32).unsqueeze(1)
    return X, y
