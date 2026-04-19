import numpy as np
def hard_function(x): return np.sin(20 * x) * np.exp(-0.1 * x**2)
def generate_task6_data(n=1):
    x = np.random.uniform(-5, 5, (n, 1)).astype(np.float32)
    return x, hard_function(x).astype(np.float32)
