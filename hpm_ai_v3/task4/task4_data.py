import numpy as np
def generate_primitive_A(n):
    x = np.random.uniform(-2, 2, (n, 1)).astype(np.float32)
    return x, np.sin(1.5 * x) + 0.5 + np.random.normal(0, 0.05, (n, 1))

def generate_primitive_B(n):
    x = np.random.uniform(-3, 3, (n, 1)).astype(np.float32)
    return x, 0.8 * x - 0.3 + np.random.normal(0, 0.05, (n, 1))

def generate_composition_C(n):
    x = np.random.uniform(-2, 2, (n, 1)).astype(np.float32)
    return x, np.sin(1.5 * (0.8 * x - 0.3)) + 0.5 + np.random.normal(0, 0.05, (n, 1))

def create_batch(gen, b): return gen(b)
