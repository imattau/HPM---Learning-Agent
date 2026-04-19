import sys, os; sys.path.append(os.path.abspath("hpm_ai_v3"))
import torch, numpy as np
from task1.regression_pattern import RegressionPattern
from compiler.symbolic_compiler import SymbolicRegressionCompiler
from task1.task1_data import generate_task1_data

# 1. Train a pattern on Task 1
print("Training neural pattern on Task 1...")
pattern = RegressionPattern(input_dim=6, output_dim=1)
train_data = generate_task1_data(2000)
for x, y in train_data:
    pattern.update_parameters({"input": torch.tensor(x).unsqueeze(0), "target": torch.tensor([y]).unsqueeze(0)})

# 2. Compile it
print("Distilling into symbolic expression...")
compiler = SymbolicRegressionCompiler()

def sample_gen(n):
    x = np.zeros((n, 6), dtype=np.float32)
    x[:, 0] = np.random.uniform(-2, 2, n) # Structural A
    x[:, 1] = np.random.uniform(-2, 2, n) # Structural B
    return x

sym_pat = compiler.compile(pattern, 6, sample_gen, ['a', 'b', 'cr', 'cb', 'sc', 'ss'])

# 3. Test the symbolic distillation
print("\nVerifying Distilled Logic:")
for _ in range(5):
    a, b = np.random.uniform(-2, 2, 2)
    ctx = {'a': a, 'b': b, 'cr': 0, 'cb': 0, 'sc': 0, 'ss': 0}
    # True logic is a*b + sin(a) if a > 0 else a + b
    true = (a*b + np.sin(a)) if a > 0 else (a + b)
    print(f"a={a:.2f}, b={b:.2f} | True={true:.3f} | Distilled={sym_pat.forward_fn(ctx):.3f}")
