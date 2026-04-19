import sys, os; sys.path.append(os.path.abspath("hpm_ai_v3"))
import torch, numpy as np
from neural_pattern import RegressionPattern
from compiler.symbolic_compiler import SymbolicRegressionCompiler

# Train a dummy pattern
pattern = RegressionPattern(input_dim=1, output_dim=1)
for _ in range(500):
    x = torch.tensor([[np.random.uniform(-1, 1)]])
    y = torch.sin(x)
    pattern.update_parameters({"input": x, "target": y})

# Compile
comp = SymbolicRegressionCompiler()
sym = comp.compile(pattern, 1, lambda n: np.random.uniform(-1, 1, (n, 1)), ['x0'])
print(f"Compiled successfully: {sym}")
