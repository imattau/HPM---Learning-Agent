import numpy as np
import torch
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.metrics import mean_squared_error
from typing import Optional, Callable, List
from pattern import HPMPattern

class SymbolicPattern(HPMPattern):
    def __init__(self, forward_fn, required_keys, pattern_id=None):
        super().__init__(pattern_id)
        self.forward_fn = forward_fn
        self.required_observation_keys = required_keys
        self.substrate_type = "symbolic_gp"
        self.weight = 0.05
    def log_prob(self, obs): return torch.tensor(0.0)
    def sample(self, ctx):
        # Flatten context inputs for symbolic call
        flat_ctx = {k: v.item() if isinstance(v, torch.Tensor) and v.numel()==1 else v for k, v in ctx.items()}
        # For our test logic, handle flat dict
        return {"y": torch.tensor([self.forward_fn(flat_ctx)])}
    def update_parameters(self, obs, lr=0.01): pass
    def structural_distance(self, other): return 0.0
    def extract_causal_graph(self): import networkx as nx; return nx.DiGraph()
    def intervene(self, intervention, context): return self.sample(context)

class SymbolicRegressionCompiler:
    def __init__(self):
        self.func_set = [make_function(function=lambda x,y: x+y, name='add', arity=2), 
                         make_function(function=lambda x,y: x*y, name='mul', arity=2),
                         make_function(function=np.sin, name='sin', arity=1)]
        
    def compile(self, pattern, input_dim, sample_gen, var_names=None):
        X, y = [], []
        for _ in range(500):
            x = sample_gen(1)
            X.append(x.flatten())
            with torch.no_grad():
                y.append(pattern.sample({"input": torch.tensor(x).float()})["y"].item())
        
        est = SymbolicRegressor(population_size=500, generations=10, function_set=self.func_set, verbose=0).fit(np.array(X), np.array(y))
        
        var_names = var_names or [f'x{i}' for i in range(input_dim)]
        def fn(ctx):
            # This is a hacky eval, in real production use safer AST traversal
            expr = str(est._program).replace("X0", str(ctx[var_names[0]]))
            return eval(expr)
        
        return SymbolicPattern(fn, var_names)
