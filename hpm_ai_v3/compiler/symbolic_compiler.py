import numpy as np
import torch
from pysr import PySRRegressor
from pattern import HPMPattern

class SymbolicPattern(HPMPattern):
    def __init__(self, model, required_keys, pattern_id=None):
        super().__init__(pattern_id)
        self.model = model
        self.required_observation_keys = required_keys
        self.substrate_type = "symbolic_gp"
        self.weight = 0.05
    def log_prob(self, obs): return torch.tensor(0.0)
    def sample(self, ctx):
        # PySR model can be called directly
        inputs = np.array([[ctx[k] for k in self.required_observation_keys]])
        return {"y": torch.tensor([self.model.predict(inputs)[0]])}
    def update_parameters(self, obs, lr=0.01): pass
    def structural_distance(self, other): return 0.0
    def extract_causal_graph(self): import networkx as nx; return nx.DiGraph()
    def intervene(self, intervention, context): return self.sample(context)

class SymbolicRegressionCompiler:
    def compile(self, pattern, input_dim, sample_gen, var_names=None):
        X, y = [], []
        for _ in range(500):
            x = sample_gen(1)
            X.append(x.flatten())
            with torch.no_grad():
                y.append(pattern.sample({"input": torch.tensor(x).float()})["y"].item())
        X, y = np.array(X), np.array(y)
        
        # PySR symbolic regression
        model = PySRRegressor(
            niterations=10, 
            binary_operators=["+", "*", "-"], 
            unary_operators=["sin"],
            verbosity=0
        )
        model.fit(X, y)
        return SymbolicPattern(model, var_names or [f'x{i}' for i in range(input_dim)])
