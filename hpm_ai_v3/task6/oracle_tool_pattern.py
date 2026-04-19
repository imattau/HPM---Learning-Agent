import torch, networkx as nx
from typing import Dict, Any, Optional, Callable
from pattern import HPMPattern
class OracleToolPattern(HPMPattern):
    def __init__(self, oracle_fn, cost=0.3, pattern_id=None):
        super().__init__(pattern_id)
        self.oracle_fn = oracle_fn
        self.cost = cost
        self.substrate_type = "tool"
    def log_prob(self, obs): return torch.tensor(0.0)
    def sample(self, ctx): return {"y": torch.tensor(self.oracle_fn(ctx["input"].detach().cpu().numpy()))}
    def update_parameters(self, obs, lr=0.01): self.accuracy = -self.loss_ema if self.loss_ema else -0.5
    def structural_distance(self, other): return 0.0
    
    def intervene(self, intervention, context):
        return self.sample(context)
    
    def extract_causal_graph(self): return nx.DiGraph()
