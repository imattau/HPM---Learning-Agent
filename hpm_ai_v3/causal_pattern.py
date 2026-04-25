import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import networkx as nx
from typing import Dict, Any, List, Optional
import numpy as np
from .pattern import HPMPattern

class CausalPattern(HPMPattern):
    def __init__(self, input_dim: int, z1_dim: int = 16, z2_dim: int = 4, pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.input_dim = input_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.required_observation_keys = ["x"]
        
        self.z2_loc = nn.Parameter(torch.zeros(z2_dim))
        self.z2_scale = nn.Parameter(torch.ones(z2_dim))
        self.fc_z2_to_z1 = nn.Linear(z2_dim, z1_dim * 2)
        self.fc_z1_to_x = nn.Linear(z1_dim, input_dim * 2)
        self.fc_x_to_z1 = nn.Linear(input_dim, z1_dim * 2)
        self.fc_z1_to_z2 = nn.Linear(z1_dim, z2_dim * 2)
        
        self.mi_estimator = nn.Sequential(nn.Linear(z1_dim + z2_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mi_optimizer = torch.optim.Adam(self.mi_estimator.parameters(), lr=0.001)
        self.optimizer = Adam({"lr": 0.001})
        self.svi = None
        self.causal_graph = nx.DiGraph()
        self.causal_graph.add_edge("z2", "z1")
        self.causal_graph.add_edge("z1", "x")
        
    def parameters(self):
        return (list(self.fc_z2_to_z1.parameters()) + list(self.fc_z1_to_x.parameters()) +
                list(self.fc_x_to_z1.parameters()) + list(self.fc_z1_to_z2.parameters()) + [self.z2_loc, self.z2_scale])
    
    def model(self, observations: Optional[Dict[str, torch.Tensor]] = None):
        batch_size = observations["x"].shape[0] if observations else 1
        with pyro.plate("batch", batch_size):
            z2 = pyro.sample("z2", dist.Normal(self.z2_loc, torch.exp(self.z2_scale)).to_event(1))
            z1_p = self.fc_z2_to_z1(z2).chunk(2, dim=-1)
            z1 = pyro.sample("z1", dist.Normal(z1_p[0], torch.exp(0.5 * z1_p[1])).to_event(1))
            x_p = self.fc_z1_to_x(z1).chunk(2, dim=-1)
            x = pyro.sample("x", dist.Normal(x_p[0], torch.exp(0.5 * x_p[1])).to_event(1),
                            obs=observations["x"] if observations else None)
        return x
    
    def guide(self, observations: Dict[str, torch.Tensor]):
        x = observations["x"]
        with pyro.plate("batch", x.shape[0]):
            z1_p = self.fc_x_to_z1(x).chunk(2, dim=-1)
            z1 = pyro.sample("z1", dist.Normal(z1_p[0], torch.exp(0.5 * z1_p[1])).to_event(1))
            z2_p = self.fc_z1_to_z2(z1).chunk(2, dim=-1)
            z2 = pyro.sample("z2", dist.Normal(z2_p[0], torch.exp(0.5 * z2_p[1])).to_event(1))
        return z1, z2
    
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        conditioned = poutine.condition(self.model, data=observations)
        return poutine.trace(conditioned).get_trace().log_prob_sum()
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        conditioned = poutine.condition(self.model, data={"z2": context["z2"]}) if "z2" in context else self.model
        with pyro.plate("samples", num_samples):
            trace = poutine.trace(conditioned).get_trace()
        return {"x": trace.nodes["x"]["value"]}
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        intervened = poutine.do(self.model, data=intervention)
        conditioned = poutine.condition(intervened, data=context)
        trace = poutine.trace(conditioned).get_trace()
        return {n: trace.nodes[n]["value"] for n in ["z2", "z1", "x"] if n in trace.nodes}
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        if self.svi is None: self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        loss = self.svi.step(observations)
        self._update_loss_ema(loss)
        self.accuracy = -self.loss_ema
    
    def _update_loss_ema(self, loss: float, alpha: float = 0.9):
        self.loss_ema = loss if self.loss_ema is None else alpha * self.loss_ema + (1 - alpha) * loss
    
    def compression_score(self, observations: torch.Tensor) -> float:
        with torch.no_grad():
            guide_trace = poutine.trace(self.guide).get_trace({"x": observations})
            z1, z2 = guide_trace.nodes["z1"]["value"], guide_trace.nodes["z2"]["value"]
        
        z1_shuffled = z1[torch.randperm(z1.size(0))]
        joint, marginal = torch.cat([z1, z2], dim=-1), torch.cat([z1_shuffled, z2], dim=-1)
        
        self.mi_estimator.train()
        for _ in range(5):
            self.mi_optimizer.zero_grad()
            loss = -(self.mi_estimator(joint).mean() - torch.logsumexp(self.mi_estimator(marginal), dim=0) + np.log(marginal.size(0)))
            loss.backward()
            self.mi_optimizer.step()
        
        with torch.no_grad():
            mi = self.mi_estimator(joint).mean() - torch.logsumexp(self.mi_estimator(marginal), dim=0) + np.log(marginal.size(0))
        return float(mi)
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, CausalPattern): return 1.0
        g1, g2 = self.causal_graph, other.causal_graph
        edges1, edges2 = set(g1.edges()), set(g2.edges())
        graph_dist = 1.0 - len(edges1 & edges2) / len(edges1 | edges2) if edges1 | edges2 else 0.0
        dim_dist = abs(self.z2_dim - other.z2_dim) / max(self.z2_dim, other.z2_dim, 1)
        return 0.7 * graph_dist + 0.3 * dim_dist
    
    def extract_causal_graph(self) -> nx.DiGraph: return self.causal_graph.copy()
