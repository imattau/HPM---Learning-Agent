import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import networkx as nx
from typing import Dict, Any, Optional, List
from pattern import HPMPattern

class RegressionPattern(HPMPattern):
    def __init__(self, input_dim: int, output_dim: int = 1, z1_dim: int = 16, z2_dim: int = 8, pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.required_observation_keys = ["input", "target"]
        self.substrate_type = "neural"
        self.sparsity_lambda = 0.01
        
        self.fc_x_to_z1 = nn.Linear(input_dim, z1_dim * 2)
        self.z2_loc = nn.Parameter(torch.zeros(z2_dim))
        self.z2_scale = nn.Parameter(torch.ones(z2_dim))
        self.fc_z2_to_z1 = nn.Linear(z2_dim, z1_dim * 2)
        self.fc_z1_to_y = nn.Linear(z1_dim, output_dim * 2)
        self.fc_z1_to_z2 = nn.Linear(z1_dim, z2_dim * 2)
        
        self.optimizer = Adam({"lr": 0.001})
        self.svi = None
        self.causal_graph = nx.DiGraph()
        self.causal_graph.add_edge("x", "z1")
        self.causal_graph.add_edge("z2", "z1")
        self.causal_graph.add_edge("z1", "y")

        self.to(self._device)
        
    def to(self, device: torch.device):
        self._device = device
        self.fc_x_to_z1.to(device)
        self.z2_loc.data = self.z2_loc.data.to(device)
        self.z2_scale.data = self.z2_scale.data.to(device)
        self.fc_z2_to_z1.to(device)
        self.fc_z1_to_y.to(device)
        self.fc_z1_to_z2.to(device)
        return self

    def parameters(self):
        return (list(self.fc_x_to_z1.parameters()) + list(self.fc_z2_to_z1.parameters()) +
                list(self.fc_z1_to_y.parameters()) + list(self.fc_z1_to_z2.parameters()) +
                [self.z2_loc, self.z2_scale])
    
    def model(self, observations: Optional[Dict[str, torch.Tensor]] = None):
        if observations:
            observations = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        batch_size = observations["input"].shape[0] if observations else 1
        with pyro.plate("batch", batch_size):
            z2 = pyro.sample("z2", dist.Normal(self.z2_loc, torch.exp(self.z2_scale.clamp(-10, 10))).to_event(1))
            pyro.factor("z2_sparsity", -self.sparsity_lambda * z2.abs().sum())
            z1_p = self.fc_z2_to_z1(z2).chunk(2, dim=-1)
            z1 = pyro.sample("z1", dist.Normal(z1_p[0], torch.exp(0.5 * z1_p[1].clamp(-10, 10))).to_event(1))
            y_p = self.fc_z1_to_y(z1).chunk(2, dim=-1)
            y = pyro.sample("y", dist.Normal(y_p[0], torch.exp(0.5 * y_p[1].clamp(-10, 10))).to_event(1),
                            obs=observations["target"] if observations else None)
        return y
    
    def guide(self, observations: Dict[str, torch.Tensor]):
        observations = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        x = observations["input"]
        with pyro.plate("batch", x.shape[0]):
            z1_p = self.fc_x_to_z1(x).chunk(2, dim=-1)
            z1 = pyro.sample("z1", dist.Normal(z1_p[0], torch.exp(0.5 * z1_p[1].clamp(-10, 10))).to_event(1))
            z2_p = self.fc_z1_to_z2(z1).chunk(2, dim=-1)
            z2 = pyro.sample("z2", dist.Normal(z2_p[0], torch.exp(0.5 * z2_p[1].clamp(-10, 10))).to_event(1))
        return z1, z2
    
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        observations = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        conditioned = poutine.condition(self.model, data=observations)
        return poutine.trace(conditioned).get_trace().log_prob_sum()
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        x = context["input"].unsqueeze(0) if context["input"].dim() == 1 else context["input"]
        x = x.to(self._device)
        with torch.no_grad():
            guide_trace = poutine.trace(self.guide).get_trace({"input": x})
            z1 = guide_trace.nodes["z1"]["value"]
            y_p = self.fc_z1_to_y(z1).chunk(2, dim=-1)
            y = dist.Normal(y_p[0], torch.exp(0.5 * y_p[1])).sample((num_samples,))
        return {"y": y.squeeze(0)}
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        observations = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        if self.svi is None: self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        loss = self.svi.step(observations)
        self.loss_ema = loss if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema
    
    def batch_update(self, observations_batch: List[Dict[str, torch.Tensor]], learning_rate: float = 0.01):
        if not observations_batch: return
        inputs = torch.cat([obs["input"] for obs in observations_batch], dim=0)
        targets = torch.cat([obs["target"] for obs in observations_batch], dim=0)
        self.update_parameters({"input": inputs, "target": targets}, learning_rate)
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, RegressionPattern): return 1.0
        g1, g2 = self.causal_graph, other.causal_graph
        edges1, edges2 = set(g1.edges()), set(g2.edges())
        graph_dist = 1.0 - len(edges1 & edges2) / len(edges1 | edges2) if edges1 | edges2 else 0.0
        dim_dist = abs(self.z2_dim - other.z2_dim) / max(self.z2_dim, other.z2_dim, 1)
        return 0.7 * graph_dist + 0.3 * dim_dist
    
    def extract_causal_graph(self) -> nx.DiGraph: return self.causal_graph.copy()
    
    def compression_score(self, observations: Any) -> float:
        x = observations["input"] if isinstance(observations, dict) else observations
        x = x.to(self._device)
        with torch.no_grad():
            guide_trace = poutine.trace(self.guide).get_trace({"input": x})
            z2 = guide_trace.nodes["z2"]["value"]
        return min(1.0, z2.var(dim=0).mean().item() / 3.0)

    def surface_dependence(self, x_batch: torch.Tensor) -> float:
        if x_batch.dim() == 1: x_batch = x_batch.unsqueeze(0)
        x_batch = x_batch.to(self._device)
        with torch.no_grad():
            guide_trace = poutine.trace(self.guide).get_trace({"input": x_batch})
            z2 = guide_trace.nodes["z2"]["value"]
        surface = x_batch[:, 2:6]
        z2_centered = z2 - z2.mean(dim=0)
        surface_centered = surface - surface.mean(dim=0)
        try:
            ridge = 1e-5 * torch.eye(surface.shape[1], device=self._device)
            gram = surface_centered.T @ surface_centered + ridge
            coeffs = torch.linalg.solve(gram, surface_centered.T @ z2_centered)
            pred = surface_centered @ coeffs
            r2 = 1 - ((z2_centered - pred) ** 2).sum() / (z2_centered ** 2).sum()
            return float(r2.mean().item())
        except:
            return float(torch.corrcoef(torch.cat([z2.T, surface.T]))[:z2.shape[1], z2.shape[1]:].abs().mean().item())
