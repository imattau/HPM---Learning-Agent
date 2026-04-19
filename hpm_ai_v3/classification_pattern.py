import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import networkx as nx
from typing import Dict, Any, Optional, List
from .pattern import HPMPattern

class ClassificationPattern(HPMPattern):
    def __init__(self, input_dim: int = 784, num_classes: int = 10, z1_dim: int = 64, z2_dim: int = 16, pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.required_observation_keys = ["input", "target"]
        self.substrate_type = "neural"
        self.sparsity_lambda = 0.01
        
        self.fc_x_to_z1 = nn.Linear(input_dim, z1_dim * 2)
        self.z2_loc = nn.Parameter(torch.zeros(z2_dim))
        self.z2_scale = nn.Parameter(torch.ones(z2_dim))
        self.fc_z2_to_z1 = nn.Linear(z2_dim, z1_dim * 2)
        self.fc_z1_to_logits = nn.Linear(z1_dim, num_classes)
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
        self.fc_z1_to_logits.to(device)
        self.fc_z1_to_z2.to(device)
        return self

    def parameters(self):
        return (list(self.fc_x_to_z1.parameters()) + list(self.fc_z2_to_z1.parameters()) +
                list(self.fc_z1_to_logits.parameters()) + list(self.fc_z1_to_z2.parameters()) +
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
            logits = self.fc_z1_to_logits(z1)
            y = pyro.sample("y", dist.Categorical(logits=logits),
                            obs=observations["target"].long().squeeze() if observations and "target" in observations else None)
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
            logits = self.fc_z1_to_logits(z1)
            probs = torch.softmax(logits, dim=-1)
            y = dist.Categorical(probs).sample((num_samples,))
        return {"y": y.squeeze(0), "logits": logits, "probs": probs}

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.sample({"input": x})
            return out["probs"].argmax(dim=-1)

    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        observations = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        if self.svi is None: self.svi = pyro.infer.SVI(self.model, self.guide, self.optimizer, loss=pyro.infer.Trace_ELBO())
        loss = self.svi.step(observations)
        self.loss_ema = loss if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema

    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, ClassificationPattern): return 1.0
        g1, g2 = self.causal_graph, other.causal_graph
        edges1, edges2 = set(g1.edges()), set(g2.edges())
        graph_dist = 1.0 - len(edges1 & edges2) / len(edges1 | edges2) if edges1 | edges2 else 0.0
        dim_dist = abs(self.z1_dim - other.z1_dim) / max(self.z1_dim, other.z1_dim, 1)
        return 0.7 * graph_dist + 0.3 * dim_dist

    def extract_causal_graph(self) -> nx.DiGraph: return self.causal_graph.copy()
    def intervene(self, intervention, context): raise NotImplementedError
