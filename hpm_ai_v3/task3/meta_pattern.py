import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from typing import Dict, Any, List, Optional
from collections import deque
from pattern import HPMPattern

class MetaPattern(HPMPattern):
    def __init__(self, history_dim: int = 4, hidden_dim: int = 16, pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.history_dim = history_dim
        self.hidden_dim = hidden_dim
        self.required_observation_keys = ["history", "reward"]
        self.substrate_type = "neural"
        
        self.net = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.baseline = nn.Linear(history_dim, 1)
        self.optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.baseline.parameters()), lr=0.001)

    def forward(self, history: torch.Tensor):
        out = self.net(history)
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar
    
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.tensor(0.0)
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        history = context["history"]
        if history.dim() == 1: history = history.unsqueeze(0)
        with torch.no_grad():
            mean, logvar = self.forward(history)
            delta_cur = dist.Normal(mean, torch.exp(0.5 * logvar)).sample((num_samples,))
        return {"delta_cur": delta_cur.squeeze(0)}
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        history = observations["history"]
        reward = observations["reward"]
        mean, logvar = self.forward(history)
        std = torch.exp(0.5 * logvar)
        delta_cur = observations.get("delta_cur_taken", mean)
        
        log_prob = torch.distributions.Normal(mean, std).log_prob(delta_cur).sum(dim=-1)
        baseline = self.baseline(history).squeeze(-1)
        advantage = reward - baseline
        
        pg_loss = -(log_prob * advantage.detach()).mean()
        bl_loss = nn.MSELoss()(baseline, reward)
        loss = pg_loss + 0.5 * bl_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_ema = loss.item() if self.loss_ema is None else 0.9*self.loss_ema + 0.1*loss.item()
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        return 1.0
    def extract_causal_graph(self):
        import networkx as nx
        g = nx.DiGraph()
        g.add_edge("history", "delta_cur")
        return g
    def intervene(self, intervention, context): return self.sample(context)
