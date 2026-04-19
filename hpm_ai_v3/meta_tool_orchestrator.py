"""
meta_tool_orchestrator.py - Meta-pattern that learns to select tools dynamically.
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .pattern import HPMPattern


class MetaToolOrchestrator(HPMPattern):
    """
    Meta-pattern that learns to select which tools to invoke based on context.
    Outputs a probability distribution over available tools.
    """
    def __init__(self, 
                 available_tools: List[str],
                 context_feature_dim: int = 16,
                 hidden_dim: int = 32,
                 pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.available_tools = available_tools
        self.num_tools = len(available_tools)
        self.context_feature_dim = context_feature_dim
        self.hidden_dim = hidden_dim
        self.required_observation_keys = ["context_features", "reward"]
        self.substrate_type = "neural"
        
        # Policy network: context features -> tool selection logits
        self.policy_net = nn.Sequential(
            nn.Linear(context_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_tools)
        )
        
        # Baseline for variance reduction
        self.baseline_net = nn.Linear(context_feature_dim, 1)
        
        self.optimizer = Adam({"lr": 0.001})
        self.svi = None
        
        # Tool usage history for learning
        self.tool_usage_history: List[Tuple[str, float]] = []
        self.to(self._device)

    def to(self, device: torch.device):
        self._device = device
        self.policy_net.to(device)
        self.baseline_net.to(device)
        return self
    
    def parameters(self):
        return list(self.policy_net.parameters()) + list(self.baseline_net.parameters())
    
    def model(self, observations: Optional[Dict[str, torch.Tensor]] = None):
        if observations:
            observations = {k: v.to(self._device) for k, v in observations.items()}
        batch_size = observations["context_features"].shape[0] if observations else 1
        with pyro.plate("batch", batch_size):
            features = observations["context_features"] if observations else torch.zeros(batch_size, self.context_feature_dim, device=self._device)
            logits = self.policy_net(features)
            tool_idx = pyro.sample("tool_idx", dist.Categorical(logits=logits))
            if observations and "reward" in observations:
                pyro.sample("reward", dist.Delta(tool_idx.float()), obs=observations["reward"])
        return tool_idx
    
    def guide(self, observations: Dict[str, torch.Tensor]):
        observations = {k: v.to(self._device) for k, v in observations.items()}
        features = observations["context_features"]
        logits = self.policy_net(features)
        tool_idx = pyro.sample("tool_idx", dist.Categorical(logits=logits))
        return tool_idx
    
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.tensor(0.0, device=self._device)
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        features = context["context_features"]
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = features.to(self._device)
        with torch.no_grad():
            logits = self.policy_net(features)
            probs = torch.softmax(logits, dim=-1)
            if num_samples == 1:
                tool_idx = torch.multinomial(probs, 1).squeeze(-1)
            else:
                tool_idx = torch.multinomial(probs, num_samples, replacement=True)
        
        if num_samples == 1:
            selected_tool = self.available_tools[tool_idx.item()]
        else:
            selected_tool = [self.available_tools[i] for i in tool_idx.tolist()]
            
        return {"tool_idx": tool_idx, "probs": probs, "selected_tool": selected_tool}
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """
        Policy gradient update using REINFORCE with baseline.
        """
        observations = {k: v.to(self._device) for k, v in observations.items()}
        features = observations["context_features"]
        reward = observations["reward"]
        tool_idx_taken = observations.get("tool_idx_taken")
        if tool_idx_taken is None:
            return
        
        logits = self.policy_net(features)
        dist_cat = torch.distributions.Categorical(logits=logits)
        log_prob = dist_cat.log_prob(tool_idx_taken)
        
        baseline = self.baseline_net(features).squeeze(-1)
        advantage = reward - baseline.detach()
        
        pg_loss = -(log_prob * advantage).mean()
        baseline_loss = nn.MSELoss()(baseline, reward)
        
        total_loss = pg_loss + 0.5 * baseline_loss
        
        # Using pyro's optimizer via SVI or manual? The user's code had manual backward but self.optimizer was Adam from pyro.
        # Pyro's Adam expects a dictionary or a list of params? Actually pyro.optim.Adam is a wrapper.
        # Let's use standard torch optimizer for simplicity if we are doing manual backward.
        # But wait, self.optimizer is pyro.optim.Adam in the user code.
        
        # User's code:
        # self.optimizer.zero_grad()
        # total_loss.backward()
        # self.optimizer.step()
        
        # This only works if self.optimizer is a torch.optim.Optimizer.
        # Pyro.optim.Adam is NOT a torch optimizer, it's a wrapper that creates them on the fly.
        
        # I'll fix the optimizer to be a standard torch optimizer.
        if not hasattr(self, 'torch_optimizer'):
            self.torch_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            
        self.torch_optimizer.zero_grad()
        total_loss.backward()
        self.torch_optimizer.step()
        
        loss_val = total_loss.item()
        if self.loss_ema is None:
            self.loss_ema = loss_val
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss_val
        self.accuracy = -self.loss_ema
        
        # Record usage
        tool_name = self.available_tools[tool_idx_taken.item()]
        self.tool_usage_history.append((tool_name, reward.item()))
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, MetaToolOrchestrator):
            return 1.0
        p1 = torch.cat([p.flatten() for p in self.policy_net.parameters()])
        p2 = torch.cat([p.flatten() for p in other.policy_net.parameters()])
        return float(torch.norm(p1 - p2).item() / (torch.norm(p1) + torch.norm(p2) + 1e-8))
    
    def extract_causal_graph(self):
        import networkx as nx
        g = nx.DiGraph()
        g.add_edge("context_features", "tool_selection")
        return g
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "tool_idx" in intervention:
            return {"tool_idx": intervention["tool_idx"]}
        return self.sample(context)
