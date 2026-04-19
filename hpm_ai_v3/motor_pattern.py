from pattern import HPMPattern
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import networkx as nx

class MotorPattern(HPMPattern):
    """
    Pattern encoded in bodily states and motor routines.
    Represents a trajectory generator or policy for action.
    """
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 horizon: int = 10,
                 pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.substrate_type = "motor"
        
        # Simple linear-Gaussian policy: action = A * state + b + noise
        self.A = torch.randn(action_dim, state_dim) * 0.1
        self.b = torch.zeros(action_dim)
        self.noise_std = 0.1
        
        # For trajectory generation
        self.transition_matrix = torch.eye(state_dim)  # default identity
        
        # Causal graph: state_t -> action_t -> state_{t+1}
        self.causal_graph = nx.DiGraph()
        self.causal_graph.add_edge("state", "action")
        self.causal_graph.add_edge("action", "next_state")
        
    def model(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate action and next state."""
        action_mean = state @ self.A.T + self.b
        action = action_mean + torch.randn_like(action_mean) * self.noise_std
        next_state = state @ self.transition_matrix.T  # simplified
        return action, next_state
    
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute log prob of observed action given state.
        observations must contain 'state' and 'action'.
        """
        state = observations["state"]
        action = observations["action"]
        action_mean = state @ self.A.T + self.b
        # Gaussian log likelihood
        diff = action - action_mean
        log_prob = -0.5 * (diff**2).sum(dim=-1) / (self.noise_std**2)
        log_prob -= 0.5 * action.shape[-1] * np.log(2 * np.pi * self.noise_std**2)
        return log_prob.mean()
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Generate trajectories."""
        state = context.get("state", torch.zeros(num_samples, self.state_dim))
        actions = []
        states = [state]
        for _ in range(self.horizon):
            action, next_state = self.model(states[-1])
            actions.append(action)
            states.append(next_state)
        return {
            "states": torch.stack(states, dim=1),
            "actions": torch.stack(actions, dim=1)
        }
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Intervention on action policy (e.g., force action = value).
        """
        state = context.get("state", torch.zeros(1, self.state_dim))
        if "action" in intervention:
            action = intervention["action"]
        else:
            action, _ = self.model(state)
        if "A" in intervention:
            self.A.data = intervention["A"]
        return {"action": action, "state": state}
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """Online linear regression update."""
        state = observations["state"]
        action = observations["action"]
        # Simple gradient step on MSE
        action_pred = state @ self.A.T + self.b
        error = action - action_pred
        grad_A = -2 * torch.einsum('bi,bj->ij', error, state) / state.shape[0]
        grad_b = -2 * error.mean(dim=0)
        with torch.no_grad():
            self.A -= learning_rate * grad_A
            self.b -= learning_rate * grad_b
        
        # Update loss EMA
        loss = (error**2).mean().item()
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, MotorPattern):
            return 1.0
        # Compare policy matrices
        diff_A = torch.norm(self.A - other.A).item()
        diff_b = torch.norm(self.b - other.b).item()
        return min(1.0, (diff_A + diff_b) / 10.0)
    
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
