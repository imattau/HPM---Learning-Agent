"""
agent_pattern.py - Pattern that wraps another HPM agent as a callable substrate.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
from ..pattern import HPMPattern


class AgentPattern(HPMPattern):
    """
    Wraps another HPM agent as a callable pattern.
    This enables recursive agent composition and delegation.
    """
    def __init__(self, 
                 agent: Any,
                 agent_name: str = "unnamed_agent",
                 cost: float = 0.5,
                 description: str = "",
                 pattern_id: Optional[str] = None):
        super().__init__(pattern_id or f"agent_{agent_name}")
        self.agent = agent
        self.agent_name = agent_name
        self.cost = cost
        self.description = description
        self.substrate_type = "agent"
        
        # Determine interface by inspecting agent
        self.required_observation_keys = getattr(agent, 'required_observation_keys', ["input"])
        self.output_key = getattr(agent, 'output_key', "output")
        
        # Causal graph
        self.causal_graph = nx.DiGraph()
        for key in self.required_observation_keys:
            self.causal_graph.add_edge(key, f"agent_{agent_name}")
        self.causal_graph.add_edge(f"agent_{agent_name}", self.output_key)
        
        # Cache for repeated calls
        self.cache = {}
        
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Log prob based on agent's output matching target if present.
        """
        if self.output_key not in observations:
            return torch.tensor(-10.0, device=self._device)
        
        try:
            result = self.sample(observations, num_samples=1)
            pred = result[self.output_key]
            target = observations[self.output_key]
            if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
                mse = ((pred.to(self._device) - target.to(self._device)) ** 2).mean()
                return -mse
            return torch.tensor(0.0, device=self._device)
        except Exception:
            return torch.tensor(-100.0, device=self._device)
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Invoke the wrapped agent.
        """
        # Simple caching for identical contexts
        # Note: torch tensors aren't hashable, so this is just a demo-level cache
        # In practice, use image hashes or similar
        cache_key = None
        try:
            cache_key = str(sorted([(k, str(v)) for k, v in context.items()]))
            if cache_key in self.cache and num_samples == 1:
                return self.cache[cache_key]
        except:
            pass
        
        # Determine which method to call on the agent
        if hasattr(self.agent, 'process'):
            result = self.agent.process(context)
        elif hasattr(self.agent, 'predict'):
            result = self.agent.predict(context)
        elif hasattr(self.agent, 'step'):
            # In AugmentedHPMAgent, step returns tool/reward.
            # We might need a more unified 'invoke' method.
            if hasattr(self.agent, 'invoke'):
                result = self.agent.invoke(context)
            else:
                result = self.agent.step(context)
        else:
            raise AttributeError(f"Agent {self.agent_name} has no callable interface.")
        
        # Ensure result is a dict with the expected output key
        if not isinstance(result, dict):
            result = {self.output_key: result}
        elif self.output_key not in result:
            # If the agent returns a dict but with a different key, try to adapt
            first_key = next(iter(result))
            result = {self.output_key: result[first_key]}
        
        # Convert to tensor if needed (only for numeric types)
        for k, v in result.items():
            if not isinstance(v, torch.Tensor):
                try:
                    if isinstance(v, (int, float, list, np.ndarray)):
                        result[k] = torch.tensor(v, dtype=torch.float32, device=self._device)
                    else:
                        # Keep as is (e.g. dict, string)
                        pass
                except (ValueError, TypeError):
                    pass
            else:
                result[k] = v.to(self._device)
        
        if num_samples == 1 and cache_key:
            self.cache[cache_key] = result
        return result
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Intervention can override agent's output."""
        if self.output_key in intervention:
            val = intervention[self.output_key]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32, device=self._device)
            return {self.output_key: val}
        return self.sample(context)
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """
        AgentPattern doesn't learn parameters itself; we track loss for evaluators.
        """
        with torch.no_grad():
            logp = self.log_prob(observations)
            loss = -logp.item()
        
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema - self.cost
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, AgentPattern):
            return 1.0
        # Distance based on agent name
        return 0.0 if self.agent_name == other.agent_name else 0.8
    
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
    
    def to(self, device: torch.device):
        self._device = device
        if hasattr(self.agent, 'to'):
            self.agent.to(device)
