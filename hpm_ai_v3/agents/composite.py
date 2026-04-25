"""
composite_agent_pattern.py - Pattern that chains multiple AgentPatterns.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
from ..pattern import HPMPattern
from .base import AgentPattern


class CompositeAgentPattern(HPMPattern):
    """
    Wraps a sequence of HPM AgentPatterns into a single composite pipeline.
    This enables agents to delegate tasks to a fixed chain of specialists.
    """
    def __init__(self, 
                 patterns: List[AgentPattern], 
                 pattern_id: Optional[str] = None,
                 cost_aggregation: str = 'sum'):
        """
        Args:
            patterns: Ordered list of AgentPatterns to execute in sequence.
            pattern_id: Optional unique identifier.
            cost_aggregation: 'sum' or 'min' or 'max' for combined cost.
        """
        super().__init__(pattern_id or f"agent_pipeline_{'_'.join(p.agent_name for p in patterns)}")
        self.patterns = patterns
        self.substrate_type = "composite_agent"
        self.cost_aggregation = cost_aggregation
        
        # Compute combined cost
        if cost_aggregation == 'sum':
            self.cost = sum(p.cost for p in patterns)
        elif cost_aggregation == 'min':
            self.cost = min(p.cost for p in patterns)
        else:  # 'max'
            self.cost = max(p.cost for p in patterns)
        
        # Collect all required keys that aren't produced within the pipeline
        produced_keys = set()
        required_keys = set()
        for pat in patterns:
            for inp in pat.required_observation_keys:
                if inp not in produced_keys:
                    required_keys.add(inp)
            produced_keys.add(pat.output_key)
            
        self.required_observation_keys = list(required_keys)
        self.output_key = patterns[-1].output_key
        
        # Build causal graph
        self.causal_graph = nx.DiGraph()
        for i, pat in enumerate(patterns):
            self.causal_graph.add_node(pat.agent_name, type='agent')
            for inp in pat.required_observation_keys:
                self.causal_graph.add_edge(inp, pat.agent_name)
            self.causal_graph.add_edge(pat.agent_name, pat.output_key)
            
    @property
    def tool_name(self) -> str:
        """Required property for population interface."""
        return f"agent_composite:{self.id}"

    def get_sequence(self) -> List[str]:
        """Return list of agent names in order."""
        return [p.agent_name for p in self.patterns]
        
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Approximate log prob based on end-to-end matching.
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
        Execute the pipeline of agents in sequence.
        Intermediate results are added to the context.
        """
        current_context = context.copy()
        
        for pat in self.patterns:
            # Ensure required inputs for this agent are available
            out = pat.sample(current_context, num_samples=num_samples)
            current_context.update(out)
            
        return {self.output_key: current_context[self.output_key]}
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Intervention on the composite can override any intermediate or final result.
        """
        modified_context = context.copy()
        modified_context.update(intervention)
        return self.sample(modified_context)
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """
        Update performance metrics (loss_ema and accuracy).
        The composite doesn't have its own parameters, but its 'utility' changes.
        """
        with torch.no_grad():
            logp = self.log_prob(observations)
            loss = -logp.item()
            
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
            
        # Accuracy is the negative loss plus a penalty for complexity (cost)
        self.accuracy = -self.loss_ema - self.cost
        
    def structural_distance(self, other: 'HPMPattern') -> float:
        """
        Distance based on overlap in agent sequences.
        """
        if not isinstance(other, CompositeAgentPattern):
            return 1.0
            
        names1 = [p.agent_name for p in self.patterns]
        names2 = [p.agent_name for p in other.patterns]
        
        if names1 == names2:
            return 0.0
            
        # Jaccard similarity of sequences
        s1 = set(names1)
        s2 = set(names2)
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        
        return 1.0 - (intersection / union)
        
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
    
    def to(self, device: torch.device):
        self._device = device
        for p in self.patterns:
            p.to(device)
        return self
