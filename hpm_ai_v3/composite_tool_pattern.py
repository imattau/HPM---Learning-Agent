"""
composite_tool_pattern.py - Pattern that chains multiple ToolPatterns.
Fully compatible with existing HPM base classes.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
from .pattern import HPMPattern
from .tool_pattern import ToolPattern


class CompositeToolPattern(HPMPattern):
    """
    A pattern that composes multiple ToolPatterns into a pipeline.
    The output of each tool becomes input to the next.
    """
    def __init__(self, 
                 patterns: List[ToolPattern], 
                 pattern_id: Optional[str] = None,
                 cost_aggregation: str = 'sum'):
        """
        Args:
            patterns: Ordered list of ToolPatterns to execute in sequence.
            pattern_id: Optional unique identifier.
            cost_aggregation: 'sum' or 'min' or 'max' for combined cost.
        """
        super().__init__(pattern_id or f"pipeline_{'_'.join(p.tool_name for p in patterns)}")
        self.patterns = patterns
        self.substrate_type = "composite_tool"
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
            for inp in pat.input_keys:
                if inp not in produced_keys:
                    required_keys.add(inp)
            produced_keys.add(pat.output_key)
            
        self.required_observation_keys = list(required_keys)
        # Output key is the last tool's output key
        self.output_key = patterns[-1].output_key
        
        # Build causal graph
        self.causal_graph = nx.DiGraph()
        prev_output = None
        for i, pat in enumerate(self.patterns):
            node_name = f"tool_{i}_{pat.tool_name}"
            self.causal_graph.add_node(node_name, tool=pat.tool_name)
            for inp in pat.input_keys:
                if i == 0:
                    self.causal_graph.add_edge(inp, node_name)
                else:
                    # In a simple pipeline, we assume it takes previous tool's output
                    # but real tools might need specific keys. 
                    # For simplicity, we assume linear chain if not specified.
                    self.causal_graph.add_edge(prev_output, node_name)
            prev_output = pat.output_key
        self.causal_graph.add_edge(prev_output, "final_output")
        
        # Cache for repeated calls
        self.cache = {}
        self.to(self._device)

    def to(self, device: torch.device):
        """Move component patterns to device if needed."""
        self._device = device
        for pat in self.patterns:
            pat.to(device)
        return self
        
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        observations = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        if self.output_key not in observations:
            return torch.tensor(-10.0, device=self._device)
        
        # Execute pipeline to get prediction
        result = self.sample(observations, num_samples=1)
        pred = result[self.output_key]
        target = observations[self.output_key]
        
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            mse = ((pred - target) ** 2).mean()
            return -mse
        return torch.tensor(0.0, device=self._device)
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        current_context = context.copy()
        for pat in self.patterns:
            out = pat.sample(current_context, num_samples=num_samples)
            current_context.update(out)
        
        final_output = current_context[self.output_key]
        return {self.output_key: final_output}
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        modified_context = context.copy()
        modified_context.update(intervention)
        return self.sample(modified_context, num_samples=1)
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        with torch.no_grad():
            logp = self.log_prob(observations)
            loss = -logp.item()
        
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema - self.cost
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, CompositeToolPattern):
            return 1.0
        seq1 = [p.tool_name for p in self.patterns]
        seq2 = [p.tool_name for p in other.patterns]
        max_len = max(len(seq1), len(seq2))
        if max_len == 0: return 0.0
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return 1.0 - (matches / max_len)
    
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
