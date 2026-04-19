"""
tool_pattern.py - Pattern that wraps an external tool/API.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Callable, List
from .pattern import HPMPattern


class ToolPattern(HPMPattern):
    """
    Pattern encoded in an external tool or artefact.
    The pattern is a wrapper around an external function call.
    """
    def __init__(self, 
                 tool_fn: Callable,
                 input_keys: List[str],
                 output_key: str,
                 tool_name: str = "generic_tool",
                 cost: float = 0.1,
                 pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.tool_fn = tool_fn
        self.input_keys = input_keys
        self.output_key = output_key
        self.tool_name = tool_name
        self.cost = cost
        self.substrate_type = "tool"
        self.required_observation_keys = input_keys.copy()
        
        # Tool usage tracking
        self.usage_count = 0
        self.cache = {}  # simple memoization
        self.cache_hits = 0
        
        # Causal graph: inputs -> tool -> output
        self.causal_graph = nx.DiGraph()
        for key in input_keys:
            self.causal_graph.add_edge(key, "tool_output")
        self.causal_graph.add_node("tool")
        
        # Tool patterns have high base accuracy but cost penalty
        self.accuracy = 0.0  # Will be updated based on usage
        self.to(self._device)

    def to(self, device: torch.device):
        self._device = device
        return self
        
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Tool patterns are deterministic; return high log prob if output present."""
        if self.output_key not in observations:
            return torch.tensor(-10.0, device=self._device)
        
        # If we have target output, compare
        try:
            inputs = {k: observations[k] for k in self.input_keys if k in observations}
            output_pred = self._call_tool(inputs)
            if isinstance(output_pred, torch.Tensor):
                output_pred = output_pred.to(self._device)
                target = observations[self.output_key].to(self._device)
                diff = target - output_pred
                mse = (diff ** 2).mean()
                return -mse
            else:
                return torch.tensor(0.0, device=self._device)  # Neutral if non-tensor
        except Exception:
            return torch.tensor(-100.0, device=self._device)
    
    def _call_tool(self, inputs: Dict[str, Any]) -> Any:
        """Call the underlying tool function."""
        # Convert torch tensors to numpy/python types as needed
        converted = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    converted[k] = v.item()
                else:
                    converted[k] = v.cpu().numpy()
            else:
                converted[k] = v
        return self.tool_fn(**converted)
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Execute tool with given inputs."""
        self.usage_count += 1
        
        inputs = {k: context[k] for k in self.input_keys if k in context}
        cache_key = str(inputs)
        
        if cache_key in self.cache:
            self.cache_hits += 1
            outputs = [self.cache[cache_key] for _ in range(num_samples)]
        else:
            outputs = []
            for _ in range(num_samples):
                out = self._call_tool(inputs)
                outputs.append(out)
            self.cache[cache_key] = outputs[0]
        
        # Convert to tensor if needed
        processed_outputs = []
        for o in outputs:
            if not isinstance(o, torch.Tensor):
                if isinstance(o, (list, np.ndarray)):
                    processed_outputs.append(torch.tensor(o, dtype=torch.float32, device=self._device))
                elif isinstance(o, (int, float)):
                    processed_outputs.append(torch.tensor([o], dtype=torch.float32, device=self._device))
                else:
                    processed_outputs.append(o)
            else:
                processed_outputs.append(o.to(self._device))
        
        if isinstance(processed_outputs[0], torch.Tensor):
            result_val = torch.stack(processed_outputs) if num_samples > 1 else processed_outputs[0]
        else:
            result_val = processed_outputs[0]
            
        result = {self.output_key: result_val}
        return result
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Intervention on tool: modify inputs or override output."""
        if self.output_key in intervention:
            return {self.output_key: intervention[self.output_key]}
        modified_context = context.copy()
        modified_context.update(intervention)
        return self.sample(modified_context, num_samples=1)
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """Tool patterns don't learn parameters; update accuracy based on usage."""
        with torch.no_grad():
            logp = self.log_prob(observations)
            loss = -logp.item()
        
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        
        # Accuracy includes cost penalty
        self.accuracy = -self.loss_ema - self.cost
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, ToolPattern):
            return 1.0
        name_diff = 0.0 if self.tool_name == other.tool_name else 0.5
        input_diff = len(set(self.input_keys) ^ set(other.input_keys)) / max(len(self.input_keys), len(other.input_keys), 1)
        return (name_diff + input_diff) / 2.0
    
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
