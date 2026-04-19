from pattern import HPMPattern
import torch
import numpy as np
from typing import Dict, Any, Optional, Callable
import networkx as nx
import uuid

class ToolPattern(HPMPattern):
    """
    Pattern encoded in an external tool or artefact (e.g., calculator, notebook, software).
    The pattern is a wrapper around an external function call.
    """
    def __init__(self, 
                 tool_fn: Callable,
                 input_keys: list,
                 output_key: str,
                 tool_name: str = "generic_tool",
                 pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.tool_fn = tool_fn
        self.input_keys = input_keys
        self.output_key = output_key
        self.tool_name = tool_name
        self.substrate_type = "tool"
        
        # Tool usage history for adaptation
        self.usage_count = 0
        self.cache = {}  # memoization
        
        # Causal graph: inputs -> tool -> output
        self.causal_graph = nx.DiGraph()
        for key in input_keys:
            self.causal_graph.add_edge(key, "tool_output")
        self.causal_graph.add_node("tool")
        
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Tool patterns are deterministic; we assign high probability if output matches.
        """
        if self.output_key not in observations:
            return torch.tensor(-10.0)  # penalty for missing output
        
        inputs = {k: observations[k] for k in self.input_keys if k in observations}
        try:
            output_pred = self.tool_fn(**inputs)
            if isinstance(output_pred, torch.Tensor):
                diff = observations[self.output_key] - output_pred
                mse = (diff**2).mean()
                return -mse  # log prob proxy
            else:
                return torch.tensor(-1.0)  # mismatch
        except:
            return torch.tensor(-100.0)  # error
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Execute tool with given inputs."""
        inputs = {k: context[k] for k in self.input_keys if k in context}
        # Check cache first
        cache_key = str(inputs)
        if cache_key in self.cache:
            outputs = [self.cache[cache_key] for _ in range(num_samples)]
        else:
            outputs = []
            for _ in range(num_samples):
                out = self.tool_fn(**inputs)
                outputs.append(out)
            self.cache[cache_key] = outputs[0]
        
        # Convert to tensor if needed
        if not isinstance(outputs[0], torch.Tensor):
            outputs = [torch.tensor(o, dtype=torch.float32) for o in outputs]
        return {self.output_key: torch.stack(outputs)}
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Intervention on tool is modifying inputs or replacing function."""
        modified_context = context.copy()
        modified_context.update(intervention)
        return self.sample(modified_context, num_samples=1)
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """Tool patterns learn by caching successful outputs."""
        self.usage_count += 1
        # Compute loss
        with torch.no_grad():
            logp = self.log_prob(observations)
            loss = -logp.item()
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema
    
    def structural_distance(self, other: 'HPMPattern') -> float:
        if not isinstance(other, ToolPattern):
            return 1.0
        # Compare tool name and input structure
        name_diff = 0.0 if self.tool_name == other.tool_name else 0.5
        input_diff = len(set(self.input_keys) ^ set(other.input_keys)) / max(len(self.input_keys), len(other.input_keys), 1)
        return (name_diff + input_diff) / 2.0
    
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
