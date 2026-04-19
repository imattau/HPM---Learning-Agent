from pattern import HPMPattern
import torch
import networkx as nx
import inspect
from typing import Dict, Any, Callable, Optional, List
import numpy as np

class SymbolicPattern(HPMPattern):
    """
    A compiled, discrete representation of a pattern.
    Uses explicit Python functions or SymPy for symbolic algebra.
    """
    def __init__(self, 
                 forward_fn: Callable,
                 inverse_fn: Optional[Callable] = None,
                 causal_graph: Optional[nx.DiGraph] = None,
                 pattern_id: Optional[str] = None,
                 required_keys: Optional[List[str]] = None,
                 output_key: str = "x"):
        super().__init__(pattern_id)
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn
        self.causal_graph = causal_graph or nx.DiGraph()
        self.substrate_type = "symbolic"
        self.required_observation_keys = required_keys or []
        self.output_key = output_key
        
        # For logging likelihood (assume Gaussian noise model)
        self.noise_std = 0.1
        self.to(self._device)

    def to(self, device: torch.device):
        self._device = device
        return self
        
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute log prob assuming Gaussian observation noise."""
        observations = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        x_pred = self.forward_fn(observations)
        if not isinstance(x_pred, torch.Tensor):
            if isinstance(x_pred, (int, float, list, np.ndarray)):
                x_pred = torch.tensor(x_pred, dtype=torch.float32, device=self._device)
            else:
                return torch.tensor(0.0, device=self._device) # Non-numeric output
        
        if self.output_key not in observations:
            return torch.tensor(-1.0, device=self._device)
            
        x_true = observations[self.output_key]
        mse = torch.mean((x_true - x_pred) ** 2)
        # Gaussian log likelihood (ignoring constant)
        n = x_true.numel()
        return -0.5 * n * torch.log(torch.tensor(2 * np.pi * self.noise_std**2, device=self._device)) - mse / (2 * self.noise_std**2)
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Generate samples by running forward function."""
        context = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in context.items()}
        samples = []
        for _ in range(num_samples):
            out = self.forward_fn(context)
            if not isinstance(out, torch.Tensor):
                if isinstance(out, (int, float, list, np.ndarray)):
                    out = torch.tensor(out, dtype=torch.float32, device=self._device)
                else:
                    # Keep as is (e.g. string tool name)
                    pass
            samples.append(out)
        
        result_val = torch.stack(samples) if num_samples > 1 and isinstance(samples[0], torch.Tensor) else samples[0]
        return {self.output_key: result_val}
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Modify the function call to respect intervention."""
        # For symbolic patterns, intervention is modifying inputs to forward_fn
        modified_context = context.copy()
        modified_context.update(intervention)
        result = self.forward_fn(modified_context)
        return {"x": result if isinstance(result, torch.Tensor) else torch.tensor(result)}
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """
        Symbolic patterns don't update via gradient descent.
        Instead, they may be refined by symbolic regression (optional).
        Here we just update the loss EMA based on prediction error.
        """
        with torch.no_grad():
            x_pred = self.forward_fn(observations)
            if not isinstance(x_pred, torch.Tensor):
                x_pred = torch.tensor(x_pred)
            loss = torch.mean((observations["x"] - x_pred) ** 2).item()
            
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema
        
    def structural_distance(self, other: 'HPMPattern') -> float:
        """Compare symbolic expressions using tree edit distance."""
        if isinstance(other, SymbolicPattern):
            try:
                import inspect
                src1 = inspect.getsource(self.forward_fn)
                src2 = inspect.getsource(other.forward_fn)
                import Levenshtein
                return 1.0 - Levenshtein.ratio(src1, src2)
            except:
                return 0.5
        return 1.0
    
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
