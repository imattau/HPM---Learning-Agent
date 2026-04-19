import torch
import numpy as np
from typing import List, Tuple, Optional
from causal_pattern import CausalPattern
from symbolic_pattern import SymbolicPattern
from motor_pattern import MotorPattern
from pattern import HPMPattern
import sympy as sp
from sklearn.linear_model import LinearRegression

class SubstrateCompiler:
    """
    Attempts to compile neural or motor patterns into symbolic expressions.
    """
    def __init__(self, 
                 compression_threshold: float = 0.8,
                 accuracy_tolerance: float = 0.1):
        self.compression_threshold = compression_threshold
        self.accuracy_tolerance = accuracy_tolerance
        
    def should_compile(self, pattern: HPMPattern) -> bool:
        """Decision rule: high compression and stable accuracy."""
        if pattern.substrate_type == "symbolic":
            return False
        if isinstance(pattern, MotorPattern):
            # Compile when policy is stable (low loss variance)
            return pattern.loss_ema is not None and pattern.loss_ema < 0.5
        if isinstance(pattern, CausalPattern):
            # Check if pattern has high compression (MI)
            mi = pattern.compression_score(torch.randn(100, pattern.input_dim))  # Placeholder
            return mi > self.compression_threshold and pattern.loss_ema is not None and pattern.loss_ema < 1.0
        return False
    
    def compile_motor_to_symbolic(self, motor_pattern: MotorPattern) -> Optional[SymbolicPattern]:
        """
        Compile a motor pattern into a symbolic policy function.
        """
        A = motor_pattern.A.detach().numpy()
        b = motor_pattern.b.detach().numpy()
        
        def policy_fn(context):
            state = context.get("state")
            if state is None:
                return np.zeros(motor_pattern.action_dim)
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            action = state @ A.T + b
            return torch.tensor(action, dtype=torch.float32)
        
        sym_pat = SymbolicPattern(policy_fn, causal_graph=motor_pattern.causal_graph)
        sym_pat.substrate_type = "symbolic_from_motor"
        sym_pat.accuracy = motor_pattern.accuracy
        return sym_pat
        
    def compile_to_symbolic(self, pattern: CausalPattern) -> Optional[SymbolicPattern]:
        """
        Extract a symbolic function from the neural pattern.
        """
        # Generate samples from pattern to learn a symbolic approximation
        num_samples = 1000
        z2_samples = torch.randn(num_samples, pattern.z2_dim) * 2  # Vary abstract latent
        x_samples = []
        
        for i in range(num_samples):
            sample = pattern.sample({"z2": z2_samples[i:i+1]}, num_samples=1)
            x_samples.append(sample["x"].squeeze(0))
        x_samples = torch.stack(x_samples)
        
        # Linear regression for simplicity
        X = z2_samples.numpy()
        y = x_samples.mean(dim=1).numpy()  # Use mean of output as target
        
        model = LinearRegression()
        model.fit(X, y)
        
        coef = model.coef_
        intercept = model.intercept_
        
        def forward_fn(context):
            z2 = context.get("z2")
            if z2 is None:
                z2 = np.zeros(pattern.z2_dim)
            if isinstance(z2, torch.Tensor):
                z2 = z2.numpy()
            pred = intercept + np.dot(z2, coef)
            return torch.tensor(pred, dtype=torch.float32)
        
        # Verify accuracy
        test_x = x_samples[:10]
        test_pred = torch.stack([forward_fn({"z2": z2_samples[i]}) for i in range(10)])
        mse = torch.mean((test_x - test_pred) ** 2).item()
        
        if mse > pattern.loss_ema + self.accuracy_tolerance:
            return None  # Not accurate enough
        
        # Create symbolic pattern
        sym_pattern = SymbolicPattern(forward_fn, causal_graph=pattern.extract_causal_graph())
        sym_pattern.loss_ema = mse
        sym_pattern.accuracy = -mse
        sym_pattern.compilation_count = pattern.compilation_count + 1
        
        return sym_pattern
