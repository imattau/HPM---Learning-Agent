# HPM AI v3 - Production-Grade Architecture Design

## Objective
Implement the HPM AI v3 core architecture, addressing the three critical gaps: Causal Representation, Substrate Shifting, and Institutional Truth (Pattern Fields). The implementation uses Pyro for probabilistic programming, a dual-memory system for substrate shifting, and Ray for multi-agent pattern field simulation.

## Core Mandates
- **Causal Representation**: Use `CausalPattern` with Pyro, explicit latent hierarchy, and `intervene()` method.
- **Substrate Shifting**: `SubstrateCompiler` monitors compression and converts neural patterns to `SymbolicPattern`.
- **Institutional Truth**: `PatternField` with multi-agent replication protocol using Ray.
- **Evaluator Pluralism**: Separate evaluators (epistemic, affective, curiosity, coherence, social, insight).
- **Creativity via Recombination**: Structural recombination with insight boost.
- **Pattern Density & Stickiness**: Compute and utilize pattern density for weight persistence.
- **Bodily & Artefact Substrates**: Implement `MotorPattern` and `ToolPattern`.
- **Forgetting & Decay**: Age-based decay and structural interference.

## Implementation Steps

### 1. Core Pattern Representation (`hpm_ai_v3/pattern.py`)
Abstract base class defining the standard interface for all patterns.

```python
from abc import ABC, abstractmethod
import pyro
import pyro.distributions as dist
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import uuid
import time
import networkx as nx

class HPMPattern(ABC):
    """
    Abstract base for all HPM patterns.
    Patterns are causal generative models with explicit latent hierarchy.
    """
    def __init__(self, pattern_id: Optional[str] = None):
        self.id = pattern_id or str(uuid.uuid4())[:8]
        self.weight = 0.01  # Population weight
        
        # Evaluator state
        self.loss_ema: Optional[float] = None
        self.accuracy: float = 0.0
        self.affective_score: float = 0.0
        self.social_score: float = 0.0
        self.curiosity_reward: float = 0.0
        self.coherence_score: float = 0.0
        self.insight_boost: float = 0.0
        
        # Pattern density components (Appendix A.8)
        self.structural_connectivity: float = 0.0
        self.evaluator_reinforcement: float = 0.0
        self.field_amplification: float = 0.0
        
        # Substrate tracking
        self.substrate_type: str = "neural"  # "neural", "symbolic", "hybrid"
        self.compilation_count: int = 0

        # Density stickiness parameter (Appendix A.8.2)
        self.density_weight: float = 0.2  # η in stability function
        self.stickiness: float = 0.0      # Computed bonus for weight update

        self.last_used = time.time()
        self.use_count = 0

    @abstractmethod
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute log probability of observations under this pattern.
        Returns scalar log p(obs | pattern).
        """
        pass

    @abstractmethod
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Generate samples from the pattern (forward simulation)."""
        pass

    @abstractmethod
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Counterfactual: p(effect | do(cause = value)).
        This is the key HPM distinction from standard generative models.
        """
        pass

    @abstractmethod
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """Online learning step (SVI or gradient descent)."""
        pass

    @abstractmethod
    def structural_distance(self, other: 'HPMPattern') -> float:
        """
        Compare causal graphs, not data distributions.
        Used for inhibition (conflict) and recombination.
        """
        pass

    @abstractmethod
    def extract_causal_graph(self) -> nx.DiGraph:
        """Return networkx DiGraph representing causal structure."""
        pass

    def total_score(self, 
                    beta_aff: float = 0.3,
                    gamma_soc: float = 0.1,
                    delta_cur: float = 0.2,
                    eta_coh: float = 0.2,
                    zeta_ins: float = 0.5) -> float:
        """
        Total_i = Accuracy + Affective + Social + Curiosity + Coherence + Insight.
        (Equation from Appendix D, extended)
        """
        return (self.accuracy + 
                beta_aff * self.affective_score + 
                gamma_soc * self.social_score +
                delta_cur * self.curiosity_reward +
                eta_coh * self.coherence_score +
                zeta_ins * self.insight_boost)

    @property
    def pattern_density(self) -> float:
        """D(h) = α*C(h) + β*E(h) + γ*F(h) (Appendix A.8.1)"""
        return (0.4 * self.structural_connectivity + 
                0.3 * self.evaluator_reinforcement + 
                0.3 * self.field_amplification)

    def compute_stickiness(self, base_loss: float) -> float:
        """
        Compute stickiness bonus S(h) as defined in Appendix A.8.2.
        S(h) = σ(η * D(h) - δ * L(h))
        where σ is logistic, η and δ are scaling parameters.
        """
        eta = 2.0   # density scaling
        delta = 0.5 # loss penalty scaling
        x = eta * self.pattern_density - delta * base_loss
        # Logistic function
        self.stickiness = 1.0 / (1.0 + np.exp(-x))
        return self.stickiness

    def mark_used(self):
        self.last_used = time.time()
        self.use_count += 1
```

### 2. Causal Probabilistic Pattern (`hpm_ai_v3/causal_pattern.py`)
Neural implementation using Pyro.

```python
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
from pattern import HPMPattern

class CausalPattern(HPMPattern):
    """
    Pattern implemented as a Pyro probabilistic program with explicit
    latent hierarchy z^(2) -> z^(1) -> x.
    """
    def __init__(self, 
                 input_dim: int,
                 z1_dim: int = 16,   # Surface latent (procedural)
                 z2_dim: int = 4,     # Deep latent (causal/structural)
                 pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.input_dim = input_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        
        # Neural network parameters for generative model
        # p(z2)
        self.z2_loc = torch.zeros(z2_dim)
        self.z2_scale = torch.ones(z2_dim)
        
        # p(z1 | z2)
        self.fc_z2_to_z1 = torch.nn.Linear(z2_dim, z1_dim * 2)  # mean and logvar
        
        # p(x | z1)
        self.fc_z1_to_x = torch.nn.Linear(z1_dim, input_dim * 2)
        
        # Inference networks (encoder)
        self.fc_x_to_z1 = torch.nn.Linear(input_dim, z1_dim * 2)
        self.fc_z1_to_z2 = torch.nn.Linear(z1_dim, z2_dim * 2)
        
        # Optimizer for online learning
        self.optimizer = Adam({"lr": 0.001})
        self.svi = None  # Will initialize on first use
        
        # Causal graph (explicit structural knowledge)
        self.causal_graph = nx.DiGraph()
        self._build_default_graph()
        
    def _build_default_graph(self):
        """Initialize a default causal structure (can be modified by recombination)."""
        self.causal_graph.add_edge("z2", "z1")
        self.causal_graph.add_edge("z1", "x")
        
    def model(self, observations: Optional[Dict[str, torch.Tensor]] = None):
        """Pyro generative model: p(z2, z1, x)."""
        # Level 2: Abstract causal latent
        z2 = pyro.sample("z2", dist.Normal(self.z2_loc, self.z2_scale).to_event(1))
        
        # Level 1: Intermediate latent conditioned on z2
        z1_params = self.fc_z2_to_z1(z2)
        z1_loc, z1_logvar = z1_params.chunk(2, dim=-1)
        z1 = pyro.sample("z1", dist.Normal(z1_loc, torch.exp(0.5 * z1_logvar)).to_event(1))
        
        # Observations conditioned on z1
        x_params = self.fc_z1_to_x(z1)
        x_loc, x_logvar = x_params.chunk(2, dim=-1)
        
        with pyro.plate("data", x_loc.shape[0] if observations else 1):
            x = pyro.sample("x", dist.Normal(x_loc, torch.exp(0.5 * x_logvar)).to_event(1),
                           obs=observations.get("x") if observations else None)
        return x
    
    def guide(self, observations: Dict[str, torch.Tensor]):
        """Pyro inference guide: q(z1, z2 | x)."""
        x = observations["x"]
        
        # Encode x -> z1
        z1_params = self.fc_x_to_z1(x)
        z1_loc, z1_logvar = z1_params.chunk(2, dim=-1)
        z1 = pyro.sample("z1", dist.Normal(z1_loc, torch.exp(0.5 * z1_logvar)).to_event(1))
        
        # Encode z1 -> z2
        z2_params = self.fc_z1_to_z2(z1)
        z2_loc, z2_logvar = z2_params.chunk(2, dim=-1)
        z2 = pyro.sample("z2", dist.Normal(z2_loc, torch.exp(0.5 * z2_logvar)).to_event(1))
        
        return z1, z2
    
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute log p(x | pattern)."""
        conditioned_model = poutine.condition(self.model, data=observations)
        trace = poutine.trace(conditioned_model).get_trace()
        return trace.log_prob_sum()
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Generate samples from prior or conditional on context."""
        if "z2" in context:
            # Condition on abstract latent
            conditioned = poutine.condition(self.model, data={"z2": context["z2"]})
        else:
            conditioned = self.model
            
        with pyro.plate("samples", num_samples):
            trace = poutine.trace(conditioned).get_trace()
        return {"x": trace.nodes["x"]["value"]}
    
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform do-calculus intervention.
        Example: intervention = {"z2": new_value} or {"x": new_value} (for abduction).
        """
        # Use Pyro's do operation
        intervened_model = poutine.do(self.model, data=intervention)
        conditioned_model = poutine.condition(intervened_model, data=context)
        trace = poutine.trace(conditioned_model).get_trace()
        
        result = {}
        for node in ["z2", "z1", "x"]:
            if node in trace.nodes:
                result[node] = trace.nodes[node]["value"]
        return result
    
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """SVI step."""
        if self.svi is None:
            self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        
        loss = self.svi.step(observations)
        
        # Update EMA loss
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.accuracy = -self.loss_ema
        
    def structural_distance(self, other: 'HPMPattern') -> float:
        """Graph edit distance between causal structures."""
        if not isinstance(other, CausalPattern):
            return 1.0
        g1 = self.causal_graph
        g2 = other.causal_graph
        
        # Compute normalized graph edit distance
        # Simplified: Jaccard distance on edge sets
        edges1 = set(g1.edges())
        edges2 = set(g2.edges())
        if not edges1 and not edges2:
            return 0.0
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        return 1.0 - (intersection / union)
    
    def extract_causal_graph(self) -> nx.DiGraph:
        return self.causal_graph.copy()
    
    def compression_score(self, observations: torch.Tensor) -> float:
        """
        Mutual information I(z1; z2 | x) as compression measure (Appendix D.7).
        """
        # Approximate using sampled latents
        guide_trace = poutine.trace(self.guide).get_trace({"x": observations})
        z1_sample = guide_trace.nodes["z1"]["value"]
        z2_sample = guide_trace.nodes["z2"]["value"]
        
        # Simple correlation-based MI proxy
        # Real implementation would use density estimation
        z1_centered = z1_sample - z1_sample.mean(dim=0)
        z2_centered = z2_sample - z2_sample.mean(dim=0)
        cov = torch.mm(z1_centered.T, z2_centered) / (z1_sample.shape[0] - 1)
        
        # Use determinant ratio as rough MI estimate
        var_z1 = torch.var(z1_sample, dim=0).mean()
        var_z2 = torch.var(z2_sample, dim=0).mean()
        mi_approx = 0.5 * torch.log(var_z1 * var_z2 / (torch.det(cov) + 1e-6))
        return float(mi_approx.abs())
```

### 3. Symbolic Pattern (`hpm_ai_v3/symbolic_pattern.py`)
Compiled discrete representations.

```python
from pattern import HPMPattern
import torch
import networkx as nx
import inspect
from typing import Dict, Any, Callable, Optional
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
                 pattern_id: Optional[str] = None):
        super().__init__(pattern_id)
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn
        self.causal_graph = causal_graph or nx.DiGraph()
        self.substrate_type = "symbolic"
        
        # For logging likelihood (assume Gaussian noise model)
        self.noise_std = 0.1
        
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute log prob assuming Gaussian observation noise."""
        x_pred = self.forward_fn(observations)
        if not isinstance(x_pred, torch.Tensor):
            x_pred = torch.tensor(x_pred, dtype=torch.float32)
        x_true = observations["x"]
        mse = torch.mean((x_true - x_pred) ** 2)
        # Gaussian log likelihood (ignoring constant)
        n = x_true.numel()
        return -0.5 * n * torch.log(torch.tensor(2 * np.pi * self.noise_std**2)) - mse / (2 * self.noise_std**2)
    
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Generate samples by running forward function."""
        samples = []
        for _ in range(num_samples):
            out = self.forward_fn(context)
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, dtype=torch.float32)
            samples.append(out)
        return {"x": torch.stack(samples)}
    
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
```

### 4. Motor Pattern (`hpm_ai_v3/motor_pattern.py`)

```python
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
```

### 5. Tool Pattern (`hpm_ai_v3/tool_pattern.py`)

```python
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
```

### 6. Substrate Compiler (`hpm_ai_v3/compiler.py`)

```python
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
```

### 7. Evaluators (`hpm_ai_v3/evaluators.py`)

```python
import torch
import numpy as np
from typing import List, Dict, Any
from pattern import HPMPattern
from causal_pattern import CausalPattern
from symbolic_pattern import SymbolicPattern

class EvaluatorManager:
    def __init__(self, lambda_L: float = 0.1):
        self.lambda_L = lambda_L
        
    def update_structural_connectivity(self, pattern: HPMPattern):
        """
        Estimate internal connectivity of pattern.
        """
        if isinstance(pattern, CausalPattern):
            # Use average absolute weight as proxy for connectivity
            total_param_norm = 0.0
            count = 0
            for param in pattern.parameters():
                total_param_norm += torch.norm(param.data).item()
                count += param.data.numel()
            pattern.structural_connectivity = min(1.0, total_param_norm / max(1, count * 10))
        elif isinstance(pattern, SymbolicPattern):
            # Symbolic complexity
            try:
                import inspect
                src = inspect.getsource(pattern.forward_fn)
                ops = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log']
                count = sum(src.count(op) for op in ops)
                pattern.structural_connectivity = min(1.0, count / 50.0)
            except:
                pattern.structural_connectivity = 0.5
                
    def update_evaluator_reinforcement(self, pattern: HPMPattern):
        """
        Evaluator saturation: sum of all non-epistemic evaluator scores.
        """
        pattern.evaluator_reinforcement = (
            pattern.affective_score + 
            pattern.curiosity_reward + 
            pattern.coherence_score + 
            pattern.social_score
        ) / 4.0  # normalize
        
    def update_epistemic(self, pattern: HPMPattern, observations: Dict[str, torch.Tensor]):
        """Running loss and accuracy."""
        with torch.no_grad():
            logp = pattern.log_prob(observations)
            instant_loss = -logp.item()
        
        if pattern.loss_ema is None:
            pattern.loss_ema = instant_loss
        else:
            pattern.loss_ema = (1 - self.lambda_L) * pattern.loss_ema + self.lambda_L * instant_loss
        pattern.accuracy = -pattern.loss_ema
        
    def update_curiosity(self, pattern: HPMPattern, learning_progress: float):
        """Curiosity as learning progress (improvement in compression)."""
        pattern.curiosity_reward = 0.9 * pattern.curiosity_reward + 0.1 * learning_progress
        
    def update_coherence(self, pattern: HPMPattern, population: List[HPMPattern]):
        """Coherence: how well pattern's causal graph aligns with majority."""
        if not population:
            return
        distances = [pattern.structural_distance(other) for other in population if other != pattern]
        if distances:
            avg_dist = np.mean(distances)
            pattern.coherence_score = 1.0 - avg_dist  # High coherence = low distance
        else:
            pattern.coherence_score = 1.0
            
    def update_affective(self, pattern: HPMPattern, reward: float):
        """Direct affective signal."""
        pattern.affective_score = 0.9 * pattern.affective_score + 0.1 * reward
        
    def update_social(self, pattern: HPMPattern, social_signal: float):
        """Social feedback from pattern field."""
        pattern.social_score = 0.9 * pattern.social_score + 0.1 * social_signal
        
    def compute_insight(self, new_pattern: HPMPattern, parent_a: HPMPattern, parent_b: HPMPattern) -> float:
        """Insight boost for recombined pattern."""
        nov_a = new_pattern.structural_distance(parent_a)
        nov_b = new_pattern.structural_distance(parent_b)
        novelty = (nov_a + nov_b) / 2.0
        effectiveness = new_pattern.accuracy if new_pattern.accuracy > 0 else 0.0
        insight = novelty * 0.6 + effectiveness * 0.4
        return insight
```

### 8. Meta-Pattern Dynamics (`hpm_ai_v3/population.py`)

```python
import numpy as np
import time
import torch
import networkx as nx
from typing import List, Tuple, Optional, Dict
from pattern import HPMPattern
from causal_pattern import CausalPattern
from symbolic_pattern import SymbolicPattern
from motor_pattern import MotorPattern
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler

class PatternPopulation:
    def __init__(self, 
                 initial_patterns: List[HPMPattern],
                 eta: float = 0.1,
                 beta_c: float = 0.05,
                 recombination_prob: float = 0.01,
                 lambda_s: float = 0.3,
                 decay_rate: float = 0.005,
                 interference_strength: float = 0.02,
                 age_decay_rate: float = 0.01):
        self.patterns = initial_patterns
        self.eta = eta
        self.beta_c = beta_c
        self.recombination_prob = recombination_prob
        self.lambda_s = lambda_s
        self.decay_rate = decay_rate
        self.interference_strength = interference_strength
        self.age_decay_rate = age_decay_rate
        
        self._update_kappa_matrix()
        
    def _update_kappa_matrix(self):
        n = len(self.patterns)
        self.kappa = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.kappa[i, j] = self.patterns[i].structural_distance(self.patterns[j])
                    
    def step(self, 
             evaluator_mgr: EvaluatorManager,
             observations: Dict,
             compiler: SubstrateCompiler,
             pattern_field_signal: Optional[List[float]] = None):
        n = len(self.patterns)
        
        # 1. Update evaluators
        for i, p in enumerate(self.patterns):
            evaluator_mgr.update_epistemic(p, observations)
            evaluator_mgr.update_coherence(p, self.patterns)
            
            if hasattr(p, 'compression_score'):
                comp = p.compression_score(observations.get("x", torch.zeros(1)))
                evaluator_mgr.update_curiosity(p, comp)
            
            if pattern_field_signal is not None:
                evaluator_mgr.update_social(p, pattern_field_signal[i])
            
            p.update_parameters(observations)
            
        # Update density and stickiness
        for i, p in enumerate(self.patterns):
            evaluator_mgr.update_structural_connectivity(p)
            evaluator_mgr.update_evaluator_reinforcement(p)
            p.field_amplification = pattern_field_signal[i] if pattern_field_signal else 0.0
            base_loss = p.loss_ema if p.loss_ema is not None else 1.0
            p.compute_stickiness(base_loss)
            
        # 2. Check for substrate shifting
        for i, p in enumerate(self.patterns):
            if compiler.should_compile(p):
                sym_p = None
                if isinstance(p, CausalPattern):
                    sym_p = compiler.compile_to_symbolic(p)
                elif isinstance(p, MotorPattern):
                    sym_p = compiler.compile_motor_to_symbolic(p)
                if sym_p is not None:
                    sym_p.weight = p.weight
                    sym_p.accuracy = p.accuracy
                    self.patterns[i] = sym_p
                    
        # 3. Structural Recombination
        if np.random.random() < self.recombination_prob and len(self.patterns) >= 2:
            weights = np.array([p.weight for p in self.patterns])
            if weights.sum() > 0:
                parents = np.random.choice(self.patterns, size=2, replace=False, p=weights/weights.sum())
                new_pattern = self._recombine(parents[0], parents[1])
                if new_pattern is not None:
                    insight = evaluator_mgr.compute_insight(new_pattern, parents[0], parents[1])
                    new_pattern.insight_boost = insight
                    new_pattern.weight = 0.01
                    self.patterns.append(new_pattern)
                    self._update_kappa_matrix()
                
        # 4. Compute scores
        n = len(self.patterns) # re-compute if patterns added
        totals = np.array([p.total_score() for p in self.patterns])
        weights = np.array([p.weight for p in self.patterns])
        
        # Mark used patterns (top 3 by weight)
        if n > 0:
            top_k_idx = np.argsort(weights)[-min(3, n):]
            for idx in top_k_idx:
                if weights[idx] > 0.01:
                    self.patterns[idx].mark_used()
        
        avg_total = np.dot(weights, totals) / max(weights.sum(), 1e-6)
        stickiness_bonus = np.array([p.stickiness for p in self.patterns])
        avg_stickiness = np.dot(weights, stickiness_bonus) / max(weights.sum(), 1e-6)
        
        # 5. Replicator dynamics
        new_weights = np.zeros(n)
        current_time = time.time()
        
        for i in range(n):
            advantage = (totals[i] - avg_total) + self.lambda_s * (stickiness_bonus[i] - avg_stickiness)
            growth = self.eta * advantage * weights[i]
            
            inhibition = 0
            for j in range(n):
                if i != j:
                    inhibition += self.kappa[i, j] * weights[i] * weights[j]
            inhibition *= self.beta_c
            
            decay = self.decay_rate * weights[i] * (1.0 - self.patterns[i].affective_score)
            
            interference = 0.0
            for j in range(n):
                if i != j:
                    similarity = 1.0 - self.kappa[i, j]
                    interference += similarity * weights[j]
            interference *= self.interference_strength * weights[i]
            
            idle_time = current_time - self.patterns[i].last_used
            age_decay = self.age_decay_rate * idle_time / 3600.0
            
            new_weights[i] = weights[i] + growth - inhibition - decay - interference - age_decay
            new_weights[i] = max(0.0, new_weights[i])
            
        total_w = new_weights.sum()
        if total_w > 0:
            for i, p in enumerate(self.patterns):
                p.weight = new_weights[i] / total_w
        else:
            for p in self.patterns:
                p.weight = 1.0 / n
                
        self.patterns = [p for p in self.patterns if p.weight > 1e-4]
        self._update_kappa_matrix()
        
    def _recombine(self, p1: HPMPattern, p2: HPMPattern) -> Optional[HPMPattern]:
        if isinstance(p1, CausalPattern) and isinstance(p2, CausalPattern):
            new_p = CausalPattern(p1.input_dim, p1.z1_dim, p1.z2_dim)
            with torch.no_grad():
                for param1, param2, new_param in zip(p1.parameters(), p2.parameters(), new_p.parameters()):
                    new_param.data = 0.5 * (param1.data + param2.data)
            new_p.causal_graph = nx.compose(p1.causal_graph, p2.causal_graph)
            return new_p
        elif isinstance(p1, SymbolicPattern) and isinstance(p2, SymbolicPattern):
            def combined_fn(ctx):
                out1 = p1.forward_fn(ctx)
                out2 = p2.forward_fn(ctx)
                if isinstance(out1, torch.Tensor):
                    return (out1 + out2) / 2.0
                return (torch.tensor(out1) + torch.tensor(out2)) / 2.0
            return SymbolicPattern(combined_fn)
        return None
```

### 9. Pattern Field (`hpm_ai_v3/pattern_field.py`)

```python
import ray
import numpy as np
import torch
from typing import List, Dict, Any
from population import PatternPopulation
from pattern import HPMPattern
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler

@ray.remote
class HPMAgent:
    """Ray actor representing a single HPM agent."""
    def __init__(self, agent_id: str, initial_patterns: List[HPMPattern]):
        self.agent_id = agent_id
        self.population = PatternPopulation(initial_patterns)
        self.evaluator_mgr = EvaluatorManager()
        self.compiler = SubstrateCompiler()
        self.private_data_buffer = []
        
    def step(self, observations: Dict, social_signals: List[float] = None):
        self.private_data_buffer.append(observations)
        if len(self.private_data_buffer) > 100:
            self.private_data_buffer.pop(0)
        self.population.step(self.evaluator_mgr, observations, self.compiler, social_signals)
        
    def get_top_patterns(self, k: int = 3) -> List[HPMPattern]:
        sorted_pats = sorted(self.population.patterns, key=lambda p: p.weight, reverse=True)
        return sorted_pats[:k]
    
    def test_pattern(self, pattern: HPMPattern) -> float:
        if not self.private_data_buffer:
            return 0.0
        losses = []
        for obs in self.private_data_buffer:
            with torch.no_grad():
                logp = pattern.log_prob(obs)
                losses.append(-logp.item())
        return -np.mean(losses)
    
    def receive_social_signal(self, pattern_id: str, signal: float):
        for p in self.population.patterns:
            if p.id == pattern_id:
                self.evaluator_mgr.update_social(p, signal)
                break

class PatternField:
    """Institutional layer managing replication."""
    def __init__(self, num_agents: int = 5, pattern_factory: callable = None, replication_threshold: float = 0.1):
        ray.init(ignore_reinit_error=True)
        self.agents = []
        for i in range(num_agents):
            patterns = [pattern_factory() for _ in range(3)]
            agent = HPMAgent.remote(f"agent_{i}", patterns)
            self.agents.append(agent)
            
        self.replication_threshold = replication_threshold
        self.shared_ledger = {}
        
    def step_field(self, observations_batch: List[Dict]):
        futures = [agent.step.remote(observations_batch[i % len(observations_batch)], None) 
                   for i, agent in enumerate(self.agents)]
        ray.get(futures)
        
        published = []
        agent_pubs = ray.get([agent.get_top_patterns.remote(k=1) for agent in self.agents])
        for agent_idx, patterns in enumerate(agent_pubs):
            for pat in patterns:
                published.append((agent_idx, pat))
                
        social_feedback = {i: {} for i in range(len(self.agents))}
        
        for pub_agent_idx, pattern in published:
            replication_scores = []
            for test_agent_idx, test_agent in enumerate(self.agents):
                if test_agent_idx == pub_agent_idx: continue
                score = ray.get(test_agent.test_pattern.remote(pattern))
                replication_scores.append(score)
                
            avg_rep_score = np.mean(replication_scores) if replication_scores else 0.0
            self.shared_ledger[pattern.id] = {
                "publisher": pub_agent_idx,
                "replication_score": avg_rep_score,
                "timestamp": len(self.shared_ledger)
            }
            
            signal = 1.0 if avg_rep_score > self.replication_threshold else -0.5
            
            if pub_agent_idx in social_feedback:
                social_feedback[pub_agent_idx][pattern.id] = signal
            if signal > 0:
                for test_agent_idx in range(len(self.agents)):
                    if test_agent_idx != pub_agent_idx:
                        social_feedback[test_agent_idx][pattern.id] = signal * 0.5
                        
        signal_futures = []
        for agent_idx, feedback_dict in social_feedback.items():
            if feedback_dict:
                for pat_id, sig in feedback_dict.items():
                    signal_futures.append(self.agents[agent_idx].receive_social_signal.remote(pat_id, sig))
        if signal_futures: ray.get(signal_futures)
        
    def get_field_convergence(self) -> float:
        top_patterns = ray.get([agent.get_top_patterns.remote(k=1) for agent in self.agents])
        all_pats = [p for sublist in top_patterns for p in sublist]
        if len(all_pats) < 2: return 1.0
        distances = []
        for i in range(len(all_pats)):
            for j in range(i+1, len(all_pats)):
                distances.append(all_pats[i].structural_distance(all_pats[j]))
        return 1.0 - np.mean(distances)
```

### 10. Minimal Training Loop (`hpm_ai_v3/training_loop.py`)

```python
import torch
import numpy as np
import ray
from causal_pattern import CausalPattern
from pattern_field import PatternField

def create_random_pattern():
    return CausalPattern(input_dim=2, z1_dim=8, z2_dim=2)

def main():
    field = PatternField(num_agents=5, pattern_factory=create_random_pattern)

    def generate_batch(batch_size=10):
        if np.random.random() > 0.5:
            x = torch.randn(batch_size, 2) * 0.5 + torch.tensor([2.0, 2.0])
        else:
            x = torch.randn(batch_size, 2) * 0.5 + torch.tensor([-2.0, -2.0])
        return {"x": x}

    for step in range(100):
        obs_batch = [generate_batch(1) for _ in range(5)]
        field.step_field(obs_batch)
        
        if step % 20 == 0:
            conv = field.get_field_convergence()
            print(f"Step {step}: Field convergence = {conv:.3f}")

    ray.shutdown()

if __name__ == "__main__":
    main()
```
