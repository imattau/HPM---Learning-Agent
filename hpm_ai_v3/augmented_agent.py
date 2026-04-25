"""
augmented_agent.py - HPM agent with dynamic tool orchestration.
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any

from .population import PatternPopulation
from .evaluators import EvaluatorManager
from .compiler import SubstrateCompiler
from .tools.registry import ToolRegistry
from .meta_tool_orchestrator import MetaToolOrchestrator
from .classification_pattern import ClassificationPattern
from .pattern import HPMPattern


class AugmentedHPMAgent:
    """
    HPM agent that can discover and use external tools dynamically.
    """
    def __init__(self, 
                 tool_names: List[str],
                 base_patterns: Optional[List[HPMPattern]] = None,
                 context_feature_dim: int = 16,
                 history_len: int = 20,
                 agent_id: Optional[str] = None):
        
        self.agent_id = agent_id or f"agent_{id(self) % 1000}"
        self.tool_names = tool_names
        self.tool_patterns = []
        for name in tool_names:
            pat = ToolRegistry.create_pattern(name)
            if pat:
                self.tool_patterns.append(pat)
            else:
                print(f"Warning: Tool '{name}' not found in registry.")
        
        # Base neural patterns (e.g., classifiers)
        if base_patterns is None:
            base_patterns = [ClassificationPattern() for _ in range(3)]
        self.base_patterns = base_patterns
        
        # Combine all patterns in population
        all_patterns = base_patterns + self.tool_patterns
        self.population = PatternPopulation(all_patterns)
        self.eval_mgr = EvaluatorManager()
        self.compiler = SubstrateCompiler(use_gp=False)
        
        # Meta-orchestrator for tool selection
        self.meta = MetaToolOrchestrator(
            available_tools=tool_names,
            context_feature_dim=context_feature_dim
        )
        
        # History for context features
        self.history_len = history_len
        self.loss_history = deque(maxlen=history_len)
        self.tool_usage_history = deque(maxlen=history_len)
        
    def _compute_context_features(self, raw_input: Dict[str, Any]) -> torch.Tensor:
        """
        Convert raw input and history into a fixed-size feature vector for the meta-pattern.
        """
        features = []
        
        # Input statistics (e.g., mean, variance of pixel values)
        if "image" in raw_input:
            img = raw_input["image"]
            if isinstance(img, torch.Tensor):
                features.extend([img.mean().item(), img.std().item(), img.max().item(), img.min().item()])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Recent loss trend
        if len(self.loss_history) > 0:
            recent_loss = list(self.loss_history)
            features.append(np.mean(recent_loss))
            features.append(np.std(recent_loss) if len(recent_loss) > 1 else 0.0)
            if len(recent_loss) >= 5:
                features.append(recent_loss[-1] - recent_loss[0])  # trend
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Tool usage frequency
        if len(self.tool_usage_history) > 0:
            tool_counts = {}
            for tool in self.tool_usage_history:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            for tool in self.tool_names:
                features.append(tool_counts.get(tool, 0) / len(self.tool_usage_history))
        else:
            features.extend([0.0] * len(self.tool_names))
        
        # Pad to fixed dimension
        target_dim = self.meta.context_feature_dim
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def invoke(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent as a functional pattern.
        """
        working_context = context.copy()
        
        # 1. Tool Orchestration (if tools available)
        if self.tool_names:
            context_features = self._compute_context_features(context)
            meta_out = self.meta.sample({"context_features": context_features})
            selected_tool = meta_out["selected_tool"]
            
            if selected_tool in [p.tool_name for p in self.tool_patterns]:
                tool_pat = next(p for p in self.tool_patterns if p.tool_name == selected_tool)
                try:
                    tool_result = tool_pat.sample(working_context)
                    working_context.update(tool_result)
                except:
                    pass
        
        # 2. Prediction from .population
        top_p = self.population.get_top_patterns(k=1)
        if top_p:
            pred = top_p[0].sample(working_context)
            return pred
            
        return working_context

    def step(self, raw_input: Dict[str, Any], target: Optional[torch.Tensor] = None):
        """
        One step of perception, tool orchestration, and learning.
        """
        # Compute context features
        context_features = self._compute_context_features(raw_input)
        
        # Meta-pattern selects which tool(s) to use
        meta_out = self.meta.sample({"context_features": context_features})
        selected_tool = meta_out["selected_tool"]
        tool_idx = meta_out["tool_idx"]
        
        # Execute selected tool
        context = raw_input.copy()
        if selected_tool in [p.tool_name for p in self.tool_patterns]:
            tool_pat = next(p for p in self.tool_patterns if p.tool_name == selected_tool)
            tool_result = tool_pat.sample(context)
            context.update(tool_result)
            self.tool_usage_history.append(selected_tool)
        
        # Use population to make prediction
        if "input" not in context:
            # Need to construct input from context
            if "features" in context:
                context["input"] = context["features"]
            elif "image" in context:
                context["input"] = context["image"].flatten()
        
        # Get prediction from top pattern
        obs = context.copy()
        if target is not None:
            obs["target"] = target
        
        # Ensure 'input' is present for standard evaluators
        if "input" not in obs and "features" in obs:
            obs["input"] = obs["features"]
        
        # Run population step
        self.population.step(self.eval_mgr, obs, self.compiler)
        
        # Compute reward for meta-pattern
        if target is not None:
            top_patterns = self.population.get_top_patterns(k=1)
            reward = 0.0
            if top_patterns:
                top = top_patterns[0]
                pred_out = top.sample(obs)
                if "y" in pred_out:
                    pred = pred_out["y"]
                    if pred.dim() > 0:
                        pred_class = pred.argmax().item() if pred.shape[-1] > 1 else pred.item()
                        reward = 1.0 if pred_class == target.item() else -0.5
                elif "probs" in pred_out:
                    pred_class = pred_out["probs"].argmax().item()
                    reward = 1.0 if pred_class == target.item() else -0.5
            
            # Update loss history
            self.loss_history.append(-reward)
            
            # Update meta-pattern
            meta_obs = {
                "context_features": context_features.unsqueeze(0),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "tool_idx_taken": tool_idx.unsqueeze(0) if tool_idx.dim() == 0 else tool_idx
            }
            self.meta.update_parameters(meta_obs)
        
        return context
    
    def get_top_patterns(self, k: int = 3):
        return self.population.get_top_patterns(k)
