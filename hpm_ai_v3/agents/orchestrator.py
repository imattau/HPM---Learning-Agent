"""
unified_orchestrator.py - Meta-orchestrator that can select among tools and agents.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from ..meta_tool_orchestrator import MetaToolOrchestrator
from ..tools.registry import ToolRegistry
from .registry import AgentRegistry
from ..pattern import HPMPattern


class UnifiedOrchestrator(MetaToolOrchestrator):
    """
    Meta-pattern that learns to select among available tools AND agents.
    """
    def __init__(self, 
                 available_tools: List[str],
                 available_agents: List[str],
                 context_feature_dim: int = 16,
                 hidden_dim: int = 32,
                 pattern_id: Optional[str] = None):
        # Combine both into a single action space for the base orchestrator
        self.tools_list = available_tools
        self.agents_list = available_agents
        all_actions = available_tools + available_agents
        super().__init__(all_actions, context_feature_dim, hidden_dim, pattern_id)
        
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        out = super().sample(context, num_samples)
        selected = out["selected_tool"]
        # Determine if selected is agent or tool
        is_agent = selected in self.agents_list
        out["is_agent"] = is_agent
        return out
    
    def create_pattern(self, name: str) -> Optional[HPMPattern]:
        """Create a pattern for the selected tool or agent."""
        if name in self.tools_list:
            return ToolRegistry.create_pattern(name)
        elif name in self.agents_list:
            return AgentRegistry.create_pattern(name)
        return None
