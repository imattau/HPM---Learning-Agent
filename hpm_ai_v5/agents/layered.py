"""Layered HPM stack where each level consumes the lower level's latent state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent
from ..pipeline import HPMPipeline, PipelineResult


@dataclass
class LayeredAgent(BaseAgent):
    """
    Orchestrates multiple HPMPipelines in a stack.
    L1 output feeds L2, L2 output feeds L3, etc.
    """

    layers: list[HPMPipeline] = field(default_factory=list)

    def step(self, raw: Any, *, goal: dict[str, float] | None = None, context: dict[str, Any] | None = None) -> list[PipelineResult]:
        """
        Execute a full pass through all layers.
        Returns a list of PipelineResults, one for each layer.
        """
        results = []
        current_input = raw
        current_context = dict(context or {})
        
        for i, pipeline in enumerate(self.layers):
            # Each layer might have its own goal, but we can pass the main goal to the top layer
            # or use context to differentiate.
            layer_goal = goal if i == len(self.layers) - 1 else None
            
            res = pipeline.step(current_input, goal=layer_goal, context=current_context)
            results.append(res)
            
            # The "latent state" of this layer becomes the "raw input" for the next layer.
            # Usually we use the selected pattern's name or the preprocessed state vector.
            if res.action and res.action.selected_pattern:
                current_input = res.action.selected_pattern.name
            else:
                current_input = res.input.state.value
                
            # Update context for next layer
            current_context[f"l{i+1}_pattern"] = res.action.selected_pattern.name if res.action and res.action.selected_pattern else None
            current_context[f"l{i+1}_confidence"] = res.action.confidence if res.action else 0.0
            
        return results
