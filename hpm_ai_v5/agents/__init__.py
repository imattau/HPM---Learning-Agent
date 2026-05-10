"""Agent layer for v5."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Agent": ("hpm_ai_v5.agents.base", "Agent"),
    "AgentInput": ("hpm_ai_v5.agents.base", "AgentInput"),
    "AgentOutput": ("hpm_ai_v5.agents.base", "AgentOutput"),
    "BaseAgent": ("hpm_ai_v5.agents.base", "BaseAgent"),
    "AACResult": ("hpm_ai_v5.agents.adapter_composition", "AACResult"),
    "AACTaskResult": ("hpm_ai_v5.agents.adapter_composition", "AACTaskResult"),
    "AutomaticAdapterComposer": ("hpm_ai_v5.agents.adapter_composition", "AutomaticAdapterComposer"),
    "SelfAdaptiveAgent": ("hpm_ai_v5.agents.adaptive", "SelfAdaptiveAgent"),
    "ATISIntentAgent": ("hpm_ai_v5.agents.atis", "ATISIntentAgent"),
    "ATISRouterAgent": ("hpm_ai_v5.agents.atis", "ATISRouterAgent"),
    "HierarchicalPlanningAgent": ("hpm_ai_v5.agents.hierarchical_planner", "HierarchicalPlanningAgent"),
    "LayeredAgent": ("hpm_ai_v5.agents.layered", "LayeredAgent"),
    "OpenAdapterDiscoveryAgent": ("hpm_ai_v5.agents.open_adapter_discovery", "OpenAdapterDiscoveryAgent"),
    "OpenAdapterDiscoveryResult": ("hpm_ai_v5.agents.open_adapter_discovery", "OpenAdapterDiscoveryResult"),
    "OpenAdapterPipelineSpec": ("hpm_ai_v5.agents.open_adapter_discovery", "OpenAdapterPipelineSpec"),
    "OpenAdapterTask": ("hpm_ai_v5.agents.open_adapter_discovery", "OpenAdapterTask"),
    "OpenAdapterTaskResult": ("hpm_ai_v5.agents.open_adapter_discovery", "OpenAdapterTaskResult"),
    "OpenAdapterExample": ("hpm_ai_v5.agents.open_adapter_discovery", "OpenAdapterExample"),
    "MetaPattern": ("hpm_ai_v5.agents.meta_pattern", "MetaPattern"),
    "MetaPatternDecision": ("hpm_ai_v5.agents.meta_pattern", "MetaPatternDecision"),
    "MetaPatternDiscoveryAgent": ("hpm_ai_v5.agents.meta_pattern", "MetaPatternDiscoveryAgent"),
    "AgentPacket": ("hpm_ai_v5.agents.packet", "AgentPacket"),
    "AgentPipeline": ("hpm_ai_v5.agents.pipeline", "AgentPipeline"),
    "ScoringWeightAdaptationAgent": ("hpm_ai_v5.agents.scoring", "ScoringWeightAdaptationAgent"),
    "WeightDecision": ("hpm_ai_v5.agents.scoring", "WeightDecision"),
    "UtilityCandidate": ("hpm_ai_v5.agents.utility", "UtilityCandidate"),
    "UtilityDecision": ("hpm_ai_v5.agents.utility", "UtilityDecision"),
    "UtilityLearningAgent": ("hpm_ai_v5.agents.utility", "UtilityLearningAgent"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


__all__ = list(_EXPORTS)
