"""Agent layer for v5."""

from .base import Agent, AgentInput, AgentOutput, BaseAgent
from .adapter_composition import AACResult, AACTaskResult, AutomaticAdapterComposer
from .open_adapter_discovery import (
    OpenAdapterDiscoveryAgent,
    OpenAdapterDiscoveryResult,
    OpenAdapterPipelineSpec,
    OpenAdapterTask,
    OpenAdapterTaskResult,
    OpenAdapterExample,
)
from .meta_pattern import MetaPattern, MetaPatternDecision, MetaPatternDiscoveryAgent
from .packet import AgentPacket
from .pipeline import AgentPipeline
from .scoring import ScoringWeightAdaptationAgent, WeightDecision
from .utility import UtilityCandidate, UtilityDecision, UtilityLearningAgent

__all__ = [
    "Agent",
    "AgentInput",
    "AgentOutput",
    "AACResult",
    "AACTaskResult",
    "AgentPacket",
    "AgentPipeline",
    "BaseAgent",
    "AutomaticAdapterComposer",
    "OpenAdapterDiscoveryAgent",
    "OpenAdapterDiscoveryResult",
    "OpenAdapterPipelineSpec",
    "OpenAdapterTask",
    "OpenAdapterTaskResult",
    "OpenAdapterExample",
    "MetaPattern",
    "MetaPatternDecision",
    "MetaPatternDiscoveryAgent",
    "ScoringWeightAdaptationAgent",
    "WeightDecision",
    "UtilityCandidate",
    "UtilityDecision",
    "UtilityLearningAgent",
]
