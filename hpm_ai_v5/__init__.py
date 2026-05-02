"""HPM AI v5.

Minimal, KISS-oriented pattern-learning core.
"""

from .agents import Agent, AgentInput, AgentOutput, AgentPacket, AgentPipeline, BaseAgent
from .arc import ArcExample, ArcObject, ArcPipeline, ArcSolver, ArcTask, ArcTransformation
from .core import Action, CoreConfig, Delta, Pattern, PatternEngine, PatternSequence, PatternStore, ReasoningTrace, State
from .evaluation import BenchmarkScore, EvaluationReport, V5ObjectiveEvaluator
from .pipeline import HPMPipeline

__all__ = [
    "Action",
    "CoreConfig",
    "Agent",
    "AgentInput",
    "AgentOutput",
    "AgentPacket",
    "AgentPipeline",
    "BaseAgent",
    "ArcExample",
    "ArcObject",
    "ArcPipeline",
    "ArcSolver",
    "ArcTask",
    "ArcTransformation",
    "Delta",
    "HPMPipeline",
    "Pattern",
    "PatternEngine",
    "PatternSequence",
    "PatternStore",
    "ReasoningTrace",
    "BenchmarkScore",
    "EvaluationReport",
    "V5ObjectiveEvaluator",
    "State",
]
