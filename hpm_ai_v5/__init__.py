"""HPM AI v5.

Minimal, KISS-oriented pattern-learning core.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Action": ("hpm_ai_v5.core", "Action"),
    "CoreConfig": ("hpm_ai_v5.core", "CoreConfig"),
    "Agent": ("hpm_ai_v5.agents", "Agent"),
    "AgentInput": ("hpm_ai_v5.agents", "AgentInput"),
    "AgentOutput": ("hpm_ai_v5.agents", "AgentOutput"),
    "AgentPacket": ("hpm_ai_v5.agents", "AgentPacket"),
    "AgentPipeline": ("hpm_ai_v5.agents", "AgentPipeline"),
    "BaseAgent": ("hpm_ai_v5.agents", "BaseAgent"),
    "ArcExample": ("hpm_ai_v5.arc", "ArcExample"),
    "ArcObject": ("hpm_ai_v5.arc", "ArcObject"),
    "ArcPipeline": ("hpm_ai_v5.arc", "ArcPipeline"),
    "ArcSolver": ("hpm_ai_v5.arc", "ArcSolver"),
    "ArcTask": ("hpm_ai_v5.arc", "ArcTask"),
    "ArcTransformation": ("hpm_ai_v5.arc", "ArcTransformation"),
    "Delta": ("hpm_ai_v5.core", "Delta"),
    "HPMPipeline": ("hpm_ai_v5.pipeline", "HPMPipeline"),
    "Pattern": ("hpm_ai_v5.core", "Pattern"),
    "PatternEngine": ("hpm_ai_v5.core", "PatternEngine"),
    "PatternSequence": ("hpm_ai_v5.core", "PatternSequence"),
    "PatternStore": ("hpm_ai_v5.core", "PatternStore"),
    "ReasoningTrace": ("hpm_ai_v5.core", "ReasoningTrace"),
    "BenchmarkScore": ("hpm_ai_v5.evaluation", "BenchmarkScore"),
    "EvaluationReport": ("hpm_ai_v5.evaluation", "EvaluationReport"),
    "V5ObjectiveEvaluator": ("hpm_ai_v5.evaluation", "V5ObjectiveEvaluator"),
    "State": ("hpm_ai_v5.core", "State"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


__all__ = list(_EXPORTS)
