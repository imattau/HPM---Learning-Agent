"""Pipeline wrappers for v5."""

from .adapter_pipeline import AdapterPipeline, PipelineResult
from .agent_pipeline import AgentPipeline

__all__ = ["AdapterPipeline", "AgentPipeline", "PipelineResult"]
