"""ARC end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapter import AdapterRegistry
from ..agents import AgentPipeline
from ..schemas.packet import Packet
from .adapters import (
    ArcHypothesisAdapter,
    ColorDeltaAdapter,
    ColorMapperAdapter,
    EdgeListAdapter,
    ExamplePairAdapter,
    GridAdapter,
    GridDeltaAdapter,
    LineExtensionAdapter,
    ObjectDetectorAdapter,
    ObjectDeltaAdapter,
    PatternMinerAdapter,
    ShapeDefinerAdapter,
    SpatialRelationAdapter,
    SymmetryCompletionAdapter,
    StructuralDeltaAdapter,
    TaskAdapter,
)
from .agents import ArcCriticAgent, ArcHypothesisAgent, ArcOutputAgent, ArcRouterAgent, ArcSimulationAgent
from .polygraphs import ColourPolygraph, GeometryPolygraph, ImagePolygraph, ObjectPolygraph, TransformationPolygraph


def _preprocessing_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    for adapter in (
        TaskAdapter(),
        GridAdapter(),
        EdgeListAdapter(),
        ObjectDetectorAdapter(),
        ShapeDefinerAdapter(),
        SpatialRelationAdapter(),
        ArcHypothesisAdapter(),
        SymmetryCompletionAdapter(),
        LineExtensionAdapter(),
        ColorMapperAdapter(),
        PatternMinerAdapter(),
        GridDeltaAdapter(),
        ObjectDeltaAdapter(),
        ColorDeltaAdapter(),
        StructuralDeltaAdapter(),
        ExamplePairAdapter(),
    ):
        registry.register(adapter)
    return registry


def _polygraph_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    for adapter in (
        ImagePolygraph(),
        ObjectPolygraph(),
        TransformationPolygraph(),
        ColourPolygraph(),
        GeometryPolygraph(),
    ):
        registry.register(adapter)
    return registry


def _agent_pipeline() -> AgentPipeline:
    return AgentPipeline(
        agents=[
            ArcRouterAgent(),
            ArcHypothesisAgent(),
            ArcSimulationAgent(),
            ArcCriticAgent(),
            ArcOutputAgent(),
        ]
    )


@dataclass
class ArcPipeline:
    preprocessing: AdapterRegistry = field(default_factory=_preprocessing_registry)
    polygraphs: AdapterRegistry = field(default_factory=_polygraph_registry)
    agents: AgentPipeline = field(default_factory=_agent_pipeline)

    def run(self, raw_task) -> Packet:
        packet = Packet(raw_input=raw_task, context={"arc": {}})
        packet = self.preprocessing.run(packet, target_outputs=["arc_examples"])
        packet = self.polygraphs.run(packet, target_outputs=["geometry_polygraph"])
        packet = self.agents.run(packet)
        return packet
