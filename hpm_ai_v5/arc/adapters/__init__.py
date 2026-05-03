"""ARC adapter layer."""

from .analysis import (
    ColorDeltaAdapter,
    ColorMapperAdapter,
    EdgeListAdapter,
    GridDeltaAdapter,
    ObjectDetectorAdapter,
    ObjectDeltaAdapter,
    PatternMinerAdapter,
    ShapeDefinerAdapter,
    SpatialRelationAdapter,
    StructuralDeltaAdapter,
)
from .core import ColourMapAdapter, DeltaAdapter, ExamplePairAdapter, GeometryAdapter, GridAdapter, ObjectExtractionAdapter, TaskAdapter
from .hypothesis import ArcHypothesisAdapter, LineExtensionAdapter, SymmetryCompletionAdapter

ObjectExtractionAdapter = ObjectDetectorAdapter
ColorMapperAdapterLegacy = ColourMapAdapter

__all__ = [
    "ColorDeltaAdapter",
    "ColorMapperAdapter",
    "ColorMapperAdapterLegacy",
    "EdgeListAdapter",
    "ColourMapAdapter",
    "DeltaAdapter",
    "ExamplePairAdapter",
    "GeometryAdapter",
    "GridAdapter",
    "ArcHypothesisAdapter",
    "GridDeltaAdapter",
    "LineExtensionAdapter",
    "ObjectExtractionAdapter",
    "ObjectDetectorAdapter",
    "ObjectDeltaAdapter",
    "PatternMinerAdapter",
    "ShapeDefinerAdapter",
    "SpatialRelationAdapter",
    "SymmetryCompletionAdapter",
    "StructuralDeltaAdapter",
    "TaskAdapter",
]
