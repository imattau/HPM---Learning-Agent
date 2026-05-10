"""Polygraph generators for v5."""

from .action_policy import ActionPolygraphGenerator
from .base import PolygraphGenerator, PolygraphView
from .numeric import NumericPolygraphGenerator
from .timeseries import TimeSeriesPolygraphGenerator


def __getattr__(name: str):
    if name == "AudioPolygraphGenerator":
        from .audio import AudioPolygraphGenerator
        return AudioPolygraphGenerator
    if name == "GraphPolygraphGenerator":
        from .graph import GraphPolygraphGenerator
        return GraphPolygraphGenerator
    if name == "GridPolygraphGenerator":
        from .grid import GridPolygraphGenerator
        return GridPolygraphGenerator
    if name == "InterconnectedNLPPolygraphGenerator":
        from .nlp import InterconnectedNLPPolygraphGenerator
        return InterconnectedNLPPolygraphGenerator
    if name == "TextPolygraphGenerator":
        from .text import TextPolygraphGenerator
        return TextPolygraphGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionPolygraphGenerator",
    "AudioPolygraphGenerator",
    "GraphPolygraphGenerator",
    "GridPolygraphGenerator",
    "InterconnectedNLPPolygraphGenerator",
    "NumericPolygraphGenerator",
    "PolygraphGenerator",
    "PolygraphView",
    "TextPolygraphGenerator",
    "TimeSeriesPolygraphGenerator",
]
