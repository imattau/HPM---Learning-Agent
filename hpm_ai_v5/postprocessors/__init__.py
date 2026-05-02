"""Postprocessing adapters for v5."""

from .base import Postprocessor
from .numeric import NumericPostprocessor

__all__ = ["NumericPostprocessor", "Postprocessor"]
