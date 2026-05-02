"""Composable adapter pipelines for v5."""

from .base import Adapter
from .packet import AdapterPacket
from .registry import AdapterRegistry

__all__ = ["Adapter", "AdapterPacket", "AdapterRegistry"]
