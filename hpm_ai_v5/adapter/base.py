"""Adapter contract for composable pipelines."""

from __future__ import annotations

from typing import Protocol

from .packet import AdapterPacket


class Adapter(Protocol):
    """Composable pipeline adapter."""

    name: str
    requires: list[str]
    provides: list[str]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raise NotImplementedError
