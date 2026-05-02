"""Dependency-aware adapter registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Adapter
from .packet import AdapterPacket


@dataclass
class AdapterRegistry:
    """Register adapters, resolve dependencies, and trace execution."""

    adapters: dict[str, Adapter] = field(default_factory=dict)

    def register(self, adapter: Adapter) -> None:
        self.adapters[adapter.name] = adapter

    def resolve(self, target_outputs: list[str]) -> list[Adapter]:
        ordered: list[Adapter] = []
        added: set[str] = set()

        def add_adapter(name: str) -> None:
            if name in added:
                return
            adapter = self.adapters.get(name)
            if adapter is None:
                return
            for dependency in adapter.requires:
                add_adapter(dependency)
            ordered.append(adapter)
            added.add(name)

        for output in target_outputs:
            add_adapter(output)
        return ordered

    def run(self, packet: AdapterPacket, target_outputs: list[str]) -> AdapterPacket:
        for adapter in self.resolve(target_outputs):
            packet = adapter.run(packet)
            packet.log(adapter.name, {"provides": list(adapter.provides), "requires": list(adapter.requires)}, role="adapter")
        return packet
