"""Composable agent pipeline for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .packet import AgentPacket


class AgentStep:
    name: str

    def step(self, packet: AgentPacket) -> AgentPacket:
        raise NotImplementedError


@dataclass
class AgentPipeline:
    """Run a fixed sequence of agents over a shared packet."""

    agents: list[AgentStep] = field(default_factory=list)

    def run(self, packet: AgentPacket) -> AgentPacket:
        for agent in self.agents:
            packet = agent.step(packet)
            if not packet.agent_trace or packet.agent_trace[-1] != agent.name:
                packet.agent_trace.append(agent.name)
            packet.log(agent.name, {"stage": "agent_step"})
        return packet
