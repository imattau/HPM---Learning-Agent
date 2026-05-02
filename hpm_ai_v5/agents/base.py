"""Minimal agent interface for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..adapter import AdapterPacket, AdapterRegistry
from ..core import Action, PatternEngine
from ..pipeline import HPMPipeline, PipelineResult
from .packet import AgentPacket


@dataclass
class AgentInput:
    raw: Any
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    content: Any
    action_type: str
    confidence: float
    valid: bool
    trace: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    name: str

    def observe(self, input: AgentInput) -> None:
        raise NotImplementedError

    def decide(self) -> dict[str, Any]:
        raise NotImplementedError

    def act(self) -> AgentOutput:
        raise NotImplementedError

    def step(self, input: AgentInput) -> AgentOutput:
        raise NotImplementedError


@dataclass
class BaseAgent:
    """Goal-directed wrapper around the HPM pipeline."""

    name: str
    core: PatternEngine
    preprocessors: list[Any] = field(default_factory=list)
    postprocessors: list[Any] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=lambda: {
        "turn_history": [],
        "active_goal": None,
        "active_topic": None,
        "last_pattern": None,
        "last_action": None,
        "context_memory": {},
    })
    last_decision: dict[str, Any] | None = None
    pipeline: HPMPipeline | None = None

    def __post_init__(self) -> None:
        if self.pipeline is None:
            if not self.preprocessors or not self.postprocessors:
                raise ValueError("BaseAgent requires preprocessors and postprocessors, or an explicit pipeline")
            self.pipeline = HPMPipeline(
                preprocessor=self.preprocessors[0],
                engine=self.core,
                postprocessor=self.postprocessors[0],
            )
            for adapter in self.preprocessors[1:]:
                self.pipeline.register_preprocessor(adapter)
            for adapter in self.postprocessors[1:]:
                self.pipeline.register_postprocessor(adapter)

    def observe(self, input: AgentInput) -> None:
        self.state["turn_history"].append({"raw": input.raw, "context": dict(input.context)})
        self.state["last_input"] = input

    def decide(self) -> dict[str, Any]:
        last_input = self.state.get("last_input")
        if last_input is None:
            self.last_decision = {"decision": None, "packet": None}
            return self.last_decision

        goal = self.state.get("active_goal") or {}
        decision = self.pipeline.step(last_input.raw, goal=goal, context=last_input.context)
        self.state["last_packet"] = decision.input.packet
        self.state["last_action"] = decision.action
        self.state["last_pattern"] = decision.action.selected_pattern.name if decision.action.selected_pattern else None
        self.last_decision = {"decision": decision, "packet": decision.input.packet}
        return self.last_decision

    def act(self) -> AgentOutput:
        if self.last_decision is None or self.last_decision["decision"] is None:
            return AgentOutput(content=None, action_type="unknown", confidence=0.0, valid=False, trace={"agent": self.name})

        decision: PipelineResult = self.last_decision["decision"]
        output = decision.output
        valid = output is not None
        return AgentOutput(
            content=output,
            action_type=decision.action.action_type,
            confidence=decision.action.confidence,
            valid=valid,
            trace={
                "agent": self.name,
                "adapter_trace": decision.input.packet.trace if decision.input.packet is not None else [],
                "core_decision": {
                    "selected_pattern": None if decision.action.selected_pattern is None else decision.action.selected_pattern.name,
                    "selected_sequence": None if decision.action.selected_sequence is None else list(decision.action.selected_sequence.pattern_names),
                    "selected_view": decision.action.selected_view,
                    "confidence": decision.action.confidence,
                    "trace": decision.action.trace,
                },
            },
        )

    def step(self, input: AgentInput) -> AgentOutput:
        self.observe(input)
        self.decide()
        return self.act()

    def step_packet(self, packet: AgentPacket) -> AgentPacket:
        """Adapter-style agent step for agent pipelines."""

        packet.agent_trace.append(self.name)
        self.observe(AgentInput(raw=packet.raw_input, context=dict(packet.context)))
        decision = self.decide()
        output = self.act()
        packet.state.setdefault("agent_outputs", []).append(output)
        packet.final_output = output.content
        packet.candidate_outputs.append(output)
        packet.core_actions.append(decision["decision"])
        packet.log(self.name, {"action_type": output.action_type, "confidence": output.confidence})
        return packet
