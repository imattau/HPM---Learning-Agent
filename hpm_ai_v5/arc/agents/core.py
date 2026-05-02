"""ARC specialist agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schemas.packet import Packet
from ..common import ArcTask, ArcTransformation, infer_transformation, merge_transformations, score_transformation


def _arc(packet: Packet) -> dict[str, Any]:
    return packet.context.setdefault("arc", {})


@dataclass
class ArcRouterAgent:
    name: str = "arc_router"

    def step_packet(self, packet: Packet) -> Packet:
        arc = _arc(packet)
        task: ArcTask = arc["task"]
        route = "unknown"
        kinds = {
            candidate.kind
            for candidate in (infer_transformation(example.input_grid, example.output_grid) for example in task.train if example.output_grid is not None)
            if candidate is not None
        }
        labels = {
            candidate.label
            for candidate in (infer_transformation(example.input_grid, example.output_grid) for example in task.train if example.output_grid is not None)
            if candidate is not None
        }
        if len(kinds) == 1:
            route = next(iter(labels)) if len(labels) == 1 else next(iter(kinds))
        elif len(kinds) > 1:
            route = "mixed_transform"
        arc["route"] = route
        packet.log(self.name, {"route": route}, role="agent")
        return packet

    def step(self, packet: Packet) -> Packet:
        return self.step_packet(packet)


@dataclass
class ArcHypothesisAgent:
    name: str = "arc_hypothesis"

    def step_packet(self, packet: Packet) -> Packet:
        arc = _arc(packet)
        task: ArcTask = arc["task"]
        candidates = [infer_transformation(example.input_grid, example.output_grid) for example in task.train if example.output_grid is not None]
        merged = merge_transformations(candidates)
        arc["candidates"] = [candidate for candidate in candidates if candidate is not None]
        arc["candidate"] = merged
        packet.candidate_outputs.append(None if merged is None else merged.describe())
        packet.log(self.name, {"candidates": len(arc["candidates"]), "selected": None if merged is None else merged.describe()}, role="agent")
        return packet

    def step(self, packet: Packet) -> Packet:
        return self.step_packet(packet)


@dataclass
class ArcSimulationAgent:
    name: str = "arc_simulation"

    def step_packet(self, packet: Packet) -> Packet:
        arc = _arc(packet)
        task: ArcTask = arc["task"]
        candidate: ArcTransformation | None = arc.get("candidate")
        if candidate is None:
            arc["simulation"] = {"train_accuracy": 0.0, "consistency": 0.0, "score": -1.0}
            packet.log(self.name, {"reason": "no_candidate"}, role="agent")
            return packet

        metrics = score_transformation(candidate, task.train)
        arc["simulation"] = metrics
        packet.log(self.name, metrics, role="agent")
        return packet

    def step(self, packet: Packet) -> Packet:
        return self.step_packet(packet)


@dataclass
class ArcCriticAgent:
    name: str = "arc_critic"

    def step_packet(self, packet: Packet) -> Packet:
        arc = _arc(packet)
        candidate: ArcTransformation | None = arc.get("candidate")
        simulation: dict[str, float] = arc.get("simulation", {})
        accepted = bool(candidate is not None and simulation.get("train_accuracy", 0.0) == 1.0)
        arc["accepted"] = accepted
        packet.log(self.name, {"accepted": accepted, **simulation}, role="agent")
        return packet

    def step(self, packet: Packet) -> Packet:
        return self.step_packet(packet)


@dataclass
class ArcOutputAgent:
    name: str = "arc_output"

    def step_packet(self, packet: Packet) -> Packet:
        arc = _arc(packet)
        task: ArcTask = arc["task"]
        candidate: ArcTransformation | None = arc.get("candidate")
        accepted = arc.get("accepted", False)
        if not accepted or candidate is None or not task.test_inputs:
            packet.final_output = None
            packet.validated_output = None
            packet.log(self.name, {"final_output": None}, role="agent")
            return packet

        predicted = candidate.apply(task.test_inputs[0])
        packet.final_output = [list(row) for row in predicted]
        packet.validated_output = packet.final_output
        packet.log(self.name, {"final_output": packet.final_output}, role="agent")
        return packet

    def step(self, packet: Packet) -> Packet:
        return self.step_packet(packet)
