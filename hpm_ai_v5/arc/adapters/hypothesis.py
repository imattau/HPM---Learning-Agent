"""ARC candidate hypothesis adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...adapter import AdapterPacket
from ..common import ArcTask, ArcTransformation, infer_transformation, merge_transformations


def _arc(packet: AdapterPacket) -> dict[str, Any]:
    return packet.context.setdefault("arc", {})


def _candidate_signature(candidate: ArcTransformation) -> tuple[Any, ...]:
    return (
        candidate.kind,
        candidate.axis,
        candidate.line_index,
        candidate.crop_bbox,
        candidate.preserve_canvas,
        candidate.dx,
        candidate.dy,
        candidate.background,
        tuple(sorted(candidate.colour_map.items())),
        candidate.label,
    )


def _store_candidates(
    packet: AdapterPacket,
    *,
    key: str,
    source: str,
    candidates: list[ArcTransformation | None],
) -> list[ArcTransformation]:
    arc = _arc(packet)
    stored: list[ArcTransformation] = arc.setdefault(key, [])
    trace: list[dict[str, Any]] = arc.setdefault(f"{key}_trace", [])
    seen = {_candidate_signature(candidate) for candidate in stored}
    added: list[ArcTransformation] = []
    for candidate in candidates:
        if candidate is None:
            continue
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        stored.append(candidate)
        added.append(candidate)
        trace.append(
            {
                "source": source,
                "kind": candidate.kind,
                "label": candidate.label,
                "description": candidate.describe(),
            }
        )
    arc["candidate"] = merge_transformations(stored)
    merged = arc.get("candidate")
    packet.candidate_outputs.append(None if merged is None else merged.describe())
    return added


@dataclass
class ArcHypothesisAdapter:
    name: str = "arc_hypotheses"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_relations"]
        self.provides = ["arc_hypotheses"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        candidates = [infer_transformation(example.input_grid, example.output_grid) for example in task.train if example.output_grid is not None]
        added = _store_candidates(packet, key="hypothesis_candidates", source=self.name, candidates=candidates)
        _arc(packet)["hypotheses"] = [candidate.describe() for candidate in _arc(packet).get("hypothesis_candidates", [])]
        packet.views.append(
            {
                "name": "arc_hypothesis",
                "candidate_count": len(_arc(packet).get("hypothesis_candidates", [])),
                "added": len(added),
            }
        )
        packet.log(self.name, {"candidates": len(candidates), "stored": len(_arc(packet).get("hypothesis_candidates", []))}, role="adapter")
        return packet


@dataclass
class SymmetryCompletionAdapter:
    name: str = "arc_symmetry"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    symmetry_kinds: tuple[str, ...] = ("rotate_90", "rotate_180", "rotate_270", "mirror_horizontal", "mirror_vertical")

    def __post_init__(self) -> None:
        self.requires = ["arc_hypotheses"]
        self.provides = ["arc_symmetry"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        arc = _arc(packet)
        task: ArcTask = arc["task"]
        candidates = []
        for example in task.train:
            if example.output_grid is None:
                continue
            candidate = infer_transformation(example.input_grid, example.output_grid)
            if candidate is not None and candidate.kind in self.symmetry_kinds:
                candidates.append(candidate)
        added = _store_candidates(packet, key="symmetry_candidates", source=self.name, candidates=candidates)
        arc["symmetry_hypotheses"] = [candidate.describe() for candidate in arc.get("symmetry_candidates", [])]
        packet.views.append(
            {
                "name": "arc_symmetry_completion",
                "candidate_count": len(arc.get("symmetry_candidates", [])),
                "added": len(added),
            }
        )
        packet.log(self.name, {"candidates": len(candidates), "stored": len(arc.get("symmetry_candidates", []))}, role="adapter")
        return packet


@dataclass
class LineExtensionAdapter:
    name: str = "arc_line_extension"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_symmetry"]
        self.provides = ["arc_line_extension"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        arc = _arc(packet)
        task: ArcTask = arc["task"]
        candidates = []
        for example in task.train:
            if example.output_grid is None:
                continue
            candidate = infer_transformation(example.input_grid, example.output_grid)
            if candidate is not None and candidate.kind == "extend_line":
                candidates.append(candidate)
        added = _store_candidates(packet, key="line_extension_candidates", source=self.name, candidates=candidates)
        arc["line_extension_hypotheses"] = [candidate.describe() for candidate in arc.get("line_extension_candidates", [])]
        packet.views.append(
            {
                "name": "arc_line_extension",
                "candidate_count": len(arc.get("line_extension_candidates", [])),
                "added": len(added),
            }
        )
        packet.log(self.name, {"candidates": len(candidates), "stored": len(arc.get("line_extension_candidates", []))}, role="adapter")
        return packet


__all__ = [
    "ArcHypothesisAdapter",
    "LineExtensionAdapter",
    "SymmetryCompletionAdapter",
]
