"""ARC solver entry point."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.packet import Packet
from .pipeline import ArcPipeline


@dataclass
class ArcSolver:
    pipeline: ArcPipeline = field(default_factory=ArcPipeline)

    def solve(self, raw_task) -> Packet:
        return self.pipeline.run(raw_task)
