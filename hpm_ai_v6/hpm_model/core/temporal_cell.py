from __future__ import annotations

from typing import List

from pydantic import ConfigDict, Field

from hpm_ai_v6.hpm_model.core.cell import Cell


class TemporalCell(Cell):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cause: Cell
    effect: Cell
    onset_weight: float = 0.0
    duration_weight: float = 1.0
    concurrent: List["TemporalCell"] = Field(default_factory=list)
    lapsed: bool = False
