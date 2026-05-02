"""Typed output schema for validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OutputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: Any = None
    action_type: str = "unknown"
    confidence: float = 0.0
    valid: bool = False
    trace: dict[str, Any] = Field(default_factory=dict)
