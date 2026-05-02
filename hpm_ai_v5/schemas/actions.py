"""Typed action schema for validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_type: str
    value: Any = None
    confidence: float = 0.0
    valid: bool = False
    trace: dict[str, Any] = Field(default_factory=dict)
