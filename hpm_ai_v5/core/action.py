"""Typed core action contract for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pattern import Pattern
from .sequence import PatternSequence
from .state import State


@dataclass(frozen=True, slots=True)
class Action:
    """Structured output from the core."""

    action_type: str
    value: Any
    confidence: float
    selected_pattern: Pattern | None = None
    selected_sequence: PatternSequence | None = None
    selected_view: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    forecast: State | None = None
