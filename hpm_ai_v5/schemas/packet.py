"""Shared pydantic packet model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Packet(BaseModel):
    """Shared packet across adapters, polygraphs, core, and agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    raw_input: Any = Field(alias="raw")
    goal: Any = None
    context: dict[str, Any] = Field(default_factory=dict)

    clean: Any = None
    clean_text: str | None = None
    tokens: list[Any] = Field(default_factory=list)
    entities: list[Any] = Field(default_factory=list)
    relations: list[Any] = Field(default_factory=list)

    states: list[Any] = Field(default_factory=list)
    deltas: list[Any] = Field(default_factory=list)
    views: list[Any] = Field(default_factory=list)

    core_action: Any = None
    core_actions: list[Any] = Field(default_factory=list)
    draft_output: Any = None
    candidate_outputs: list[Any] = Field(default_factory=list)
    validated_output: Any = None
    final_output: Any = None

    state: dict[str, Any] = Field(default_factory=dict)
    agent_trace: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)

    def log(self, actor: str, detail: Any | None = None, *, role: str | None = None) -> None:
        entry = {"actor": actor, "agent": actor, "adapter": actor, "detail": detail}
        if role is not None:
            entry["role"] = role
        self.trace.append(entry)

    @property
    def raw(self) -> Any:
        return self.raw_input

    @raw.setter
    def raw(self, value: Any) -> None:
        self.raw_input = value
