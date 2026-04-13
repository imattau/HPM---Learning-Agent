"""
hpm_ai_v2 — Domain-agnostic HPM agent layer.

Sits above hfn/ and provides reusable mixin-based agents implementing the
Hierarchical Pattern Modelling framework across multiple abstraction levels.
"""
from hpm_ai_v2.agents.agents import (
    InducedSchemaAgent,
    ImaginativeAgent,
    AnalogicalAgent,
    SocialAnalogicalAgent,
)

__all__ = [
    "InducedSchemaAgent",
    "ImaginativeAgent",
    "AnalogicalAgent",
    "SocialAnalogicalAgent",
]
