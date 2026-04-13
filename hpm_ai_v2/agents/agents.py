"""
Concrete HPM agents built from BaseHFNAgent + mixins.

InducedSchemaAgent   — Base + L2 + L3
ImaginativeAgent     — Base + L2 + L4
AnalogicalAgent      — Base + L2
SocialAnalogicalAgent — Base + L2 + L4 + Social + Recombination
"""
from __future__ import annotations

from typing import Any, List, Optional

from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l2_macro import L2MacroMixin
from hpm_ai_v2.agents.mixins.l3_relational import L3RelationalMixin
from hpm_ai_v2.agents.mixins.l4_forward import L4ForwardModelMixin
from hpm_ai_v2.agents.mixins.social import SocialMixin
from hpm_ai_v2.agents.mixins.recombination import RecombinationMixin


class InducedSchemaAgent(L3RelationalMixin, L2MacroMixin, BaseHFNAgent):
    """
    HPM L1–L3 agent.

    Capabilities:
    - Primitive retrieval and exact matching (L1)
    - Macro composition from solved paths (L2)
    - Meta-schema discovery across solved paths (L3)
    - BFS search over pattern space
    - Meta-strategy controller (L5 lightweight)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Register built-in strategies
        self.add_strategy("exact", self._try_exact)
        self.add_strategy("decompose", self._try_decompose)
        self.add_strategy("bfs", self._try_bfs)


class ImaginativeAgent(L4ForwardModelMixin, L2MacroMixin, BaseHFNAgent):
    """
    HPM L1–L4 agent.

    Extends InducedSchemaAgent with forward model for mental simulation.

    Capabilities:
    - All InducedSchemaAgent capabilities
    - Forward model learns per-node state transitions (L4)
    - Imaginative BFS: zero oracle calls during search
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_strategy("exact", self._try_exact)
        self.add_strategy("decompose", self._try_decompose)
        self.add_strategy("imagine", self._try_imagine)
        self.add_strategy("bfs", self._try_bfs)


class AnalogicalAgent(L2MacroMixin, BaseHFNAgent):
    """
    HPM L1–L2 agent — minimal analogy-capable agent.

    Capabilities:
    - Primitive retrieval and exact matching (L1)
    - Macro composition (L2)
    - BFS search
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_strategy("exact", self._try_exact)
        self.add_strategy("decompose", self._try_decompose)
        self.add_strategy("bfs", self._try_bfs)


class SocialAnalogicalAgent(
    RecombinationMixin,
    SocialMixin,
    L4ForwardModelMixin,
    L2MacroMixin,
    BaseHFNAgent,
):
    """
    Full HPM agent: L1–L5 + social sharing + recombination.

    MRO (left to right):
      RecombinationMixin -> SocialMixin -> L4ForwardModelMixin ->
      L2MacroMixin -> BaseHFNAgent

    Capabilities:
    - All ImaginativeAgent capabilities
    - Social pattern sharing via SocialForest
    - Cross-domain recombination (analogical insight)
    - Meta-strategy controller (L5)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_strategy("exact", self._try_exact)
        self.add_strategy("decompose", self._try_decompose)
        self.add_strategy("social", self._try_social)
        self.add_strategy("recombine", self._try_recombine)
        self.add_strategy("imagine", self._try_imagine)
        self.add_strategy("bfs", self._try_bfs)
