"""
SocialMixin — shared forest for multi-agent pattern sharing.

Provides:
- SocialForest: a TieredForest shared across agent instances
- SocialMixin: replaces self.forest with the shared SocialForest
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever


class SocialForest:
    """
    A singleton-like shared TieredForest for multi-agent environments.

    Usage:
        shared = SocialForest(D=54, cold_dir=Path("data/social"))
        agent1 = SocialAnalogicalAgent(social_forest=shared)
        agent2 = SocialAnalogicalAgent(social_forest=shared)
    """

    def __init__(
        self,
        D: int,
        cold_dir: Path = Path("data/knowledge_base/social"),
        hot_cap: int = 10_000,
    ) -> None:
        self.forest = TieredForest(D=D, cold_dir=cold_dir, hot_cap=hot_cap)
        self._agents: List[Any] = []

    def register_agent(self, agent: Any) -> None:
        self._agents.append(agent)

    def broadcast(self, node: HFN, source_agent: Any, initial_weight: float = 0.1) -> None:
        """Share a node from source_agent to all other agents in the pool."""
        for agent in self._agents:
            if agent is source_agent:
                continue
            # If the agent has social capabilities, use receive_pattern
            if hasattr(agent, "receive_pattern"):
                agent.receive_pattern(node.id, node)
            elif node.id not in agent.forest:
                agent.observer.register(node, protected=False, initial_weight=initial_weight)


class SocialMixin:
    """
    Mixin that replaces the agent's private forest with a shared SocialForest.

    Python MRO note: SocialMixin.__init__ must be called AFTER BaseHFNAgent.__init__
    so that self.forest already exists when we replace it.  Using cooperative
    super().__init__(**kwargs) achieves this when class order is correct, e.g.:

        class SocialAnalogicalAgent(BaseHFNAgent, L2MacroMixin, SocialMixin): ...

    After __init__, self.forest, self.retriever and self.observer all point to
    the shared forest infrastructure.
    """

    def __init__(
        self,
        social_forest: Optional[SocialForest] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if social_forest is not None:
            # Replace private forest with shared one
            self.forest = social_forest.forest
            self._social_forest = social_forest

            # Rebuild retriever + observer on the shared forest
            target_slice = slice(self.s_dim + self.dim, self.m_dim)
            self.retriever = GoalConditionedRetriever(
                self.forest,
                target_slice=target_slice,
                target_weight=50.0,
                weight_provider=lambda nid: self.observer.get_weight(nid),
            )
            self.observer = Observer(
                forest=self.forest,
                retriever=self.retriever,
                tau=getattr(self, "_tau_init", 0.5),
                node_use_diag=True,
                compression_cooccurrence_threshold=2,
            )
            social_forest.register_agent(self)
            # Priors were injected into the private forest before it was replaced.
            # Re-inject into the shared forest if it is still empty.
            if len(self.forest) == 0:
                self._inject_blank_priors()
        else:
            self._social_forest = None

        # Social memory: store patterns learned from peers
        self._social_memory: Dict[str, HFN] = {}

    def share_pattern(self, name: str) -> None:
        """Broadcast a named pattern to all agents in the social pool."""
        if self._social_forest is None:
            return
        node = self.patterns.get(name)
        if node is not None:
            self._social_forest.broadcast(node, self)

    def exchange_patterns(self) -> None:
        """Broadcast all local patterns to peers in the shared forest."""
        for name in list(self.patterns.keys()):
            self.share_pattern(name)

    def _try_social(
        self,
        inputs: List[Any],
        outputs: List[Any],
    ) -> Optional[List[HFN]]:
        """
        Strategy: try patterns received from social peers (in social memory).
        """
        for name, node in self._social_memory.items():
            code = self.renderer.render(node)
            results, errors = self.executor.run_batch(code, inputs)
            if self._check_outputs(results, outputs):
                return [node]
        return None

    def receive_pattern(self, name: str, node: HFN) -> None:
        """Accept a pattern from a peer agent into social memory."""
        self._social_memory[name] = node
        if node.id not in self.forest:
            self.observer.register(node, protected=False, initial_weight=0.1)
