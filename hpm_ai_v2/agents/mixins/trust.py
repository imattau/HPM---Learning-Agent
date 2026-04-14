"""
Trust and Reputation Mixins for Social HPM Agents.
Implements Experiment SP69: Higher-order social structures in pattern fields.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hpm_ai_v2.agents.mixins.social import SocialForest


class TrustMixin:
    """
    Local trust tracking for peer agents.
    Updates trust based on the success/failure of imported macros.
    """
    def __init__(self, trust_threshold: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        self.trust_threshold = trust_threshold

    def update_trust(self, agent_id: str, success: bool):
        """Increase or decrease trust based on outcome."""
        delta = 0.1 if success else -0.1
        self.trust_scores[agent_id] = max(0.0, min(1.0, self.trust_scores[agent_id] + delta))

    def should_import(self, agent_id: str) -> bool:
        """Check if a peer is trustworthy enough to import from."""
        return self.trust_scores[agent_id] >= self.trust_threshold

    def get_trust(self, agent_id: str) -> float:
        return self.trust_scores[agent_id]

    def weight_blackboard_entry(self, agent_id: str) -> float:
        """Weight blackboard entries by the trust of the reporter."""
        return self.trust_scores[agent_id]


class ReputationMixin:
    """
    Socially shared reputation tracking.
    Broadcasts trust scores to the shared forest and aggregates reports from others.
    """
    def __init__(self, agent_id: str, social_forest: SocialForest, **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.social_forest = social_forest
        self.reputation: Dict[str, float] = defaultdict(lambda: 0.5)

    def broadcast_trust(self, target_agent_id: str, trust_score: float):
        """Store local trust for a peer as an HFN node in the shared forest."""
        node_id = f"trust_{target_agent_id}_by_{self.agent_id}"
        # Encoding: mu = [trust_score, timestamp, 0, 0]
        mu = np.zeros(self.social_forest.forest._D)
        mu[0] = trust_score
        mu[1] = time.time()
        
        node = HFN(
            mu=mu,
            sigma=np.ones(self.social_forest.forest._D) * 0.1,
            id=node_id,
            use_diag=True
        )
        # Register in social forest (might overwrite previous report by same agent)
        self.social_forest.forest.register(node)

    def update_reputation(self):
        """Aggregate trust reports from others to compute peer reputation."""
        reports = defaultdict(list)
        for node in self.social_forest.forest.active_nodes():
            if node.id.startswith("trust_") and "_by_" in node.id:
                parts = node.id.split("_")
                # Format: trust_{target}_by_{reporter}
                target = parts[1]
                reporter = parts[3]
                if reporter != self.agent_id:
                    reports[target].append(node.mu[0])
        
        for target, values in reports.items():
            if values:
                self.reputation[target] = float(np.median(values))

    def should_exchange_with(self, agent_id: str) -> bool:
        """Check reputation before initiating exchange."""
        return self.reputation[agent_id] >= 0.3


class DomainSpecificTrustMixin:
    """
    Tracks trust per domain or macro type.
    Enables trusting an agent for strings but not for integers.
    """
    def __init__(self, trust_threshold: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        # trust[agent_id][domain] = score
        self.domain_trust: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.5)
        )
        self.trust_threshold = trust_threshold

    def update_domain_trust(self, agent_id: str, domain: str, success: bool):
        delta = 0.1 if success else -0.1
        self.domain_trust[agent_id][domain] = max(
            0.0, min(1.0, self.domain_trust[agent_id][domain] + delta)
        )

    def should_import_from_domain(self, agent_id: str, domain: str) -> bool:
        return self.domain_trust[agent_id][domain] >= self.trust_threshold

    def get_domain_trust(self, agent_id: str, domain: str) -> float:
        return self.domain_trust[agent_id][domain]
