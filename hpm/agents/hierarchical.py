from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from hpm.agents.agent import Agent


@dataclass
class LevelBundle:
    """Structured inter-level signal: belief + confidence from one Level 1 agent."""
    agent_id: str
    mu: np.ndarray        # shape (D,) — top pattern mean
    weight: float         # top pattern's store weight
    epistemic_loss: float # running epistemic loss for that pattern


def extract_bundle(agent: Agent) -> LevelBundle:
    """Extract a structured bundle from an agent's current state.

    Reads the top-weighted pattern from the agent's store.
    If the store is empty (only possible with manually-cleared stores in tests),
    returns a zero bundle with maximum uncertainty (epistemic_loss=1.0).
    """
    feature_dim = agent.config.feature_dim
    records = agent.store.query(agent.agent_id)

    if not records:
        return LevelBundle(
            agent_id=agent.agent_id,
            mu=np.zeros(feature_dim),
            weight=0.0,
            epistemic_loss=1.0,
        )

    top_pattern, top_weight = max(records, key=lambda r: r[1])
    epistemic_loss = agent.epistemic._running_loss.get(top_pattern.id, 0.0)

    return LevelBundle(
        agent_id=agent.agent_id,
        mu=top_pattern.mu.copy(),
        weight=float(top_weight),
        epistemic_loss=float(epistemic_loss),
    )


def encode_bundle(bundle: LevelBundle) -> np.ndarray:
    """Concatenate [mu, weight, epistemic_loss] into a single observation vector.

    Output shape: (D + 2,) where D = len(bundle.mu).
    This becomes the raw observation fed to Level 2 agents.
    """
    return np.concatenate([bundle.mu, [bundle.weight, bundle.epistemic_loss]])
