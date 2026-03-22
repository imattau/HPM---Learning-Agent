from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class LevelBundle:
    """Structured inter-level signal: belief + confidence from one Level 1 agent."""
    agent_id: str
    mu: np.ndarray        # shape (D,) — top pattern mean
    weight: float         # top pattern's store weight
    epistemic_loss: float # running epistemic loss for that pattern


def encode_bundle(bundle: LevelBundle) -> np.ndarray:
    """Concatenate [mu, weight, epistemic_loss] into a single observation vector.

    Output shape: (D + 2,) where D = len(bundle.mu).
    This becomes the raw observation fed to Level 2 agents.
    """
    return np.concatenate([bundle.mu, [bundle.weight, bundle.epistemic_loss]])
