from __future__ import annotations
from typing import Protocol
import numpy as np


class LevelEncoder(Protocol):
    """Domain-agnostic encoder interface for one level of a StructuredOrchestrator.

    feature_dim: Dimension of each returned vector.
    max_steps_per_obs: Expected list length from encode(). None = variable
        (e.g. L2 per-object pair); 1 = fixed (L1, L3).
    """
    feature_dim: int
    max_steps_per_obs: int | None

    def encode(
        self,
        observation,
        epistemic: tuple[float, float] | None,
    ) -> list[np.ndarray]:
        """Encode an observation into a list of feature vectors.

        Args:
            observation: Domain-specific input (e.g. (input_grid, output_grid) for ARC).
            epistemic: (weight, epistemic_loss) from the level below; None for L1.

        Returns:
            List of numpy arrays each of shape (feature_dim,).
            Length is 1 for L1/L3; N for L2 (one per matched object pair).
        """
        ...
