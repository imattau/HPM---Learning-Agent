"""
HFNStateTransitionModel — stores forward model deltas as HFN nodes.
Provides fractal uniformity by representing transition logic as persistent HFN nodes.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Any, Optional, TYPE_CHECKING
from hfn.tiered_forest import TieredForest
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hpm_ai_v2.domains.base import DomainConfig

class HFNStateTransitionModel:
    """
    A forward model that learns state transitions (deltas) and stores them
    as HFN nodes in a dedicated TieredForest.
    """
    def __init__(self, config: "DomainConfig", cold_dir: Optional[str | Path] = None):
        self.config = config
        self.delta_forest = TieredForest(
            D=config.S_DIM,
            cold_dir=cold_dir or Path("data/knowledge_base/forward_model_deltas"),
            forest_id="forward_model"
        )
        self.alpha = 1.0  # EMA learning rate

    def _get_delta_node(self, node_id: str) -> HFN:
        """Fetch or create an HFN node to store the delta for a given primitive/macro."""
        delta_id = f"delta:{node_id}"
        node = self.delta_forest.get(delta_id)
        if node is None:
            # Initialize with zero delta and flat sigma
            node = HFN(
                mu=np.zeros(self.config.S_DIM),
                sigma=np.ones(self.config.S_DIM),
                id=delta_id,
                use_diag=True
            )
            self.delta_forest.register(node)
        return node

    def record_path(self, path: List[HFN], state_sequence: List[np.ndarray]) -> None:
        """
        Record a sequence of transitions.
        path[i] caused the transition state_sequence[i] -> state_sequence[i+1].
        """
        if len(state_sequence) != len(path) + 1:
            return

        for i, node in enumerate(path):
            delta = state_sequence[i+1] - state_sequence[i]
            delta_node = self._get_delta_node(node.id)
            # Update delta node mu via Exponential Moving Average
            delta_node.mu = (1.0 - self.alpha) * delta_node.mu + self.alpha * delta

    def predict(self, current_state: np.ndarray, node: HFN) -> np.ndarray:
        """
        Predict the next state after applying a node.
        Recursively decomposes macros to apply constituent deltas.
        """
        if node.relation_type == "macro" and node.inputs:
            state = current_state.copy()
            for constituent in node.inputs:
                state = self.predict(state, constituent)
            return state

        delta_node = self.delta_forest.get(f"delta:{node.id}")
        if delta_node is None:
            return current_state
        return current_state + delta_node.mu

    def predict_path(self, start_state: np.ndarray, path: List[HFN]) -> np.ndarray:
        """Predict the state after a sequence of nodes."""
        state = start_state.copy()
        for node in path:
            state = self.predict(state, node)
        return state

    def save_state(self) -> None:
        """Persist delta nodes to cold storage."""
        self.delta_forest.save_to_cold()
