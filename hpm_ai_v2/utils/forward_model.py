"""
StateTransitionModel — learns per-node state deltas for forward (imaginative) planning.

Ported from experiment_generative_forward_model.py.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hfn.hfn import HFN


class StateTransitionModel:
    """
    Learns per-node state deltas from observed execution paths.

    During training, each solved path is stepped through one node at a time.
    The per-step deltas (state[i+1] - state[i]) are stored per node_id.
    Prediction averages the recorded deltas.

    For macro nodes: prediction is recursive — sequentially predict through constituents.
    """

    def __init__(self) -> None:
        self._deltas: dict[str, list[np.ndarray]] = defaultdict(list)
        self._n_paths = 0

    def record_path(
        self,
        path: List["HFN"],
        state_sequence: List[np.ndarray],
    ) -> None:
        """
        Record per-step deltas from a fully-stepped execution path.

        Args:
            path: list of HFN nodes (length k)
            state_sequence: oracle states at each prefix
                            [baseline, after_1, ..., after_k], length == len(path)+1
        """
        if len(state_sequence) != len(path) + 1:
            return
        for i, node in enumerate(path):
            delta = state_sequence[i + 1] - state_sequence[i]
            self._deltas[node.id].append(delta)
        self._n_paths += 1

    def predict(self, current_state: np.ndarray, node: "HFN") -> np.ndarray:
        """Predict state after applying one node. Recurses for macro nodes."""
        if node.relation_type == "macro" and node.inputs:
            state = current_state.copy()
            for constituent in node.inputs:
                state = self.predict(state, constituent)
            return state
        if node.id in self._deltas:
            mean_delta = np.mean(self._deltas[node.id], axis=0)
            return current_state + mean_delta
        return current_state  # unknown node: no predicted change

    def predict_path(
        self,
        start_state: np.ndarray,
        path: List["HFN"],
    ) -> np.ndarray:
        """Compose delta predictions along a sequence of nodes."""
        state = start_state.copy()
        for node in path:
            state = self.predict(state, node)
        return state

    def prediction_error(
        self,
        start_state: np.ndarray,
        path: List["HFN"],
        true_final_state: np.ndarray,
    ) -> float:
        """Mean absolute error of predicted vs true final state."""
        predicted = self.predict_path(start_state, path)
        return float(np.mean(np.abs(predicted - true_final_state)))

    @property
    def n_nodes_known(self) -> int:
        return len(self._deltas)

    @property
    def n_paths(self) -> int:
        return self._n_paths
