"""
ChessValueLearner — HFN-based board evaluation for chess (SP-Chess5).
Learns to predict game outcomes from fractal board summaries.
"""
from __future__ import annotations
import numpy as np
import chess
from typing import List, Tuple, Optional
from hfn.hfn import HFN
from hpm_ai_v2.domains.chess_hfn import ChessHFNEncoder

class ChessValueLearner:
    """
    Evaluates board states using an HFN node trained on game outcomes.
    Learns mu (weights) to predict 1.0 (White win), 0.0 (Black win), 0.5 (Draw).
    """
    def __init__(self, s_dim: int = 20, learning_rate: float = 0.01):
        self.s_dim = s_dim
        self.lr = learning_rate
        # The 'weights' of our value function are stored in the mu of an HFN node.
        # It takes a 20-D board summary and predicts a scalar.
        self.value_node = HFN(mu=np.random.normal(0, 0.1, s_dim), sigma=np.full(s_dim, 0.1))
        self.value_node.id = "chess_evaluator"

    def predict(self, board_mu: np.ndarray) -> float:
        """Predict win probability [0, 1] from board summary mu."""
        # Simple dot product + sigmoid for probability prediction
        z = np.dot(board_mu, self.value_node.mu)
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, board_mu: np.ndarray, outcome: float):
        """Update value node weights based on final game outcome."""
        pred = self.predict(board_mu)
        error = outcome - pred
        
        # Delta rule: Adjust mu to minimize prediction error
        # gradient of sigmoid is pred * (1 - pred)
        grad = error * pred * (1.0 - pred)
        self.value_node.mu += self.lr * grad * board_mu

    def evaluate_board(self, board: chess.Board) -> float:
        """Convenience method for chess.Board."""
        node = ChessHFNEncoder.board_to_node(board)
        val = self.predict(node.mu)
        # Perspective shift: return score relative to moving player
        if board.turn == chess.WHITE:
            return val
        else:
            return 1.0 - val
