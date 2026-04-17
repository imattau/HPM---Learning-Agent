"""
ChessTransitionLearner — Learns L4 transition deltas for chess (SP-Chess3).
"""
from __future__ import annotations
import numpy as np
import chess
from typing import List, Dict, Tuple, Optional
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer
from .chess_hfn import ChessHFNEncoder

class ChessTransitionLearner:
    """
    Learns L4 transition nodes and stores them in a TieredForest.
    Aligned with HPM Pattern Dynamics and Evaluators.
    """
    def __init__(self, forest: Optional[TieredForest] = None, observer: Optional[Observer] = None):
        self.forest = forest or TieredForest()
        self.observer = observer or Observer(self.forest)
        # Context map for quick retrieval during prediction (optional optimization)
        self.context_to_id: Dict[Tuple[int, int, int], str] = {}
        self.raw_deltas: Dict[Tuple[int, int, int], List[Tuple[np.ndarray, np.ndarray]]] = {}

    def collect_triple(self, board_curr: chess.Board, move: chess.Move, board_next: chess.Board):
        # Determine capture context
        captured = board_curr.piece_at(move.to_square)
        cap_type = captured.piece_type if captured else 0
        move_key = (move.from_square, move.to_square, cap_type)
        
        # Calculate deltas
        node_curr = ChessHFNEncoder.board_to_node(board_curr)
        node_next = ChessHFNEncoder.board_to_node(board_next)
        
        summary_delta = node_next.mu - node_curr.mu
        curr_sq_vec = np.concatenate([sq.mu for sq in node_curr.inputs])
        next_sq_vec = np.concatenate([sq.mu for sq in node_next.inputs])
        squares_delta = next_sq_vec - curr_sq_vec
        
        if move_key not in self.raw_deltas:
            self.raw_deltas[move_key] = []
        self.raw_deltas[move_key].append((summary_delta, squares_delta))

    def train(self):
        """Register L4 nodes into the TieredForest."""
        count = 0
        for move_key, deltas in self.raw_deltas.items():
            avg_summary = np.mean([d[0] for d in deltas], axis=0)
            avg_squares = np.mean([d[1] for d in deltas], axis=0)
            
            # Create L4 Node
            # We store the deltas in the mu of a dedicated transition node
            # This is a simplification of HPM L4 (delta-mapping)
            mu = np.concatenate([avg_summary, avg_squares])
            node = HFN(mu=mu, sigma=np.full(mu.size, 0.1))
            node.id = f"l4_trans_{move_key[0]}_{move_key[1]}_{move_key[2]}"
            node.relation_type = "transition"
            
            self.forest.register(node)
            self.context_to_id[move_key] = node.id
            count += 1
        print(f"Learned and registered {count} L4 transition nodes in the Forest.")

    def predict(self, board_node: HFN, move: chess.Move, board_curr: Optional[chess.Board] = None) -> Tuple[np.ndarray, np.ndarray]:
        cap_type = 0
        if board_curr:
            captured = board_curr.piece_at(move.to_square)
            cap_type = captured.piece_type if captured else 0
            
        move_key = (move.from_square, move.to_square, cap_type)
        node_id = self.context_to_id.get(move_key)
        
        if node_id:
            node = self.forest.get(node_id)
            if node:
                # First 20 are summary delta
                delta_summary = node.mu[:20]
                # Rest are squares delta
                delta_squares = node.mu[20:]
                
                pred_summary = board_node.mu + delta_summary
                curr_squares = np.concatenate([sq.mu for sq in board_node.inputs])
                pred_squares = curr_squares + delta_squares
                return pred_summary, pred_squares
                
        return board_node.mu.copy(), np.concatenate([sq.mu for sq in board_node.inputs])
