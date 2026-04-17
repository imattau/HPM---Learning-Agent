"""
MentalChessAgent — HPM agent that uses L4 forward model and learned evaluation (SP-Chess4/5).
"""
from __future__ import annotations
import numpy as np
import chess
from typing import List, Optional, Tuple
from hfn.hfn import HFN
from hpm_ai_v2.domains.chess_hfn import ChessHFNEncoder
from hpm_ai_v2.domains.chess_transition import ChessTransitionLearner
from hpm_ai_v2.utils.chess_evaluator import ChessValueLearner

class MentalChessAgent:
    """
    Agent that uses a learned L4 forward model for move selection.
    Instead of simulating with the library, it predicts the next board mu.
    Can use a learned ChessValueLearner for evaluation.
    """
    def __init__(self, learner: ChessTransitionLearner, evaluator: Optional[ChessValueLearner] = None, name: str = "MentalAgent"):
        self.learner = learner
        self.evaluator = evaluator
        self.name = name

    def score_move(self, board: chess.Board, move: chess.Move, depth: int = 1) -> float:
        """Score a move using mental simulation (L4 deltas)."""
        node_curr = ChessHFNEncoder.board_to_node(board)
        
        # 1. Depth-1 Prediction (My move)
        pred_mu_1, pred_sq_1 = self.learner.predict(node_curr, move, board_curr=board)
        
        if depth == 1:
            if self.evaluator:
                val = self.evaluator.predict(pred_mu_1)
                return val if board.turn == chess.WHITE else 1.0 - val
            else:
                gain_1 = pred_mu_1[0] - node_curr.mu[0]
                if board.turn == chess.BLACK: gain_1 = -gain_1
                return gain_1
            
        # 2. Depth-2 Prediction (Opponent's best reply)
        b_after = board.copy()
        b_after.push(move)
        opp_moves = list(b_after.legal_moves)
        if not opp_moves:
            # Game over
            if self.evaluator:
                val = self.evaluator.predict(pred_mu_1)
                return (val if board.turn == chess.WHITE else 1.0 - val) + (1.0 if b_after.is_checkmate() else 0.0)
            else:
                gain_1 = pred_mu_1[0] - node_curr.mu[0]
                if board.turn == chess.BLACK: gain_1 = -gain_1
                return gain_1 + (100.0 if b_after.is_checkmate() else 0.0)
            
        # Hallucinate BoardNode for b_after
        s_sq = np.full(13, 0.1)
        s_sum = np.full(20, 0.1)
        node_after = HFN(mu=pred_mu_1, sigma=s_sum, inputs=[HFN(mu=pred_sq_1[i*13:(i+1)*13], sigma=s_sq) for i in range(64)])
        
        best_opp_val = -np.inf
        for opp_move in opp_moves:
            pred_mu_2, _ = self.learner.predict(node_after, opp_move, board_curr=b_after)
            
            if self.evaluator:
                val_2 = self.evaluator.predict(pred_mu_2)
                opp_val = val_2 if b_after.turn == chess.WHITE else 1.0 - val_2
            else:
                opp_gain = pred_mu_2[0] - pred_mu_1[0]
                if b_after.turn == chess.BLACK: opp_gain = -opp_gain
                opp_val = opp_gain
            
            if opp_val > best_opp_val:
                best_opp_val = opp_val
                
        # Final score
        if self.evaluator:
            # Score = 1.0 - Opponent's max win prob
            return 1.0 - best_opp_val
        else:
            my_gain = pred_mu_1[0] - node_curr.mu[0]
            if board.turn == chess.BLACK: my_gain = -my_gain
            return my_gain - best_opp_val

    def select_move(self, board: chess.Board, depth: int = 1) -> Optional[chess.Move]:
        """Select best legal move using mental simulation."""
        moves = list(board.legal_moves)
        if not moves: return None
        np.random.shuffle(moves)
        
        best_move = None
        best_score = -np.inf
        
        scored_moves = []
        for move in moves:
            score = self.score_move(board, move, depth=depth)
            scored_moves.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
