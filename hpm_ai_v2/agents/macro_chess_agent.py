"""
MacroChessAgent — HPM agent using L5 temporal macros (SP-Chess6).
Extends planning with abstraction using HFN-native evaluators.
"""
from __future__ import annotations
import numpy as np
import chess
from typing import List, Optional, Tuple
from hpm_ai_v2.agents.mental_chess_agent import MentalChessAgent
from hpm_ai_v2.domains.chess_hfn import ChessHFNEncoder
from hpm_ai_v2.domains.chess_transition import ChessTransitionLearner
from hpm_ai_v2.utils.chess_evaluator import ChessValueLearner
from hpm_ai_v2.utils.chess_macro_learner import ChessMacroLearner

class MacroChessAgent(MentalChessAgent):
    """
    Agent that uses learned L5 macros to bypass search.
    Follows HPM layer 2/3: Pattern dynamics (Stabilization) and Evaluators (Score).
    """
    def __init__(self, learner: ChessTransitionLearner, evaluator: ChessValueLearner, macro_learner: ChessMacroLearner, name: str = "MacroAgent"):
        super().__init__(learner, evaluator=evaluator, name=name)
        self.macro_learner = macro_learner

    def select_move(self, board: chess.Board, depth: int = 2) -> Optional[chess.Move]:
        """Select best legal move, using Meta-Pattern Rule over Macros and Search."""
        mu_curr = ChessHFNEncoder.get_board_summary(board)
        val_curr = self.evaluator.predict(mu_curr)
        if board.turn == chess.BLACK: val_curr = 1.0 - val_curr
        
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None
        legal_moves_uci = [m.uci() for m in legal_moves]
        
        # 1. Macro Proposal (Layer 5)
        best_macro = self.macro_learner.get_best_macro(mu_curr, legal_moves_uci)
        
        # 2. Search Proposal (Layer 4)
        best_search_move = None
        best_search_score = -np.inf
        for move in legal_moves:
            score = self.score_move(board, move, depth=depth)
            if score > best_search_score:
                best_search_score = score
                best_search_move = move

        # 3. Meta-Pattern Rule: Selective advantage between Macro and Search
        # In HPM, a well-learned macro should have a high score and displace search.
        if best_macro and best_macro.hpm_score() > best_search_score:
            # print(f"  [Debug Agent {self.name}] Macro Dominates: {best_macro.moves[0]} (Score: {best_macro.hpm_score():.4f})")
            final_move = chess.Move.from_uci(best_macro.moves[0])
        else:
            final_move = best_search_move

        # 4. Discovery (Stabilization / New Pattern)
        if final_move:
            # Predict outcome to get delta_v
            node_curr = ChessHFNEncoder.board_to_node(board)
            pred_mu_1, _ = self.learner.predict(node_curr, final_move, board_curr=board)
            val_after = self.evaluator.predict(pred_mu_1)
            if board.turn == chess.BLACK: val_after = 1.0 - val_after
            
            delta_v = val_after - val_curr
            # Stabilize or Discover
            self.macro_learner.discover(mu_curr, final_move.uci(), delta_v)
            
        return final_move
