"""
ChessHFN — Fractal Board/Move Representation for SP-Chess3.
Translates python-chess states to HFN nodes.
"""
from __future__ import annotations
import numpy as np
import chess
from typing import List, Optional
from hfn.hfn import HFN

class ChessHFNEncoder:
    """
    Encodes chess components as HFN nodes.
    Square: 13-D one-hot (empty, 12 pieces).
    Board: 64 squares + 20-D summary mu.
    Move: From/To/Promotion info.
    """
    
    @staticmethod
    def piece_to_vec(piece: Optional[chess.Piece]) -> np.ndarray:
        v = np.zeros(13)
        if piece is None:
            v[0] = 1.0
        else:
            # 1-6 White P,N,B,R,Q,K; 7-12 Black P,N,B,R,Q,K
            idx = piece.piece_type
            if piece.color == chess.BLACK:
                idx += 6
            v[idx] = 1.0
        return v

    @staticmethod
    def square_to_node(board: chess.Board, square: chess.Square) -> HFN:
        piece = board.piece_at(square)
        mu = ChessHFNEncoder.piece_to_vec(piece)
        sigma = np.full_like(mu, 0.1)
        node = HFN(mu=mu, sigma=sigma)
        node.id = f"square_{chess.square_name(square)}"
        return node

    @staticmethod
    def get_board_summary(board: chess.Board) -> np.ndarray:
        """20-D summary vector for the board node."""
        mu = np.zeros(20)
        # 0: Material balance (scaled)
        def get_val(p):
            if p is None: return 0.0
            vals = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0.0} 
            return vals[p.piece_type]
        
        white = sum(get_val(board.piece_at(s)) for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == chess.WHITE)
        black = sum(get_val(board.piece_at(s)) for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == chess.BLACK)
        mu[0] = (white - black) / 10.0
        
        # 1-12: Counts of each piece type
        for s in chess.SQUARES:
            p = board.piece_at(s)
            if p:
                idx = p.piece_type
                if p.color == chess.BLACK: idx += 6
                mu[idx] += 0.1
                
        return mu

    @staticmethod
    def board_to_node(board: chess.Board) -> HFN:
        squares = [ChessHFNEncoder.square_to_node(board, s) for s in chess.SQUARES]
        mu_summary = ChessHFNEncoder.get_board_summary(board)
        sigma = np.full_like(mu_summary, 0.1)
        node = HFN(mu=mu_summary, sigma=sigma, inputs=squares)
        node.relation_type = "chess_board"
        node.id = f"board_{board.fen()[:15]}"
        return node

    @staticmethod
    def move_to_node(move: chess.Move) -> HFN:
        mu = np.zeros(3)
        mu[0] = move.from_square / 64.0
        mu[1] = move.to_square / 64.0
        mu[2] = (move.promotion if move.promotion else 0) / 6.0
        sigma = np.full_like(mu, 0.1)
        node = HFN(mu=mu, sigma=sigma)
        node.relation_type = "chess_move"
        node.id = f"move_{move.uci()}"
        return node
