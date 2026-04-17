"""
ChessOracle — computes a 20-D state vector for chess tactic discovery (SP-Chess2).
"""
from __future__ import annotations
import numpy as np
import chess
from typing import Any, List, Optional, TYPE_CHECKING
from .base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.chess_domain import ChessDomainConfig

class ChessOracle(BaseOracle):
    """
    Computes a fixed-D empirical state vector for chess tactics.
    Encodes correlations with material gain and piece values.
    """
    def __init__(self, config: ChessDomainConfig):
        self.config = config

    def _get_piece_value(self, piece: Optional[chess.Piece]) -> float:
        if piece is None: return 0.0
        values = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 100.0}
        return values[piece.piece_type]

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
        inputs: Optional[List[Any]] = None,
    ) -> np.ndarray:
        s_dim = self.config.S_DIM
        s = np.zeros(s_dim)
        
        valid_indices = [i for i, e in enumerate(errors) if e is None]
        if not valid_indices or len(valid_indices) < 2:
            s[0] = 0.0
            s[9] = 1.0
            return s
        
        s[0] = 1.0
        valid_outputs = np.array([float(outputs[i]) for i in valid_indices])
        
        # Dim 3: Average score
        s[3] = float(np.mean(valid_outputs))
            
        # [NEW] Correlations with board features
        if inputs is not None:
            v_inputs = [inputs[i] for i in valid_indices] # (board, move)
            
            # Simple correlations with feature-like board/move props
            def get_corr(a, b):
                if np.std(a) < 1e-6 or np.std(b) < 1e-6: return 0.0
                return float(np.corrcoef(a, b)[0, 1])
            
            # Feature 4: Value of piece at destination
            dest_vals = np.array([self._get_piece_value(b.piece_at(m.to_square)) for b, m in v_inputs])
            s[4] = 10.0 * get_corr(valid_outputs, dest_vals)
            
            # Feature 5: Is destination square attacked by mover?
            attack_vals = np.array([1.0 if b.is_attacked_by(b.turn, m.to_square) else 0.0 for b, m in v_inputs])
            s[5] = 10.0 * get_corr(valid_outputs, attack_vals)
        
        # Code structure flags (dimensions 10+)
        if code:
            s[10] = 1.0 if 'material_balance' in code else 0.0
            s[11] = 1.0 if 'get_piece_value' in code else 0.0
            s[12] = 1.0 if 'is_attacked_by' in code else 0.0
            s[13] = 1.0 if 'inp[1]' in code else 0.0
        
        return s
