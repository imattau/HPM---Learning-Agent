"""
ChessDomainConfig — Configuration for chess tactic discovery (SP-Chess2).
"""
from __future__ import annotations
from hpm_ai_v2.domains.base import DomainConfig

CHESS_CONCEPTS = [
    "CHESS_MATERIAL_BALANCE",          # Global board state
    "CHESS_PIECE_VALUE_AT_TO",         # Value of piece at move.to_square
    "CHESS_IS_ENEMY_AT_TO",            # Is piece at move.to_square an enemy?
    "CHESS_SQUARE_ATTACKED_BY_ME_AT_TO", # Is move.to_square attacked by mover?
    "CHESS_SQUARE_DEFENDED_BY_ENEMY_AT_TO", # Is move.to_square defended by enemy?
    "CHESS_PIECE_VALUE_AT_FROM",       # Value of piece at move.from_square
    "CHESS_SUM_ATTACKED_VALUE",        # Sum of values of enemy pieces attacked from TO
    "OP_MUL",                          # b = pop(); a = pop(); push(a * b)
    "OP_ADD",                          # b = pop(); a = pop(); push(a + b)
    "OP_SUB",                          # b = pop(); a = pop(); push(a - b)
    "OP_CONST_1",                      # push(1.0)
]

class ChessDomainConfig(DomainConfig):
    """
    Configuration for the chess domain.
    Used for the SP-Chess2 experiment.
    """
    def __init__(self, s_dim: int = 20):
        super().__init__(concepts=CHESS_CONCEPTS, s_dim=s_dim)
        self.concept_idx = {c: i for i, c in enumerate(self.concepts)}
