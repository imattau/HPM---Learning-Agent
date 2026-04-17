"""
ChessRenderer — specialized renderer for chess tactic discovery (SP-Chess2).
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.base_renderer import Renderer

if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hpm_ai_v2.domains.chess_domain import ChessDomainConfig

class ChessRenderer(Renderer):
    """
    Translates HFN nodes into executable Python code for chess.
    Input: (board, move) as a tuple in `inp`.
    """
    def __init__(self, config: ChessDomainConfig):
        self.config = config

    def render(self, node: HFN) -> str:
        """Render a sequence of operations into a stack-based execution."""
        lines = [
            "import chess",
            "board, move = inp[0], inp[1]",
            "def get_piece_value(piece):",
            "    if piece is None: return 0.0",
            "    values = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 100.0}",
            "    return values[piece.piece_type]",
            "def material_balance(board):",
            "    white = sum(get_piece_value(board.piece_at(s)) for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == chess.WHITE)",
            "    black = sum(get_piece_value(board.piece_at(s)) for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == chess.BLACK)",
            "    return white - black",
            "stack = [0.0]",
            "def push(v): stack.append(float(v))",
            "def pop(): return stack.pop() if len(stack) > 1 else stack[0]",
            "def top(): return stack[-1]"
        ]
        lines.extend(self._render_node_hierarchical(node))
        lines.append("res = top()")
        return "\n".join(lines)

    def _render_node_hierarchical(self, node: HFN) -> List[str]:
        lines = []
        if node.relation_type == "macro":
            for child in node.inputs:
                lines.extend(self._render_node_hierarchical(child))
        else:
            concept = self._get_concept(node)
            if concept == "CHESS_MATERIAL_BALANCE":
                lines.append("push(material_balance(board))")
            elif concept == "CHESS_PIECE_VALUE_AT_TO":
                lines.append("push(get_piece_value(board.piece_at(move.to_square)))")
            elif concept == "CHESS_IS_ENEMY_AT_TO":
                lines.append("piece = board.piece_at(move.to_square)")
                lines.append("push(1.0 if piece and piece.color != board.turn else 0.0)")
            elif concept == "CHESS_SQUARE_ATTACKED_BY_ME_AT_TO":
                lines.append("push(1.0 if board.is_attacked_by(board.turn, move.to_square) else 0.0)")
            elif concept == "CHESS_SQUARE_DEFENDED_BY_ENEMY_AT_TO":
                lines.append("push(1.0 if board.is_attacked_by(not board.turn, move.to_square) else 0.0)")
            elif concept == "CHESS_PIECE_VALUE_AT_FROM":
                lines.append("push(get_piece_value(board.piece_at(move.from_square)))")
            elif concept == "CHESS_SUM_ATTACKED_VALUE":
                # Push sum of values of pieces attacked from the NEW square (simulated after move)
                # This is tricky: we need to pretend we moved.
                # Simplification: what is attacked from move.to_square by the piece that is now at from_square?
                lines.append("b_after = board.copy(); b_after.push(move)")
                lines.append("atk_sum = sum(get_piece_value(b_after.piece_at(s)) for s in b_after.attacks(move.to_square) if b_after.piece_at(s) and b_after.piece_at(s).color != board.turn)")
                lines.append("push(float(atk_sum))")
            elif concept == "OP_MUL":
                lines.append("b = pop(); a = pop(); push(a * b)")
            elif concept == "OP_ADD":
                lines.append("b = pop(); a = pop(); push(a + b)")
            elif concept == "OP_SUB":
                lines.append("b = pop(); a = pop(); push(a - b)")
            elif concept == "OP_CONST_1":
                lines.append("push(1.0)")
        return lines

    def render_function(self, node: HFN, func_name: str = "macro_func") -> str:
        code = self.render(node)
        indented = code.replace("\n", "\n    ")
        return f"def {func_name}(inp):\n    {indented}\n    return res"

    def _get_concept(self, node: HFN) -> Optional[str]:
        # Priority 1: Named prior
        for c in self.config.concepts:
            if node.id == f"prior_rule_{c}":
                return c
        
        # Priority 2: Vector representation
        start = self.config.S_DIM
        end = start + self.config.DIM
        vec = node.mu[start:end]
        if np.max(vec) > 0.5:
            idx = np.argmax(vec)
            return self.config.concepts[idx]
        return None
